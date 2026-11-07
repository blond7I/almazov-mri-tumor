"""Единый анализ одного среза: analyze(image) -> результат для заключения, интерфейса врача и пакетной проверки.

Все решения (порог контура, полоса неуверенной границы, зоны дополнительного просмотра, температура калибровки)
берутся из calib.json, подобранного на валидации; если файла нет — используются значения этапа 1.
"""
import glob
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .common import CLASSES, NO_TUMOR, prep, prep_geometry, clean_mask
from . import nets

DEFAULT_CALIB = dict(
    temperature=1.0,          # калибровка вероятностей классификатора (подбирается на валидации)
    screen_thr=0.5,           # «есть опухоль», если 1 - P(нет опухоли) >= порога (этап 1)
    contour_thr=0.5,          # контур = среднее карт моделей >= порога (этап 1)
    min_area=30,              # пятна меньше 30 пикселей сетки 256 не показываем (этап 1)
    band_lo=0.2,              # внешняя граница полосы неуверенности (подбирается conformal risk control)
    band_hi=0.8,              # внутренняя граница («уверенно опухоль»)
    review_max=0.3,           # зона дополнительного просмотра: хотя бы одна модель >= review_max ...
    review_mean=0.5,          # ... при среднем < review_mean (контур там не нарисован)
    review_min_area=30,
    low_conf=0.6,             # максимальная вероятность типа ниже — «низкая уверенность»
)


def _gray(img):
    if isinstance(img, (str, Path)):
        img = np.array(Image.open(img))
    img = np.asarray(img)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        rgb = img.astype(np.float32)
        colorfulness = float(np.abs(rgb - rgb.mean(2, keepdims=True)).mean())
        g = np.array(Image.fromarray(img.astype(np.uint8)).convert('L'))
        return g, colorfulness
    if img.dtype != np.uint8:      # 16-битные снимки (DICOM, PNG16): растягиваем по перцентилям
        lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
        img = (np.clip((img - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    return img, 0.0


def head_mask(x256):
    """Грубая маска головы на срезе (для «% площади»): порог Оцу по ненулевым пикселям, крупнейший компонент, заливка дыр."""
    v = x256[x256 > 5]
    if v.size < 100:
        return np.ones_like(x256, np.uint8)
    t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (x256 > max(10, t * 0.5)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if k <= 1:
        return np.ones_like(x256, np.uint8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    ff = m.copy()
    h, w = m.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return m | (1 - ff)


def feret(mask, scale):
    """Наибольший диаметр (по выпуклой оболочке) и наибольшая ширина перпендикулярно ему — в пикселях исходного снимка."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0.0, 0.0
    pts = cv2.convexHull(np.concatenate(cs)).reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return scale, scale
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    ax = (pts[j] - pts[i]) / (d[i, j] + 1e-6)
    perp = np.array([-ax[1], ax[0]])
    w = float((pts @ perp).max() - (pts @ perp).min())
    return float(d[i, j] + 1) * scale, (w + 1) * scale


@dataclass
class Models:
    cls: list = field(default_factory=list)     # [(tag, net)]
    seg: list = field(default_factory=list)
    calib: dict = field(default_factory=lambda: dict(DEFAULT_CALIB))
    version: str = ''
    input_models: object = None


def load_models(repo_dir, cache_dir=None, device='cpu', use_stage2=True):
    """Модели этапа 1 (из weights/ в репозитории) + новые обучения этапа 2 (weights/stage2/*.pt), если они есть."""
    repo = Path(repo_dir)
    W = repo / 'weights'
    cache = Path(cache_dir or (repo / '.cache'))
    cache.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    M = Models()
    man = json.load(open(W / 'cls_models.json'))
    for m in man['models']:
        f = nets.assemble([W / p for p in m['parts']], cache / f'cls_{m["tag"]}.pt', sha256=m['sha256'])
        net = nets.make_cls(m['arch']); net.load_state_dict(nets.load_state(f))
        M.cls.append(('stage1_' + m['tag'], net.to(dev).eval()))
    f = nets.assemble([W / f'seg_v3_best.pt.part{i}' for i in range(5)], cache / 'seg_stage1.pt', sha_prefix='ac9d92e985ae')
    net = nets.make_seg(); net.load_state_dict(nets.load_state(f))
    M.seg.append(('stage1', net.to(dev).eval()))
    S2 = W / 'stage2'
    if use_stage2 and (S2 / 'models.json').exists():
        for m in json.load(open(S2 / 'models.json'))['models']:
            f = nets.assemble([S2 / p for p in m['parts']], cache / f'{m["tag"]}.pt', sha256=m['sha256'])
            net = nets.make_cls('resnet50') if m['kind'] == 'cls' else nets.make_seg()
            net.load_state_dict(nets.load_state(f))
            (M.cls if m['kind'] == 'cls' else M.seg).append((m['tag'], net.to(dev).eval()))
    if (S2 / 'input_models.npz').exists():
        from .inputcheck import FeatureModels
        M.input_models = FeatureModels(S2 / 'input_models.npz')
    for c in [S2 / 'calib.json', W / 'calib.json']:
        if c.exists():
            M.calib.update(json.load(open(c)))
            break
    M.version = f'cls×{len(M.cls)} seg×{len(M.seg)}; calib={"есть" if M.calib != DEFAULT_CALIB else "этап 1"}'
    return M


def describe(mask, arrays, C, scale, head_px, pixel_spacing_mm=None):
    """Находки = связные области маски: площадь, размеры, положение на изображении, неуверенность границы."""
    pm, band, per_model = arrays['prob'], arrays['band'], arrays['per_model']
    mask = mask.astype(np.uint8)
    findings = []
    k, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    order = sorted(range(1, k), key=lambda c: -st[c, cv2.CC_STAT_AREA])
    for n, c in enumerate(order, 1):
        cm = (lab == c).astype(np.uint8)
        a256 = int(st[c, cv2.CC_STAT_AREA])
        dmax, dperp = feret(cm, scale)
        cx, cy = cen[c]
        ring = band & cv2.dilate(cm, np.ones((9, 9), np.uint8))
        f = dict(id=n, kind='finding', area_px256=a256, area_px_orig=round(a256 * scale ** 2),
                 area_pct_head=round(100 * a256 / head_px, 2),
                 dmax_px=round(dmax, 1), dperp_px=round(dperp, 1),
                 half='центр' if abs(cx - 128) < 15 else ('левая' if cx < 128 else 'правая'),
                 vert='верхняя' if cy < 256 / 3 else ('нижняя' if cy > 2 * 256 / 3 else 'средняя'),
                 centroid256=[round(float(cx), 1), round(float(cy), 1)],
                 bbox256=[int(st[c, 0]), int(st[c, 1]), int(st[c, 2]), int(st[c, 3])],
                 mean_prob=round(float(pm[cm > 0].mean()), 3),
                 band_ratio=round(float(ring.sum()) / max(a256, 1), 2),
                 models_agree=int(sum(((m_ >= C['contour_thr']) & (cm > 0)).sum() > 0.3 * a256 for m_ in per_model)),
                 n_models=len(per_model))
        if pixel_spacing_mm:
            f['dmax_mm'] = round(dmax * pixel_spacing_mm, 1)
            f['dperp_mm'] = round(dperp * pixel_spacing_mm, 1)
            f['area_mm2'] = round(a256 * scale ** 2 * pixel_spacing_mm ** 2, 1)
        findings.append(f)
    return findings


@torch.no_grad()
def _infer(M, x256, dev):
    xb = x256[None]
    L = np.stack([nets.cls_logits(net, xb, dev)[0] for _, net in M.cls])            # (m, 2, 4)
    feat = nets.cls_features(M.cls[0][1], xb, dev)[0] if M.input_models is not None else None
    maps = np.stack([nets.seg_maps(net, xb, dev, flip='both')[0] for _, net in M.seg])  # (m, 2, H, W)
    return L, maps, feat


def _softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def analyze(M, img, pixel_spacing_mm=None, source_name='', device=None):
    """Срез МРТ (путь, numpy RGB/серый) -> словарь с вероятностями, картами, находками и флагами."""
    dev = torch.device(device) if device else next(M.cls[0][1].parameters()).device
    C = M.calib
    dinfo = None
    if isinstance(img, (str, Path)) and str(img).lower().endswith(('.dcm', '.dicom')):
        from . import dicom as _dicom
        img, dinfo = _dicom.read(img)
        if pixel_spacing_mm is None:
            pixel_spacing_mm = dinfo['pixel_spacing_mm']
    g, colorfulness = _gray(img)
    h, w = g.shape
    x256 = prep(g)
    scale, oy, ox = prep_geometry(h, w)
    L, maps, feat = _infer(M, x256, dev)
    T = float(C.get('temperature', 1.0))
    P_models = _softmax(L / T).mean(1)               # по каждой модели: среднее по отражениям
    P = P_models.mean(0)                              # ансамбль
    p_tumor = 1 - float(P[NO_TUMOR])
    per_model = maps.mean(1)                          # (m, H, W): карта каждой модели (среднее по отражениям)
    pm = per_model.mean(0)
    pmax = per_model.max(0)                           # максимум по моделям (как в оценке по протоколу)
    pstd = per_model.std(0)
    mask = clean_mask(pm, C['contour_thr'], C['min_area'])
    outer = clean_mask(pm, C['band_lo'], C['min_area'])
    inner = (pm >= C['band_hi']).astype(np.uint8)
    band = ((outer | mask) & (1 - inner)).astype(np.uint8)
    review = ((pmax >= C['review_max']) & (pm < C['review_mean'])).astype(np.uint8)
    review &= (1 - cv2.dilate(mask, np.ones((5, 5), np.uint8)))
    review = clean_mask(review.astype(np.float32), 0.5, C['review_min_area'])
    hm = head_mask(x256)
    head_px = max(int(hm.sum()), 1)
    from .inputcheck import basic_checks, feature_checks, REFUSE
    checks = basic_checks(g, colorfulness, x256, hm)
    input_info = {}
    if M.input_models is not None and feat is not None:
        fc, input_info = feature_checks(M.input_models, feat)
        checks += fc
    refused = any(c[0] == REFUSE for c in checks)

    findings = describe(mask, dict(prob=pm, band=band, per_model=per_model), C, scale, head_px, pixel_spacing_mm)
    if dinfo and dinfo.get('orientation'):
        from .dicom import patient_side
        for f in findings:
            cx_orig = (f['centroid256'][0] + 0.5) * scale - 0.5 - ox
            f['patient_side'] = patient_side(dinfo['orientation'], cx_orig / max(w, 1))
    zones = []
    k2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(review)
    for n, c in enumerate(sorted(range(1, k2), key=lambda c: -st2[c, cv2.CC_STAT_AREA]), 1):
        zones.append(dict(id=n, kind='review', area_px256=int(st2[c, cv2.CC_STAT_AREA]),
                          half='центр' if abs(cen2[c][0] - 128) < 15 else ('левая' if cen2[c][0] < 128 else 'правая'),
                          centroid256=[round(float(cen2[c][0]), 1), round(float(cen2[c][1]), 1)],
                          bbox256=[int(st2[c, 0]), int(st2[c, 1]), int(st2[c, 2]), int(st2[c, 3])],
                          max_prob=round(float(pmax[lab2 == c].max()), 3)))

    top = int(np.argmax(P[:3]))
    top_p = float(P[top] / max(P[:3].sum(), 1e-6))
    flags = []
    tumor = p_tumor >= C['screen_thr']
    if tumor and not findings:
        flags.append('classifier_tumor_no_contour')
    if not tumor and findings:
        flags.append('contour_without_classifier')
    if tumor and (top_p < C['low_conf'] or float(P_models[:, :3].argmax(1).std()) > 0):
        flags.append('type_uncertain')
    if 0.3 < p_tumor < 0.7:
        flags.append('presence_uncertain')
    if any(f['band_ratio'] > 0.6 for f in findings):
        flags.append('boundary_uncertain')
    if zones:
        flags.append('review_zones')
    if max(h, w) <= 256:
        flags.append('low_resolution')
    if colorfulness > 8:
        flags.append('color_image')

    return dict(
        source=source_name, orig_hw=[h, w], scale=scale, pad_yx=[oy, ox], pixel_spacing_mm=pixel_spacing_mm,
        probs={CLASSES[i]: round(float(P[i]), 4) for i in range(4)},
        probs_by_model=[{CLASSES[i]: round(float(p[i]), 4) for i in range(4)} for p in P_models],
        p_tumor=round(p_tumor, 4), tumor=bool(tumor), top_type=CLASSES[top], top_type_p=round(top_p, 4),
        findings=[] if refused else findings, review_zones=[] if refused else zones, flags=flags, head_px256=head_px,
        refused=refused, checks=[dict(level=a, code=b, text=c) for a, b, c in checks], input_info=input_info,
        dicom={k: v for k, v in dinfo.items() if k != 'phi_present'} if dinfo else None,
        deidentified=bool(dinfo and dinfo.get('phi_present')),
        model_version=M.version, calib=dict(C),
        arrays=dict(x256=x256, prob=pm, prob_max=pmax, prob_std=pstd, mask=mask, band=band, review=review, head=hm,
                    per_model=per_model.astype(np.float32)),
    )


def overlay(res, show_band=True, show_review=True, mask=None):
    """Картинка для врача: красная заливка — контур, жёлтая полоса — неуверенная граница, голубые рамки — зоны просмотра."""
    a = res['arrays']
    m = a['mask'] if mask is None else mask
    rgb = cv2.cvtColor(a['x256'], cv2.COLOR_GRAY2RGB)
    up = 3
    rgb = cv2.resize(rgb, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    big = lambda z: cv2.resize(z.astype(np.uint8), None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    if show_band and a['band'].any():
        ov = rgb.copy(); ov[big(a['band']) > 0] = (255, 210, 0)
        rgb = cv2.addWeighted(rgb, .7, ov, .3, 0)
    if m.any():
        ov = rgb.copy(); ov[big(m) > 0] = (255, 40, 40)
        rgb = cv2.addWeighted(rgb, .7, ov, .3, 0)
        cs, _ = cv2.findContours(big(m), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cs, -1, (255, 60, 60), 2)
    for f in res['findings']:
        x, y = f['centroid256']
        cv2.putText(rgb, str(f['id']), (int(x * up) + 8, int(y * up) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(rgb, str(f['id']), (int(x * up) + 8, int(y * up) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 30, 30), 2)
    if show_review:
        for z in res['review_zones']:
            x, y, bw, bh = z['bbox256']
            cv2.rectangle(rgb, (x * up - 6, y * up - 6), ((x + bw) * up + 6, (y + bh) * up + 6), (0, 200, 255), 2)
            cv2.putText(rgb, f'?{z["id"]}', (x * up - 6, y * up - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    return rgb


def to_json(res):
    return {k: v for k, v in res.items() if k != 'arrays'}


def recompute(res, mask, edited_ids=()):
    """Находки по маске после правок врача (порог, удаление, добавление зоны, кисть)."""
    a = res['arrays']
    F = describe(mask, a, res['calib'], res['scale'], res['head_px256'], res.get('pixel_spacing_mm'))
    for f in F:
        f['edited'] = bool(edited_ids)
    return F


def mask_at(res, thr):
    """Контур при другом пороге (ползунок в интерфейсе врача)."""
    return clean_mask(res['arrays']['prob'], thr, res['calib']['min_area'])


def zone_mask(res, zone_id):
    """Маска зоны дополнительного просмотра (чтобы врач мог сделать её находкой)."""
    a = res['arrays']
    k, lab = cv2.connectedComponents(a['review'])
    for z in res['review_zones']:
        if z['id'] == zone_id:
            x, y = [int(round(v)) for v in z['centroid256']]
            x0, y0, w0, h0 = z['bbox256']
            sub = lab[y0:y0 + h0, x0:x0 + w0]
            vals, cnt = np.unique(sub[sub > 0], return_counts=True)
            if len(vals):
                region = (lab == vals[np.argmax(cnt)])
                grow = (a['prob_max'] >= res['calib']['review_max']) & (cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
                return (region | grow).astype(np.uint8)
    return np.zeros_like(a['mask'])
