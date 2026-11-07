"""Картинки для презентации и отчёта этапа 2 (по выгрузке Kaggle и результату evaluate)."""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import CLASSES, NO_TUMOR, clean_mask, dice
from .evaluate import load, stack_maps, ens_prob, u8f

INK, MUTED, ACC, WARN, OK = '#1d2733', '#6b7785', '#2f6f9f', '#d9a21b', '#3c8f5a'
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})


def _img(z, group, idx):
    try:
        return cv2.imdecode(np.frombuffer(z.read(f'{group}/{int(idx):05d}.jpg'), np.uint8), cv2.IMREAD_GRAYSCALE)
    except KeyError:
        return None


def overlay(x, pm, lo, gt=None, thr=0.5):
    rgb = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    c = clean_mask(pm, thr)
    o = clean_mask(pm, lo) | c
    band = (o & (1 - c)).astype(np.uint8)
    ov = rgb.copy(); ov[band > 0] = (255, 205, 0); rgb = cv2.addWeighted(rgb, .65, ov, .35, 0)
    ov = rgb.copy(); ov[c > 0] = (230, 40, 40); rgb = cv2.addWeighted(rgb, .7, ov, .3, 0)
    cs, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cs, -1, (230, 40, 40), 1)
    if gt is not None and gt.any():
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cs, -1, (40, 200, 90), 1)
    return rgb


def reliability(R, out):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], ls='--', c=MUTED, lw=1)
    for key, lab, col in [('reliability_before', f'до калибровки (ECE {R["classifier"]["ece_ensemble_before"]:.3f})', MUTED),
                          ('reliability_after', f'после (ECE {R["classifier"]["ece_ensemble_after"]:.3f})', ACC)]:
        rel = R['classifier'][key]
        ax.plot([r['conf'] for r in rel], [r['acc'] for r in rel], marker='o', ms=4, c=col, label=lab)
    ax.set_xlabel('уверенность модели'); ax.set_ylabel('доля верных ответов'); ax.set_xlim(.3, 1.01); ax.set_ylim(.3, 1.01)
    ax.legend(frameon=False, fontsize=9, loc='upper left'); fig.tight_layout(); fig.savefig(out / 'reliability.png'); plt.close(fig)


def risk_coverage(R, out):
    rc = R['risk_coverage']
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([100 * r['defer'] for r in rc], [100 * r['accuracy'] for r in rc], marker='o', c=ACC, label='точность на оставшихся')
    ax.plot([100 * r['defer'] for r in rc], [100 * r['errors_caught'] for r in rc], marker='s', c=WARN, label='доля пойманных ошибок')
    for r in rc:
        if r['defer'] == .15:
            ax.annotate(f'{100 * r["errors_caught"]:.0f}% ошибок', (15, 100 * r['errors_caught']), textcoords='offset points',
                        xytext=(6, -14), fontsize=9, color=INK)
    ax.set_xlabel('отложено врачу, % наименее уверенных'); ax.set_ylabel('%'); ax.set_ylim(0, 101)
    ax.legend(frameon=False, fontsize=9, loc='center right'); fig.tight_layout(); fig.savefig(out / 'risk_coverage.png'); plt.close(fig)


def band_examples(res_dir, R, out, n=3):
    E, S, Cl, segs, clss, ex = load(res_dir)
    z = zipfile.ZipFile(Path(res_dir) / 'images_jpg.zip')
    pm, _, _ = stack_maps(S, segs, 'fte')
    gt = E['m_fte'] > 0
    lo = R['crc']['band_lo']
    gain = []
    for i, (p, g) in enumerate(zip(pm, gt)):
        c = clean_mask(p)
        miss = g & (c == 0)
        if c.sum() == 0 or miss.sum() < 40:
            continue
        o = clean_mask(p, lo) | c
        gain.append((float((miss & (o & (1 - c)).astype(bool)).sum() / miss.sum()), i))
    pick = [i for _, i in sorted(gain, reverse=True)[:n]]
    if not pick:
        return
    fig, axs = plt.subplots(1, len(pick), figsize=(3.4 * len(pick), 3.6))
    axs = np.atleast_1d(axs)
    for ax, i in zip(axs, pick):
        x = _img(z, 'fte', E['idx_fte'][i])
        ax.imshow(overlay(x, pm[i], lo, gt[i])); ax.axis('off')
        d = dice((clean_mask(pm[i]) > 0)[None], gt[i][None])[0]
        ax.set_title(f'{CLASSES[int(E["y_fte"][i])]}, Dice {d:.2f}', fontsize=10)
    fig.suptitle('красное — контур, жёлтое — неуверенная граница, зелёное — контур врача', fontsize=10, color=MUTED)
    fig.tight_layout(); fig.savefig(out / 'band_examples.png'); plt.close(fig)


def gallery(res_dir, R, out):
    """4 верных и 4 ошибочных случая теста: что сказала система."""
    E, S, Cl, segs, clss, ex = load(res_dir)
    z = zipfile.ZipFile(Path(res_dir) / 'images_jpg.zip')
    pm, _, _ = stack_maps(S, segs, 'ote')
    T = R['classifier']['temperature']
    P = ens_prob([Cl[k]['lte'] for k in clss], T)
    y = E['y_ote']; pred = P.argmax(1); conf = P.max(1)
    rs = np.random.RandomState(1)
    right = np.where(pred == y)[0]; wrong = np.where(pred != y)[0]
    pick = list(rs.choice(right, min(4, len(right)), replace=False)) + list(wrong[np.argsort(-conf[wrong])][:2]) + \
        list(wrong[np.argsort(conf[wrong])][:2])
    fig, axs = plt.subplots(2, 4, figsize=(13, 7.2))
    for ax, i in zip(axs.ravel(), pick):
        x = _img(z, 'ote', E['idx_ote'][i])
        g = (E['m_ote'][i] > 0) if E['has_ote'][i] else None
        ax.imshow(overlay(x, pm[i], R['crc']['band_lo'], g)); ax.axis('off')
        ok = pred[i] == y[i]
        flag = 'отложить врачу' if conf[i] < np.quantile(conf, .15) else ''
        ax.set_title(f'{"✓" if ok else "✗"} ИИ: {CLASSES[pred[i]]} {conf[i]:.0%}\nистина: {CLASSES[y[i]]}  {flag}', fontsize=9,
                     color=OK if ok else '#b33a3a')
    fig.tight_layout(); fig.savefig(out / 'gallery.png'); plt.close(fig)


def extra_grid(res_dir, R, out, n=16):
    """«Новые» снимки полного архива: что на них, и как ошибается модель этапа 1."""
    E, S, Cl, segs, clss, ex = load(res_dir)
    if ex is None:
        return
    Z, meta = ex
    z = zipfile.ZipFile(Path(res_dir) / 'images_jpg.zip')
    nw = np.where((meta.status == 'новый').values & (meta.label.values >= 0))[0]
    if not len(nw):
        return
    P1 = ens_prob([Z['logits_cls_stage1']], 1.0)
    y = meta.label.values
    cm = np.zeros((4, 4), int)
    for i in nw:
        cm[y[i], P1[i].argmax()] += 1
    R2 = dict(confusion_stage1_new=cm.tolist(), names=meta.name.values[nw].tolist(), h=meta.h.values[nw].tolist(),
              w=meta.w.values[nw].tolist(), label=y[nw].tolist(), pred_stage1=P1[nw].argmax(1).tolist())
    json.dump(R2, open(out / 'extra_new_details.json', 'w'), ensure_ascii=False, indent=1)
    sel = nw[:n]
    cols = 8
    fig, axs = plt.subplots(int(np.ceil(len(sel) / cols)), cols, figsize=(2.1 * cols, 2.4 * np.ceil(len(sel) / cols)))
    for ax in np.atleast_1d(axs).ravel():
        ax.axis('off')
    for ax, i in zip(np.atleast_1d(axs).ravel(), sel):
        x = _img(z, 'extra', i)
        if x is not None:
            ax.imshow(x, cmap='gray')
        ok = P1[i].argmax() == y[i]
        ax.set_title(f'{CLASSES[y[i]][:9]} → {CLASSES[P1[i].argmax()][:9]}', fontsize=7, color=OK if ok else '#b33a3a')
    fig.tight_layout(); fig.savefig(out / 'extra_new_grid.png'); plt.close(fig)


def make_all(res_dir, eval_dir):
    out = Path(eval_dir); out.mkdir(parents=True, exist_ok=True)
    R = json.load(open(out / 'stage2_eval.json'))
    for f in (reliability, risk_coverage):
        try:
            f(R, out)
        except Exception as e:
            print('figure failed:', f.__name__, e)
    for f in (band_examples, gallery, extra_grid):
        try:
            f(res_dir, R, out)
        except Exception as e:
            print('figure failed:', f.__name__, type(e).__name__, e)
    return sorted(p.name for p in out.glob('*.png'))
