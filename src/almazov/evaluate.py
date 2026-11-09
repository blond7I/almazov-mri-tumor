"""Оценка этапа 2 по протоколу docs/stage2_protocol.md (на CPU, по выгрузке Kaggle-ноутбука).

    python -m almazov.evaluate ПАПКА_С_РЕЗУЛЬТАТАМИ --out results/stage2 [--calib weights/stage2/calib.json]

Все пороги подбираются на валидации; тест считается один раз.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from .common import CLASSES, NO_TUMOR, clean_mask, dice

ALPHA = 0.10
RNG = np.random.RandomState(0)


# ------------------------------------------------------------------ утилиты
def boot(fn, groups, n=1000, seed=0):
    """95% ДИ бутстрэпом по группам (пациент / группа копий)."""
    rs = np.random.RandomState(seed)
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n):
        ii = np.concatenate([by[g] for g in rs.choice(ug, len(ug), replace=True)])
        v = fn(ii)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])] if vals else None


def auroc(score, label):
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label).astype(bool)
    if label.all() or (~label).all():
        return None
    return float(roc_auc_score(label, score))


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def ens_prob(Ls, T=1.0):
    """Ls: список (n, 2, 4) логитов моделей -> среднее softmax(L/T) по моделям и отражениям."""
    return np.mean([softmax(L / T).mean(1) for L in Ls], 0)


def ece(P, Y, bins=15):
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    rel = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs((pred[m] == Y[m]).mean() - conf[m].mean())
            rel.append(dict(lo=float(a), hi=float(b), n=int(m.sum()), acc=float((pred[m] == Y[m]).mean()), conf=float(conf[m].mean())))
    return float(e), rel


def u8f(a):
    return a.astype(np.float32) / 255 if a.dtype == np.uint8 else a


# ------------------------------------------------------------------ загрузка выгрузки
def load(res_dir):
    """Члены ансамбля: модели этапа 1 и этапа 2 + (по переменной ALMAZOV_EXTRA_MEMBERS, через запятую) кандидаты,
    например seg_b1,seg_b2,cls_b1 — их файлы в том же формате кладутся в ту же папку (extra_b.npz — их ответы на новых снимках)."""
    import os
    d = Path(res_dir)
    E = np.load(d / 'eval_data.npz', allow_pickle=True)
    more = [m.strip() for m in os.environ.get('ALMAZOV_EXTRA_MEMBERS', '').split(',') if m.strip()]
    segs = [k for k in ['seg_stage1', 'seg_s1', 'seg_s2', 'seg_s3'] + [m for m in more if m.startswith('seg_')] if (d / f'{k}.npz').exists()]
    clss = [k for k in ['cls_stage1', 'cls_s1', 'cls_s2'] + [m for m in more if m.startswith('cls_')] if (d / f'{k}.npz').exists()]
    S = {k: np.load(d / f'{k}.npz') for k in segs}
    C = {k: np.load(d / f'{k}.npz') for k in clss}
    ex = None
    if (d / 'extra.npz').exists():
        z = np.load(d / 'extra.npz', allow_pickle=True)
        z = {k: z[k] for k in z.files}
        if more and (d / 'extra_b.npz').exists():
            zb = np.load(d / 'extra_b.npz', allow_pickle=True)
            z.update({k: zb[k] for k in zb.files})
        ex = (z, pd.read_csv(d / 'extra_meta.csv', keep_default_na=False))
    return E, S, C, segs, clss, ex


def stack_maps(S, members, key):
    M = np.stack([u8f(S[k][key]) for k in members])          # (m, n, H, W)
    return M.mean(0), M.max(0), M.std(0)


# ------------------------------------------------------------------ 3.1 полоса неуверенности (CRC)
def patient_risk(pm, gt, pids, thr, min_area=30):
    """Доля пикселей опухоли вне внешнего контура (карта >= thr, без пятен < min_area) — по каждому пациенту."""
    miss, tot = {}, {}
    for p, g, pid in zip(pm, gt, pids):
        if g.sum() == 0:
            continue
        o = clean_mask(p, thr, min_area)
        miss[pid] = miss.get(pid, 0) + int((g & (o == 0)).sum())
        tot[pid] = tot.get(pid, 0) + int(g.sum())
    keys = sorted(tot)
    return np.array([miss[k] / tot[k] for k in keys]), keys


def crc_threshold(pm, gt, pids, alpha=ALPHA, grid=np.round(np.arange(0.02, 0.51, 0.01), 2)):
    """Наибольший порог внешнего контура, при котором граница CRC (n/(n+1))·R + 1/(n+1) <= alpha для него и всех меньших."""
    best, table = None, []
    for t in grid:
        r, keys = patient_risk(pm, gt, pids, t)
        n = len(r)
        bound = n / (n + 1) * r.mean() + 1 / (n + 1)
        table.append(dict(thr=float(t), risk=float(r.mean()), bound=float(bound)))
        if bound <= alpha:
            best = float(t)
        else:
            break
    return best, table


def inner_threshold(pm, gt, target=0.95, grid=np.round(np.arange(0.5, 0.99, 0.01), 2)):
    for t in grid:
        sel = pm >= t
        if sel.sum() and (gt[sel]).mean() >= target:
            return float(t)
    return float(grid[-1])


def ring_same_area(c, area):
    """Кольцо вокруг контура c, расширяемое до площади не меньше area (базовая линия для полосы)."""
    if c.sum() == 0 or area <= 0:
        return np.zeros_like(c)
    r = c.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(60):
        r = cv2.dilate(r, k)
        if (r & (1 - c)).sum() >= area:
            break
    return (r & (1 - c)).astype(np.uint8)


def band_vs_ring(pm, gt, lo, thr=0.5):
    cap_b = cap_r = miss = 0
    per = []
    for p, g in zip(pm, gt):
        c = clean_mask(p, thr)
        o = clean_mask(p, lo) | c
        band = (o & (1 - c)).astype(np.uint8)
        m = g & (c == 0)
        if c.sum() == 0 or m.sum() == 0:
            continue
        ring = ring_same_area(c, int(band.sum()))
        cb, cr = int((m & band).sum()), int((m & ring).sum())
        cap_b += cb; cap_r += cr; miss += int(m.sum())
        per.append((cb / m.sum(), cr / m.sum()))
    per = np.array(per) if per else np.zeros((0, 2))
    return dict(slices=int(len(per)), captured_by_band=cap_b / max(miss, 1), captured_by_ring_same_area=cap_r / max(miss, 1),
                share_slices_band_better=float((per[:, 0] > per[:, 1]).mean()) if len(per) else None)


def review_zones_eval(pm, pmax, gt, is_normal, C):
    """Зоны дополнительного просмотра: точность, сколько пропущенных компонентов опухоли отмечено, ложные зоны на норме."""
    z_hit = z_all = 0
    missed_comp = missed_flagged = 0
    false_on_normal = []
    for p, mx, g, nrm in zip(pm, pmax, gt, is_normal):
        c = clean_mask(p, C['contour_thr'], C['min_area'])
        rv = ((mx >= C['review_max']) & (p < C['review_mean'])).astype(np.uint8)
        rv &= (1 - cv2.dilate(c, np.ones((5, 5), np.uint8)))
        rv = clean_mask(rv.astype(np.float32), 0.5, C['review_min_area'])
        k, lab = cv2.connectedComponents(rv)
        if nrm:
            false_on_normal.append(k - 1)
            continue
        for j in range(1, k):
            z_all += 1
            z_hit += int((g[lab == j]).any())
        kg, labg = cv2.connectedComponents(g.astype(np.uint8))
        for j in range(1, kg):
            comp = labg == j
            if not (c[comp]).any():
                missed_comp += 1
                missed_flagged += int((rv[comp]).any())
    return dict(zones=z_all, zone_precision=z_hit / max(z_all, 1), missed_tumor_components=missed_comp,
                missed_components_flagged=missed_flagged / max(missed_comp, 1),
                false_zones_per_normal=float(np.mean(false_on_normal)) if false_on_normal else None)


# ------------------------------------------------------------------ 3.1 полоса v2: семейства «база + кольцо радиуса r», CRC по r
FAMILIES = {
    'ring': 'контур + кольцо',
    'prob': 'контур ∪ (среднее карт ≥ 0.2) + кольцо',
    'max': 'контур ∪ (максимум карт ≥ 0.3) + кольцо',
}
R_GRID = np.arange(0, 41)


def band_base(pm, pmax, fam, C=None):
    c = clean_mask(pm, (C or {}).get('contour_thr', 0.5), (C or {}).get('min_area', 30))
    if fam == 'ring':
        b = c
    elif fam == 'prob':
        b = c | clean_mask(pm, 0.2)
    else:
        b = c | clean_mask(pmax, 0.3)
    return c, b.astype(np.uint8)


def family_stats(pm, pmax, gt, pids, fam):
    """Для каждого среза с построенным контуром: доля опухоли дальше r от базы (по пациентам) и площадь полосы при каждом r."""
    miss = {}; tot = {}; area = []
    for p, mx, g, pid in zip(pm, pmax, gt, pids):
        c, b = band_base(p, mx, fam)
        if c.sum() == 0:                 # гарантия — для срезов, где система нарисовала контур (условие зависит только от входа)
            continue
        d = cv2.distanceTransform((1 - b).astype(np.uint8), cv2.DIST_L2, 5)
        if g.any():
            dg = d[g]
            miss.setdefault(pid, np.zeros(len(R_GRID)))
            miss[pid] += np.array([(dg > r).sum() for r in R_GRID])
            tot[pid] = tot.get(pid, 0) + int(g.sum())
        outside_c = (c == 0)
        area.append(np.array([((d <= r) & outside_c).sum() for r in R_GRID]))
    keys = sorted(tot)
    risk = np.stack([miss[k] / tot[k] for k in keys]) if keys else np.zeros((0, len(R_GRID)))
    return risk, np.stack(area) if area else np.zeros((0, len(R_GRID))), keys


def crc_r(risk, alpha=ALPHA):
    n = len(risk)
    bound = n / (n + 1) * risk.mean(0) + 1 / (n + 1)
    ok = np.where(bound <= alpha)[0]
    return (int(R_GRID[ok[0]]) if len(ok) else None), bound


def band_v2(pm_v, mx_v, gv, pv, pm_t, mx_t, gt, pt):
    """Выбор семейства и радиуса только на валидации; тест — один раз для выбранного (и для справки — для остальных)."""
    out = dict(alpha=ALPHA, families=FAMILIES, note='условная гарантия: срезы, где система построила контур')
    best = None
    for fam in FAMILIES:
        rv, av, kv = family_stats(pm_v, mx_v, gv, pv, fam)
        r, bound = crc_r(rv)
        entry = dict(val_patients=len(kv), r=r, val_bound_at_r=float(bound[r]) if r is not None else None,
                     val_mean_band_area_px=float(av[:, r].mean()) if r is not None else None)
        out[fam] = entry
        if r is not None and (best is None or entry['val_mean_band_area_px'] < out[best]['val_mean_band_area_px']):
            best = fam
    out['selected_on_val'] = best
    for fam in FAMILIES:
        r = out[fam]['r']
        if r is None:
            continue
        rt, at, kt = family_stats(pm_t, mx_t, gt, pt, fam)
        out[fam]['test'] = dict(patients=len(kt), mean_risk=float(rt[:, r].mean()),
                                risk_ci=boot(lambda ii: rt[ii, r].mean(), np.arange(len(rt))),
                                share_patients_risk_gt_alpha=float((rt[:, r] > ALPHA).mean()),
                                mean_band_area_px=float(at[:, r].mean()),
                                risk_without_band=float(rt[:, 0].mean()) if fam == 'ring' else None)
    return out


# ------------------------------------------------------------------ главный расчёт
def run(res_dir, out_dir, calib_out=None):
    from .pipeline import DEFAULT_CALIB
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    E, S, Cl, segs, clss, ex = load(res_dir)
    R = dict(members=dict(seg=segs, cls=clss), protocol='docs/stage2_protocol.md')
    C = dict(DEFAULT_CALIB)

    # ---------- сегментация: валидация (подбор) и тест
    pm_v, mx_v, sd_v = stack_maps(S, segs, 'fva')
    pm_t, mx_t, sd_t = stack_maps(S, segs, 'fte')
    gv, gt = E['m_fva'] > 0, E['m_fte'] > 0
    pv, ptid = E['pid_fva'].astype(str), E['pid_fte'].astype(str)
    lo, table = crc_threshold(pm_v, gv, pv)
    hi = inner_threshold(pm_v, gv)
    C['band_lo'] = lo if lo is not None else C['band_lo']
    C['band_hi'] = hi
    r_t, keys_t = patient_risk(pm_t, gt, ptid, C['band_lo'])
    R['crc'] = dict(alpha=ALPHA, n_val_patients=int(len(set(pv[gv.reshape(len(gv), -1).any(1)]))), band_lo=C['band_lo'], band_hi=hi,
                    val_table=table, test_patients=int(len(r_t)), test_mean_risk=float(r_t.mean()),
                    test_risk_ci=boot(lambda ii: r_t[ii].mean(), np.arange(len(r_t))),
                    test_share_patients_risk_gt_alpha=float((r_t > ALPHA).mean()),
                    risk_at_contour_05=float(patient_risk(pm_t, gt, ptid, 0.5)[0].mean()))
    R['band_vs_ring'] = band_vs_ring(pm_t, gt, C['band_lo'])
    R['crc_v2'] = band_v2(pm_v, mx_v, gv, pv, pm_t, mx_t, gt, ptid)
    sel = R['crc_v2']['selected_on_val']
    if sel:
        C['band_family'], C['band_r'] = sel, R['crc_v2'][sel]['r']
    # срезы с опухолью, где контура нет вовсе: их ловит не полоса, а правило «классификатор видит опухоль — смотрите весь срез»
    no_c = np.array([clean_mask(p).sum() == 0 for p in pm_t]) & gt.reshape(len(gt), -1).any(1)
    R['complete_miss_figshare_test'] = dict(share_tumor_slices=float(no_c.sum() / max(gt.reshape(len(gt), -1).any(1).sum(), 1)), n=int(no_c.sum()))

    # до/после: Dice по пациентам
    def dice_of(p):
        return dice(np.stack([clean_mask(x) for x in p]) > 0, gt)
    d1 = dice_of(u8f(S['seg_stage1']['fte']))
    de = dice_of(pm_t)
    pid_u = np.unique(ptid)
    pp = lambda d: np.array([d[ptid == u].mean() for u in pid_u])
    d1p, dep = pp(d1), pp(de)
    yf = E['y_fte']
    R['seg_before_after'] = dict(
        stage1=dict(dice_mean=float(d1.mean()), dice_median=float(np.median(d1)),
                    by_class={CLASSES[k]: float(d1[yf == k].mean()) for k in range(3)}),
        ensemble=dict(dice_mean=float(de.mean()), dice_median=float(np.median(de)),
                      by_class={CLASSES[k]: float(de[yf == k].mean()) for k in range(3)},
                      ci=boot(lambda ii: de[ii].mean(), ptid)),
        delta_dice_by_patient=float((dep - d1p).mean()),
        delta_ci_paired=boot(lambda ii: (dep[ii] - d1p[ii]).mean(), np.arange(len(pid_u))),
        share_slices_dice_lt_05=dict(stage1=float((d1 < .5).mean()), ensemble=float((de < .5).mean())))

    # снимки организаторов из теста (с масками и без): зоны просмотра, провалы сегментации
    pm_o, mx_o, sd_o = stack_maps(S, segs, 'ote')
    has, go = E['has_ote'].astype(bool), E['m_ote'] > 0
    yo = E['y_ote']
    R['review_zones'] = review_zones_eval(pm_o[has | (yo == NO_TUMOR)], mx_o[has | (yo == NO_TUMOR)], go[has | (yo == NO_TUMOR)],
                                          (yo == NO_TUMOR)[has | (yo == NO_TUMOR)], C)
    R['review_zones_figshare_test'] = review_zones_eval(pm_t, mx_t, gt, np.zeros(len(gt), bool), C)

    # обнаружение провала сегментации (Dice < 0.5) на снимках с маской
    idx = np.where(has)[0]
    co = np.stack([clean_mask(p) for p in pm_o[idx]])
    dd = dice(co > 0, go[idx])
    fail = dd < 0.5
    oo = np.stack([clean_mask(p, C['band_lo']) for p in pm_o[idx]]) | co
    area = co.reshape(len(co), -1).sum(1)
    disagree = np.array([s[o > 0].mean() if o.any() else s.max() for s, o in zip(sd_o[idx], oo)])
    band_ratio = (oo & (1 - co)).reshape(len(co), -1).sum(1) / np.maximum(area, 1)
    lowprob = np.array([1 - p[c > 0].mean() if c.any() else 1.0 for p, c in zip(pm_o[idx], co)])
    P1 = softmax(Cl['cls_stage1']['lte']).mean(1)
    grp_o = E['group_ote']
    scores = dict(disagreement=disagree, band_ratio=band_ratio, one_minus_mean_prob=lowprob,
                  baseline_small_area=-area.astype(float), baseline_classifier_1_minus_maxprob=1 - P1[idx].max(1))
    R['seg_failure_detection'] = dict(n=int(len(idx)), failures=int(fail.sum()),
                                      auroc={k: auroc(v, fail) for k, v in scores.items()},
                                      auroc_ci={k: boot(lambda ii, v=v: auroc(v[ii], fail[ii]), grp_o[idx], n=300) for k, v in scores.items()})

    # полные промахи контура на снимках организаторов с маской: сколько из них поймал классификатор (правило «смотрите весь срез»)
    P_ens1 = ens_prob([Cl[k]['lte'] for k in clss], 1.0)
    empty = co.reshape(len(co), -1).sum(1) == 0
    flagged = (1 - P_ens1[idx, NO_TUMOR]) >= C['screen_thr']
    R['complete_miss_org_test'] = dict(n_with_mask=int(len(idx)), no_contour=int(empty.sum()),
                                       no_contour_flagged_by_classifier=float(flagged[empty].mean()) if empty.any() else None)

    # ---------- классификатор: калибровка, отказ, конформные множества
    Lv = [Cl[k]['lva'] for k in clss]
    Lt = [Cl[k]['lte'] for k in clss]
    yv = E['y_ova']
    grid = np.round(np.arange(0.5, 3.01, 0.01), 2)
    nll = [float(-np.log(ens_prob(Lv, T)[np.arange(len(yv)), yv] + 1e-12).mean()) for T in grid]
    T = float(grid[int(np.argmin(nll))])
    C['temperature'] = T
    P0, PT = ens_prob(Lt, 1.0), ens_prob(Lt, T)
    Ps1 = ens_prob([Cl['cls_stage1']['lte']], 1.0)
    e0, rel0 = ece(P0, yo); eT, relT = ece(PT, yo); es1, _ = ece(Ps1, yo)
    from .train import cls_metrics
    R['classifier'] = dict(
        temperature=T, ece_stage1=es1, ece_ensemble_before=e0, ece_ensemble_after=eT, reliability_before=rel0, reliability_after=relT,
        stage1=cls_metrics(Ps1, yo), ensemble=cls_metrics(PT, yo),
        acc_ci_stage1=boot(lambda ii: (Ps1[ii].argmax(1) == yo[ii]).mean(), grp_o),
        acc_ci_ensemble=boot(lambda ii: (PT[ii].argmax(1) == yo[ii]).mean(), grp_o),
        delta_acc_ci_paired=boot(lambda ii: (PT[ii].argmax(1) == yo[ii]).mean() - (Ps1[ii].argmax(1) == yo[ii]).mean(), grp_o))
    err = PT.argmax(1) != yo
    conf = PT.max(1)
    order = np.argsort(conf)
    rc = []
    for defer in [0, .05, .10, .15, .20]:
        k = int(round(defer * len(yo)))
        keep = order[k:]
        rc.append(dict(defer=defer, coverage=1 - defer, accuracy=float((~err[keep]).mean()),
                       errors_caught=float(err[order[:k]].sum() / max(err.sum(), 1))))
    R['risk_coverage'] = rc
    R['error_detection_auroc'] = dict(ensemble_conf=auroc(-conf, err),
                                      stage1_maxprob=auroc(-Ps1.max(1), Ps1.argmax(1) != yo))
    # конформные множества LAC
    Pv = ens_prob(Lv, T)
    s = 1 - Pv[np.arange(len(yv)), yv]
    n = len(s)
    q = float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method='higher'))
    sets = PT >= 1 - q
    cov = sets[np.arange(len(yo)), yo]
    halves = []
    for _ in range(200):
        m = RNG.rand(len(yo)) < .5
        halves.append(cov[m].mean())
    R['conformal_lac'] = dict(alpha=ALPHA, qhat=q, coverage=float(cov.mean()), avg_set_size=float(sets.sum(1).mean()),
                              coverage_by_class={CLASSES[k]: float(cov[yo == k].mean()) for k in range(4)},
                              share_singletons=float((sets.sum(1) == 1).mean()),
                              coverage_halves=dict(min=float(np.min(halves)), max=float(np.max(halves)), std=float(np.std(halves))))
    C['conformal_qhat'] = q
    # APS (адаптивные множества): без пустых множеств; та же калибровка на валидации
    def aps_scores(P, y):
        o = np.argsort(-P, 1); cs = np.cumsum(np.take_along_axis(P, o, 1), 1)
        rank = np.argmax(o == y[:, None], 1)
        return cs[np.arange(len(y)), rank]
    sa = aps_scores(Pv, yv)
    qa = float(np.quantile(sa, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method='higher'))
    o = np.argsort(-PT, 1); cs = np.cumsum(np.take_along_axis(PT, o, 1), 1)
    k_ = (cs < qa).sum(1) + 1
    sets_a = np.zeros_like(PT, bool)
    for i in range(len(PT)):
        sets_a[i, o[i, :k_[i]]] = True
    cov_a = sets_a[np.arange(len(yo)), yo]
    R['conformal_aps'] = dict(alpha=ALPHA, qhat=qa, coverage=float(cov_a.mean()), avg_set_size=float(sets_a.sum(1).mean()),
                              coverage_by_class={CLASSES[k]: float(cov_a[yo == k].mean()) for k in range(4)},
                              share_singletons=float((sets_a.sum(1) == 1).mean()))
    defer_thr = float(np.quantile(conf, .15))
    C['defer_conf_thr'] = defer_thr

    # ---------- дополнительный тест: «новые» снимки полного архива
    if ex is not None:
        Z, meta = ex
        nw = (meta.status == 'новый').values & (meta.label.values >= 0)
        if nw.any():
            y = meta.label.values[nw]
            g = meta.group.values[nw]
            Lx = [Z['logits_' + k] for k in clss if 'logits_' + k in Z]
            Px1 = ens_prob([Z['logits_cls_stage1']], 1.0)[nw]
            PxE = ens_prob(Lx, T)[nw]
            errE = PxE.argmax(1) != y
            R['extra_new_deferral'] = dict(errors=int(errE.sum()),
                                           errors_flagged_low_conf=float((PxE.max(1)[errE] < defer_thr).mean()) if errE.any() else None,
                                           share_flagged=float((PxE.max(1) < defer_thr).mean()), threshold_from='15% наименее уверенных на тесте')
            R['extra_new'] = dict(n=int(nw.sum()), groups=int(len(np.unique(g))), by_class=np.bincount(y, minlength=4).tolist(),
                                  status_counts=meta.status.value_counts().to_dict(),
                                  stage1=cls_metrics(Px1, y), ensemble=cls_metrics(PxE, y),
                                  acc_ci_stage1=boot(lambda ii: (Px1[ii].argmax(1) == y[ii]).mean(), g),
                                  acc_ci_ensemble=boot(lambda ii: (PxE[ii].argmax(1) == y[ii]).mean(), g))
            sk = ['seg_' + k for k in segs if 'seg_' + k in Z]
            if sk:
                pmx = np.mean([u8f(Z[k]) for k in sk], 0)[nw]
                has_c = np.array([clean_mask(p).sum() > 0 for p in pmx])
                R['extra_new']['contour_found'] = dict(tumor=float(has_c[y != NO_TUMOR].mean()) if (y != NO_TUMOR).any() else None,
                                                       normal_false_contour=float(has_c[y == NO_TUMOR].mean()) if (y == NO_TUMOR).any() else None)

    C['fitted_on'] = 'валидация (Figshare 38 пациентов для границы; 1419 снимков для температуры и конформных множеств)'
    json.dump(R, open(out / 'stage2_eval.json', 'w'), ensure_ascii=False, indent=1, default=float)
    if calib_out:
        Path(calib_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(C, open(calib_out, 'w'), ensure_ascii=False, indent=1)
    return R, C


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out', default='results/stage2')
    ap.add_argument('--calib', default=None)
    a = ap.parse_args(argv)
    R, C = run(a.results, a.out, a.calib)
    short = {k: v for k, v in R.items() if k in ('crc', 'band_vs_ring', 'seg_before_after', 'classifier', 'risk_coverage', 'extra_new')}
    print(json.dumps(short, ensure_ascii=False, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main()
