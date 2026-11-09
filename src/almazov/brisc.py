"""BRISC 2025 (Fateh et al., Scientific Data 2026, CC BY 4.0): 6000 срезов T1 из тех же открытых источников,
что и данные организаторов, с контурами опухолей, проверенными врачами, и отобранными здоровыми срезами T1.

1) сопоставление каждого снимка BRISC с нашими данными (организаторы, полный архив, Figshare) — по d8 и корреляции;
2) часть разбиения для каждого снимка: train / val / test (как у найденной копии) или «внешний» (копии нет);
3) оценка прототипа на снимках нашего теста и на внешних: тип, контур (Dice с контуром врачей), здоровые T1;
4) пары «снимок — маска» из обучающей части для дообучения сегментации.
"""
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .common import CLASSES, NO_TUMOR, prep, small, z1, d8, SIZE

CODE = {'gl': 0, 'me': 1, 'pi': 2, 'nt': 3, 'no': 3}
FOLDER = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'no_tumor': 3, 'notumor': 3}


def prep_mask(m):
    """Маска исходного размера -> 256x256 в той же геометрии, что prep() (квадрат с полями, без искажения)."""
    m = (np.asarray(m) > 0).astype(np.uint8)
    h, w = m.shape
    s = max(h, w)
    if h != w:
        c = np.zeros((s, s), np.uint8)
        c[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = m
        m = c
    return cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)


def find_root(base):
    hit = sorted(glob.glob(os.path.join(base, '**', 'classification_task'), recursive=True))
    return str(Path(hit[0]).parent) if hit else None


def load(root, log=print):
    """-> df (file, split_b, label, plane, img, mask), X (n,256,256) после prep, M (n,256,256) маски (0 — нет маски)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, 'classification_task', '*', '*', '*.jpg'))):
        sp, cl = Path(p).parts[-3], Path(p).parts[-2]
        name = Path(p).stem
        parts = name.split('_')                       # brisc2025_train_00001_gl_ax_t1
        plane = parts[4] if len(parts) > 4 else ''
        mk = os.path.join(root, 'segmentation_task', sp, 'masks', name + '.png')
        rows.append(dict(file=name, split_b=sp, label=FOLDER.get(cl, CODE.get(parts[3] if len(parts) > 3 else '', -1)),
                         plane={'ax': 'аксиальная', 'co': 'корональная', 'sa': 'сагиттальная'}.get(plane, plane),
                         img=p, mask=mk if os.path.exists(mk) else ''))
    df = pd.DataFrame(rows)
    X = np.zeros((len(df), SIZE, SIZE), np.uint8)
    M = np.zeros((len(df), SIZE, SIZE), np.uint8)
    t0 = time.time()
    for k, r in enumerate(df.itertuples()):
        g = np.array(Image.open(r.img).convert('L'))
        X[k] = prep(g)
        if r.mask:
            M[k] = prep_mask(np.array(Image.open(r.mask).convert('L')))
    log(f'BRISC: {len(df)} снимков ({df.label.value_counts().sort_index().to_dict()}), масок {int((df["mask"] != "").sum())}, '
        f'{time.time() - t0:.0f} с')
    return df, X, M


def _vecs(X):
    s = small(X)
    return np.stack([z1(a) for a in s]).astype(np.float32), s


def best_match(Xq, Xref, dev='cpu', chunk=2048):
    """Для каждого снимка Xq — лучшая корреляция с Xref по 8 поворотам/отражениям: (индекс, корреляция)."""
    import torch
    R = torch.tensor(_vecs(Xref)[0], device=dev)
    _, sq = _vecs(Xq)
    best = np.full(len(Xq), -1.0, np.float32)
    arg = np.zeros(len(Xq), np.int64)
    for t in range(8):
        Q = np.stack([z1(d8(a, t).copy()) for a in sq]).astype(np.float32)
        for s0 in range(0, len(Q), chunk):
            c = torch.tensor(Q[s0:s0 + chunk], device=dev) @ R.T
            v, i = c.max(1)
            v, i = v.cpu().numpy(), i.cpu().numpy()
            upd = v > best[s0:s0 + chunk]
            best[s0:s0 + chunk][upd] = v[upd]
            arg[s0:s0 + chunk][upd] = i[upd]
    return arg, best


def assign(dfb, Xb, A, df, AX, ex_meta, fX, fpart, dev='cpu', thr=0.98, thr_possible=0.93, log=print):
    """Часть нашего разбиения для каждого снимка BRISC. 'external' — копии нет ни в данных организаторов, ни в Figshare."""
    ia, ca = best_match(Xb, A, dev)
    ix, cx = best_match(Xb, AX, dev) if len(AX) else (np.zeros(len(Xb), int), np.full(len(Xb), -1.0))
    if_, cf = best_match(Xb, fX, dev)
    part, src, ref = [], [], []
    for k in range(len(dfb)):
        cand = [(ca[k], 'org', ia[k]), (cx[k], 'extra', ix[k]), (cf[k], 'fig', if_[k])]
        c, s, i = max(cand, key=lambda z: z[0])
        if c >= thr:
            if s == 'org':
                p = 'conflict' if str(df.label_conflict.values[i]) == 'True' else df.part.values[i]
            elif s == 'fig':
                p = fpart[i]
            else:
                st = ex_meta.status.values[i] if ex_meta is not None and i < len(ex_meta) else ''
                p = 'extra-new' if st == 'новый' else 'extra-linked'
        else:
            s, i = '', -1
            p = 'possible' if c >= thr_possible else 'external'
        part.append(p); src.append(s); ref.append(int(i))
    out = dfb.copy()
    out['our_part'], out['match_src'], out['match_idx'] = part, src, ref
    out['corr_org'], out['corr_extra'], out['corr_fig'] = np.round(ca, 4), np.round(cx, 4), np.round(cf, 4)
    log('BRISC -> наше разбиение:', out.our_part.value_counts().to_dict())
    return out


def evaluate(M, dfb, subset, log=print):
    """Прототип целиком (pipeline.analyze) на исходных снимках BRISC выбранной части; Dice — с маской BRISC."""
    from . import pipeline
    from .extras import compact, dice1, wilson
    d = dfb[dfb.our_part.isin(subset)]
    rows = []
    for r in d.itertuples():
        g = np.array(Image.open(r.img).convert('L'))
        res = pipeline.analyze(M, g)
        c = compact(res)
        dc = None
        if r.mask and r.label < 3 and not res['refused']:
            dc = dice1(res['arrays']['mask'], prep_mask(np.array(Image.open(r.mask).convert('L'))))
        rows.append(dict(file=r.file, our_part=r.our_part, label=CLASSES[r.label], ok=c['pred'] == CLASSES[r.label], dice=dc, **c))
    R = pd.DataFrame(rows)
    if not len(R):
        return R, {}
    tum, nor = R[R.label != CLASSES[NO_TUMOR]], R[R.label == CLASSES[NO_TUMOR]]
    ans_n = nor[~nor.refused]
    summ = dict(
        n=len(R), accuracy=round(float(R.ok.mean()), 4), accuracy_ci=wilson(int(R.ok.sum()), len(R)),
        by_class={y: f'{int(g.ok.sum())}/{len(g)}' for y, g in R.groupby('label')},
        tumors=len(tum), tumor_suspected=int(tum.tumor.sum()),
        dice_mean=round(float(tum.dice.dropna().mean()), 4) if tum.dice.notna().any() else None,
        dice_by_class={y: round(float(g.dice.dropna().mean()), 4) for y, g in tum.groupby('label') if g.dice.notna().any()},
        normals=len(nor), normals_refused=int(nor.refused.sum()), normals_answered=len(ans_n),
        normal_false_alarm=int(ans_n.tumor.sum()), normal_false_alarm_ci=wilson(int(ans_n.tumor.sum()), len(ans_n)),
        normal_false_contour=int((ans_n.n_find > 0).sum()),
        normal_auto_sequence=nor.seq.value_counts().to_dict(),
        refused_all=int(R.refused.sum()))
    log(f'BRISC {subset}:', json.dumps(summ, ensure_ascii=False))
    return R, summ


def sequence_agreement(dfb, labels_csv):
    """Наша разметка режима (ассистент) против отбора BRISC (все снимки BRISC — T1) для снимков из пула разметки."""
    lab = pd.read_csv(labels_csv)
    m = dfb[dfb.match_src == 'org'].merge(lab[['idx', 'seq', 'seq_source']], left_on='match_idx', right_on='idx', how='inner')
    if not len(m):
        return {}
    out = {}
    for y, g in m.groupby(m.label == NO_TUMOR):
        out['здоровые' if y else 'опухоли'] = g.seq.value_counts().to_dict()
    out['n'] = len(m)
    return out
