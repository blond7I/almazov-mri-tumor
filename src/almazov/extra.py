"""Дополнительная проверка на снимках, которых не было в данных этапа 1.

Полный архив организаторов (ноябрь 2026) содержит 12 064 снимка; на этапе 1 было доступно 9 150. Остальные 2 914
модели не видели. Но в наборе много копий и аугментаций одного снимка, поэтому «новыми» считаем только те, у которых
нет почти точной копии (корреляция ≥ 0.98 с учётом 8 поворотов/отражений) и нет общего пациента Figshare или общего
базового имени с обучающими/валидационными снимками — тем же правилом, что при честном разбиении этапа 1.
"""
import re

import numpy as np
import pandas as pd
import torch

from .common import NO_TUMOR, small

THR = 0.98


def _zt(x):
    x = x.reshape(len(x), -1)
    x = x - x.mean(1, keepdim=True)
    return x / (x.norm(dim=1, keepdim=True) + 1e-6)


def _d8_t(x, t):
    y = torch.rot90(x, t // 2, (1, 2))
    return torch.flip(y, (2,)) if t % 2 else y


def _best_match(Q, R, dev, block=512):
    """Для каждого снимка Q: максимум корреляции (64x64, 8 ориентаций) по всем снимкам R и его индекс."""
    Rz = _zt(torch.tensor(small(R))).to(dev)
    Q64 = torch.tensor(small(Q))
    QD = [_zt(_d8_t(Q64, t)).to(dev) for t in range(8)]
    best = np.zeros(len(Q), np.float32)
    arg = np.zeros(len(Q), int)
    for s in range(0, len(Q), block):
        c = torch.stack([QD[t][s:s + block] @ Rz.T for t in range(8)]).max(0).values
        v, a = c.max(1)
        best[s:s + block], arg[s:s + block] = v.cpu().numpy(), a.cpu().numpy()
    return best, arg


def _augbase(nm):
    m = re.match(r'^([PGMN]_\d+)(?:_[A-Z]{2}_)?\.', nm)
    return m.group(1) if m else ''


def link_extra(df, A, fX, fy, fpid, S, ex_df, AX, dev=None, log=print):
    """Связи дополнительных снимков с данными этапа 1 -> ex_df с колонками status/group/pid/fig_i."""
    dev = dev or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    e = ex_df.copy().reset_index(drop=True)
    if len(e) == 0:
        return e
    part = df.part.values
    co, ao = _best_match(AX, A, dev)
    cf, af = _best_match(AX, fX, dev)
    e['corr_org'], e['nn_org'], e['nn_org_part'] = co, ao, part[ao]
    e['nn_org_label'] = df.label.values[ao]
    fig_ok = (e.label.values != NO_TUMOR) & (fy[af] == e.label.values) & (cf >= THR)
    e['corr_fig'], e['fig_i'] = cf, np.where(fig_ok, af, -1)
    e['pid'] = np.where(fig_ok, fpid[af], '')
    e['fig_part'] = np.where(fig_ok, S['fpart'][af], '')
    base_part = {}
    for b_, p_ in zip(df.name.map(_augbase), part):
        if b_:
            base_part.setdefault(b_, set()).add(p_)
    e['augbase'] = e.name.map(_augbase)
    e['base_parts'] = e.augbase.map(lambda b_: ','.join(sorted(base_part.get(b_, set()))) if b_ else '')

    def status(r):
        hits = set()
        if r.corr_org >= THR:
            hits.add(r.nn_org_part)
        if r.fig_part:
            hits.add(r.fig_part)
        if r.base_parts:
            hits |= set(r.base_parts.split(','))
        if hits & {'train', 'val'}:
            return 'связан с обучением/валидацией'
        if 'test' in hits:
            return 'копия тестового снимка'
        return 'новый'
    e['status'] = e.apply(status, axis=1)
    e['label_conflict_nn'] = (e.corr_org >= THR) & (e.nn_org_label != e.label)
    # группы среди дополнительных снимков (копии, один пациент, одно базовое имя) — для бутстрэпа по группам
    par = list(range(len(e)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    def un(a, b):
        a, b = find(a), find(b)
        if a != b:
            par[a] = b
    ce, ae = _best_match_self(AX, dev)
    for i, (c, j) in enumerate(zip(ce, ae)):
        if c >= THR:
            un(i, int(j))
    for col in ['pix', 'pid', 'augbase']:
        for ids in e[e[col] != ''].groupby(col).groups.values():
            ids = list(ids)
            for i in ids[1:]:
                un(ids[0], i)
    e['group'] = [find(i) for i in range(len(e))]
    st = e.status.value_counts().to_dict()
    nw = e[e.status == 'новый']
    log(f'Снимки вне разбиения: {len(e)} -> {st}; новых групп: {nw.group.nunique()}; '
        f'новых по классам: {np.bincount(nw.label.clip(0), minlength=4).tolist()}')
    return e


def _best_match_self(X, dev, block=512):
    """Ближайший другой снимок внутри того же набора (для групп копий)."""
    Z = _zt(torch.tensor(small(X))).to(dev)
    X64 = torch.tensor(small(X))
    XD = [_zt(_d8_t(X64, t)).to(dev) for t in range(8)]
    best = np.zeros(len(X), np.float32)
    arg = np.zeros(len(X), int)
    for s in range(0, len(X), block):
        c = torch.stack([XD[t][s:s + block] @ Z.T for t in range(8)]).max(0).values
        idx = torch.arange(s, min(s + block, len(X)), device=dev)
        c[idx - s, idx] = -1
        v, a = c.max(1)
        best[s:s + block], arg[s:s + block] = v.cpu().numpy(), a.cpu().numpy()
    return best, arg
