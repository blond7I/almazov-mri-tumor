"""Три уровня ответа о наличии опухоли по построчным результатам прототипа (без новых прогонов моделей).

«найдено» — классификатор видит опухоль И построен контур; «нужен просмотр» — только одно из двух; «не найдено» — ни то, ни другое.
Правило без подбираемых параметров (порог классификатора 0.5 и порог контура — из calib.json). Придумано после BRISC (9.11).

Итоговый ансамбль (v3) включает 2 классификатора, обученных на половине внешних здоровых снимков BRISC (эксперимент C),
поэтому здоровые BRISC считаются ТОЛЬКО по отложенной половине (results/stage2/brisc_normals_split.csv); для сравнения
«до» — ансамбль v2 на тех же отложенных снимках (файлы *_v2.csv).
    python tools/triage_levels.py  ->  results/stage2/triage_levels.json
"""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / 'results' / 'stage2'


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def answered(d):
    return d[~d['refused'].astype(str).eq('True')]


def levels(d):
    t = d['tumor'].astype(str).eq('True')
    c = d['n_find'].astype(int) > 0
    lev = pd.Series('none', index=d.index)
    lev[t ^ c] = 'review'
    lev[t & c] = 'found'
    return lev


def summ(d):
    """d — все строки набора (с отказами); уровни считаются по отвеченным."""
    a = answered(d)
    lev = levels(a)
    n = len(a)
    out = dict(n_all=len(d), refused=len(d) - n, n=n, classifier_said_tumor=int(a['tumor'].astype(str).eq('True').sum()))
    for k in ('found', 'review', 'none'):
        m = int((lev == k).sum())
        out[k] = m
        out[k + '_share'] = round(m / n, 4) if n else None
        out[k + '_ci'] = wilson(m, n)
    return out


def brisc_sets(suffix=''):
    ext = pd.read_csv(R / f'brisc_rows_external{suffix}.csv')
    ext = ext[ext['label'] == 'Опухоли нет']
    split = R / 'brisc_normals_split.csv'
    if split.exists():
        s = pd.read_csv(split)
        ext = ext[ext['file'].isin(set(s.file[s.part == 'holdout']))]
    te = pd.read_csv(R / f'brisc_rows_our_test{suffix}.csv')
    return ext, te[te['label'] != 'Опухоли нет']


def block(suffix=''):
    ext, tum = brisc_sets(suffix)
    out = dict(brisc_external_normals=summ(ext), brisc_matched_our_test_tumors=summ(tum),
               brisc_matched_our_test_tumors_by_class={k: summ(g) for k, g in tum.groupby('label')})
    f = R / f'normals_by_sequence{suffix}.csv'
    if f.exists():
        d = pd.read_csv(f)
        out['organizer_test_normals'] = summ(d)
        out['organizer_test_normals_T1'] = summ(d[d['seq_label'] == 'T1'])
    return out


S = dict(rule='found = classifier tumor (P>=0.5) AND contour; review = exactly one of them; none = neither; refused images excluded',
         note='здоровые BRISC — только отложенная половина (не участвовала в обучении ни одной модели); правило выбрано после BRISC (9.11)',
         holdout_only=(R / 'brisc_normals_split.csv').exists())
S.update(block(''))
if (R / 'brisc_rows_external_v2.csv').exists():
    S['before_v2'] = block('_v2')
json.dump(S, open(R / 'triage_levels.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False, indent=1))
