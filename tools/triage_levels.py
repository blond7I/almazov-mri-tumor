"""Три уровня ответа о наличии опухоли по уже посчитанным построчным результатам прототипа (без новых прогонов моделей).

«найдено» — классификатор видит опухоль И построен контур; «нужен просмотр» — только одно из двух; «не найдено» — ни то, ни другое.
Правило без подбираемых параметров (порог классификатора 0.5 и порог контура — из calib.json). Выбрано после BRISC (9.11),
поэтому BRISC здесь — не независимая проверка; независимая — тест организаторов (здоровые) и дальнейшие прогоны.
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


def levels(d):
    t = d['tumor'].astype(str).eq('True')
    c = d['n_find'].astype(int) > 0
    lev = pd.Series('none', index=d.index)
    lev[t ^ c] = 'review'
    lev[t & c] = 'found'
    return lev


def summ(lev):
    n = len(lev)
    out = dict(n=n)
    for k in ('found', 'review', 'none'):
        m = int((lev == k).sum())
        out[k] = m
        out[k + '_share'] = round(m / n, 4) if n else None
        out[k + '_ci'] = wilson(m, n)
    return out


S = dict(rule='found = classifier tumor (P>=0.5) AND contour; review = exactly one of them; none = neither; refused images excluded',
         note='правило выбрано после просмотра BRISC (9.11); тест организаторов — здоровые срезы, на которых правило не подбиралось')
d = pd.read_csv(R / 'normals_by_sequence.csv')
d = d[~d['refused'].astype(str).eq('True')]
S['organizer_test_normals'] = summ(levels(d))
S['organizer_test_normals_T1'] = summ(levels(d[d['seq_label'] == 'T1']))
d = pd.read_csv(R / 'brisc_rows_external.csv')
d = d[~d['refused'].astype(str).eq('True')]
S['brisc_external_normals'] = summ(levels(d[d['label'] == 'Опухоли нет']))
d = pd.read_csv(R / 'brisc_rows_our_test.csv')
d = d[~d['refused'].astype(str).eq('True')]
tum = d[d['label'] != 'Опухоли нет']
S['brisc_matched_our_test_tumors'] = summ(levels(tum))
S['brisc_matched_our_test_tumors_by_class'] = {k: summ(levels(g)) for k, g in tum.groupby('label')}
dn = pd.read_csv(R / 'normals_by_sequence.csv'); dn = dn[~dn['refused'].astype(str).eq('True')]
de = pd.read_csv(R / 'brisc_rows_external.csv'); de = de[~de['refused'].astype(str).eq('True') & (de['label'] == 'Опухоли нет')]
S['classifier_only_false_alarm'] = dict(
    organizer_test_normals=[int(dn['tumor'].astype(str).eq('True').sum()), len(dn)],
    brisc_external_normals=[int(de['tumor'].astype(str).eq('True').sum()), len(de)])
json.dump(S, open(R / 'triage_levels.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False, indent=1))
