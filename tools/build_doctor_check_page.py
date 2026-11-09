"""Страница для врача: вслепую указать режим МРТ и плоскость на 50 снимках (проверка разметки, сделанной ассистентом).

    python tools/build_doctor_check_page.py IMAGES_DIR data/sequence_plane_labels.csv OUT.html

Состав (seed 11): все 21 здоровых снимков теста с меткой «T1» (главный вопрос — есть ли контраст),
15 здоровых снимков теста других режимов, 14 снимков с опухолью из источников без описания режима.
Наши метки на странице не показываются. Ответы — в хранилище артефакта, коллекция doctor_check.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_label_page import PAGE, load_images     # noqa: E402

REPL = [
    ('<title>Разметка режимов МРТ</title>', '<title>Проверка режимов МРТ</title>'),
    ('<h1>Режим МРТ и плоскость среза</h1>', '<h1>Проверка врачом: режим МРТ и плоскость</h1>'),
    ('const SEQ = [["T1c","T1 + контраст","1"],["T1","T1","2"],["T2","T2","3"],["FLAIR","FLAIR","4"],["unk","Не знаю","5"]];',
     'const SEQ = [["T1c","T1 + контраст","1"],["T1","T1 без контраста","2"],["T2","T2","3"],["FLAIR","FLAIR","4"],["CT","КТ","5"],["unk","Не знаю","6"]];'),
    ('const KEY = "mri-labels-v1";', 'const KEY = "mri-doctor-check-v1";'),
    ('db.collection("labels")', 'db.collection("doctor_check")'),
    ('Все снимки размечены. Спасибо! Можно закрыть страницу — ответы сохранены.',
     'Готово, спасибо! Все 50 ответов сохранены.'),
]


def main():
    img_dir, labels_csv, out = sys.argv[1:4]
    d = pd.read_csv(labels_csv)
    rs = np.random.RandomState(11)
    tn = d[(d.part == 'test') & (d.label == 3)]
    pick = list(tn[tn.seq == 'T1'].idx)
    for s, k in [('T2', 5), ('FLAIR', 5), ('КТ', 3), ('другое/не ясно', 2)]:
        c = tn[tn.seq == s].idx.values
        pick += list(rs.choice(c, min(k, len(c)), replace=False))
    vt = d[(d.label < 3) & d.seq_source.str.startswith('визуально')].idx.values
    pick += list(rs.choice(vt, 50 - len(pick), replace=False))
    pick = [int(i) for i in pick]
    imgs = load_images(img_dir, pick)
    order = rs.permutation(len(pick))
    items = [dict(i=pick[k], b=imgs[pick[k]]) for k in order]
    html = PAGE
    for a, b in REPL:
        assert a in html, a
        html = html.replace(a, b)
    html = html.replace('__ITEMS__', json.dumps(items, separators=(',', ':')))
    Path(out).write_text(html, encoding='utf-8')
    Path(out).with_suffix('.ids.json').write_text(json.dumps(pick))
    print(f'{out}: {len(items)} снимков, {len(html) / 1e6:.2f} МБ')


if __name__ == '__main__':
    main()
