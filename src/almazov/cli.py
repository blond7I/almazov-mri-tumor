"""Пакетная проверка из командной строки.

    python -m almazov.cli ПАПКА_ИЛИ_ZIP --out results_batch [--repo .] [--device cpu]

Результат: batch_results.csv (класс, вероятности, находки, флаги), картинки с контуром, JSON по каждому снимку.
Если снимки разложены по папкам классов, считаются точность, чувствительность и специфичность.
"""
import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import CLASSES, NO_TUMOR
from . import pipeline

EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
KEYS = {0: ['glioma', 'глиома'], 1: ['meningioma', 'менингиома'], 2: ['pituitary', 'гипофиз', 'опухоль гипофиза'],
        3: ['notumor', 'no_tumor', 'no tumor', 'normal', 'нет опухоли', 'опухоли нет', 'норма']}


def true_label(path):
    for part in [os.path.basename(os.path.dirname(path)), os.path.basename(path)]:
        t = part.strip().lower()
        for c, ks in KEYS.items():
            if any(t.startswith(k) for k in ks):
                return c
    return None


def unzip(path, dst):
    shutil.rmtree(dst, ignore_errors=True)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                raw = name.encode('cp437')
                try:
                    name = raw.decode('utf-8')
                except UnicodeDecodeError:
                    name = raw.decode('cp866')
            target = os.path.join(dst, name)
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as s, open(target, 'wb') as o:
                o.write(s.read())
    return dst


def run(inp, out, repo, device='cpu', limit=None, log=print):
    out = Path(out); (out / 'overlays').mkdir(parents=True, exist_ok=True); (out / 'json').mkdir(exist_ok=True)
    if str(inp).lower().endswith('.zip'):
        inp = unzip(inp, str(out / '_input'))
    files = sorted(str(p) for p in Path(inp).rglob('*') if p.suffix.lower() in EXT)[:limit]
    M = pipeline.load_models(repo, device=device)
    log(f'Модели: {M.version}; снимков: {len(files)}')
    rows, t0 = [], time.time()
    for i, f in enumerate(files):
        r = pipeline.analyze(M, f, source_name=os.path.relpath(f, inp))
        y = true_label(f)
        pred = CLASSES.index(r['top_type']) if r['tumor'] else NO_TUMOR
        stem = f'{i:04d}_{Path(f).stem}'
        cv2.imwrite(str(out / 'overlays' / f'{stem}.png'), cv2.cvtColor(pipeline.overlay(r), cv2.COLOR_RGB2BGR))
        json.dump(pipeline.to_json(r), open(out / 'json' / f'{stem}.json', 'w'), ensure_ascii=False, indent=1)
        rows.append({'file': r['source'], 'prediction': CLASSES[pred], 'p_tumor': r['p_tumor'],
                     **{f'p_{c}': r['probs'][c] for c in CLASSES},
                     'findings': len(r['findings']), 'review_zones': len(r['review_zones']),
                     'area_%_head': sum(x['area_pct_head'] for x in r['findings']),
                     'flags': ';'.join(r['flags']),
                     'true': CLASSES[y] if y is not None else '', 'correct': (pred == y) if y is not None else ''})
    df = pd.DataFrame(rows)
    df.to_csv(out / 'batch_results.csv', index=False, encoding='utf-8-sig')
    sec = (time.time() - t0) / max(len(files), 1)
    log(f'Готово: {len(df)} снимков, {sec:.2f} с на снимок. Таблица: {out / "batch_results.csv"}')
    lab = df[df.true != '']
    if len(lab):
        yt, yp = lab.true != CLASSES[NO_TUMOR], lab.prediction != CLASSES[NO_TUMOR]
        log(f'Размечено: {len(lab)}. Точность (4 класса): {(lab.prediction == lab.true).mean():.1%}')
        if yt.sum():
            log(f'Чувствительность «опухоль есть»: {(yt & yp).sum() / yt.sum():.1%}')
        if (~yt).sum():
            log(f'Специфичность: {(~yt & ~yp).sum() / (~yt).sum():.1%}')
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description='Пакетная проверка срезов МРТ')
    ap.add_argument('input')
    ap.add_argument('--out', default='results_batch')
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args(argv)
    run(a.input, a.out, a.repo, a.device, a.limit)


if __name__ == '__main__':
    main()
