"""Исследование целиком: серия DICOM -> по каждому срезу вероятность опухоли и контур -> сводка по исследованию.

Быстрый проход — модели этапа 1 (1 классификатор + 1 сегментатор), затем ключевой срез (наибольшая площадь контура)
анализируется полным ансамблем, и по нему строится проект заключения. Объём — оценка по контурам на 2D-срезах
(сумма площадей × шаг между срезами): модели обучены на отдельных срезах, поэтому это ориентир, а не измерение.
"""
import os
import tempfile
import zipfile
from dataclasses import replace

import numpy as np

from . import pipeline
from .common import CLASSES, NO_TUMOR


def collect(paths):
    """Пути к файлам и/или zip-архивам -> список путей к DICOM-файлам."""
    out = []
    for p in paths:
        if str(p).lower().endswith('.zip'):
            d = tempfile.mkdtemp(prefix='series_')
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith('/') or os.path.basename(n).startswith('.'):
                        continue
                    z.extract(n, d)
                    out.append(os.path.join(d, n))
        else:
            out.append(str(p))
    return out


def load(paths):
    """-> список (путь, изображение, сведения), упорядоченный вдоль серии; файлы, которые не читаются как DICOM, пропускаются."""
    from . import dicom
    items = []
    for p in collect(paths):
        try:
            img, info = dicom.read(p)
        except Exception:
            continue
        items.append((p, img, info))
    if not items:
        return []
    o = items[0][2].get('orientation')
    if o and all(it[2].get('position') for it in items):
        n = np.cross(np.array(o[:3]), np.array(o[3:]))
        key = lambda it: float(np.dot(n, it[2]['position']))
    elif all(it[2].get('instance') is not None for it in items):
        key = lambda it: it[2]['instance']
    else:
        key = lambda it: os.path.basename(it[0])
    return sorted(items, key=key)


def slice_gap(items):
    """Шаг между срезами, мм: по позициям, иначе SpacingBetweenSlices, иначе толщина среза; None — неизвестен."""
    o = items[0][2].get('orientation') if items else None
    if o and len(items) > 1 and all(it[2].get('position') for it in items):
        n = np.cross(np.array(o[:3]), np.array(o[3:]))
        z = np.array([np.dot(n, it[2]['position']) for it in items])
        d = np.abs(np.diff(z))
        d = d[d > 1e-3]
        if len(d):
            return float(np.median(d))
    for k in ('spacing_between', 'slice_thickness'):
        v = items[0][2].get(k) if items else None
        if v:
            return float(v)
    return None


def analyze(M, paths, fast=True, progress=None):
    items = load(paths)
    if not items:
        return None
    Mf = replace(M, cls=M.cls[:1], seg=M.seg[:1]) if fast and (len(M.cls) > 1 or len(M.seg) > 1) else M
    gap = slice_gap(items)
    rows, areas, overlays = [], [], {}
    for k, (p, img, info) in enumerate(items):
        if progress:
            progress((k + 1) / len(items), desc=f'Срез {k + 1} из {len(items)}')
        r = pipeline.analyze(Mf, img, pixel_spacing_mm=info.get('pixel_spacing_mm'))
        a_mm2 = sum(f.get('area_mm2', 0.0) for f in r['findings'])
        a_px = sum(f['area_px_orig'] for f in r['findings'])
        rows.append(dict(n=k + 1, file=os.path.basename(p), refused=r['refused'], p_tumor=r['p_tumor'], tumor=r['tumor'],
                         top=r['top_type'], findings=len(r['findings']), area_mm2=round(a_mm2, 1), area_px=a_px,
                         probs=r['probs']))
        areas.append(a_mm2 if a_mm2 else a_px)
        if r['findings'] and not r['refused']:
            overlays[k + 1] = pipeline.overlay(r)
    ans = [x for x in rows if not x['refused']]
    pos = [x for x in ans if x['tumor'] and x['findings']]
    key = max(range(len(rows)), key=lambda i: (rows[i]['findings'] > 0 and not rows[i]['refused'], areas[i], rows[i]['p_tumor']))
    p, img, info = items[key]
    key_res = pipeline.analyze(M, p)          # полный ансамбль, DICOM: мм и сторона пациента
    spacing = items[0][2].get('pixel_spacing_mm')
    vol = round(sum(x['area_mm2'] for x in pos) * gap / 1000, 1) if (pos and gap and spacing) else None
    P = np.mean([[x['probs'][c] for c in CLASSES] for x in pos], 0) if pos else None
    summ = dict(slices=len(rows), refused=len(rows) - len(ans), with_tumor=len(pos),
                range=[pos[0]['n'], pos[-1]['n']] if pos else None, key_slice=key + 1, slice_gap_mm=gap,
                volume_ml=vol, type_probs={c: round(float(P[i]), 3) for i, c in enumerate(CLASSES)} if P is not None else None,
                plane=items[0][2].get('plane'), fast=bool(Mf is not M))
    return dict(rows=rows, summary=summ, key_result=key_res, items=items, overlays=overlays)


def summary_text(S):
    s = S['summary']
    t = [f'Серия: {s["slices"]} срезов' + (f', шаг {s["slice_gap_mm"]:.1f} мм' if s['slice_gap_mm'] else '') +
         (f', плоскость {s["plane"]}' if s['plane'] else '') + '.']
    if s['refused']:
        t.append(f'Не проанализировано срезов: {s["refused"]} (вне области применения или не срез МРТ).')
    if s['with_tumor']:
        tp = s['type_probs']
        top = max([c for c in CLASSES if c != CLASSES[NO_TUMOR]], key=lambda c: tp[c])
        t.append(f'Область повышенной вероятности опухоли — на {s["with_tumor"]} срезах (с {s["range"][0]} по {s["range"][1]}); '
                 f'наибольшая площадь — на срезе {s["key_slice"]}.')
        if s['volume_ml'] is not None:
            t.append(f'Оценочный объём по контурам на срезах: ≈ {s["volume_ml"]} мл (ориентир: модели обучены на отдельных срезах).')
        t.append(f'По срезам с находкой наиболее вероятен тип: {top} ({tp[top]:.0%}).')
    else:
        t.append('Ни на одном срезе система не выделила область опухоли.')
    if s['fast']:
        t.append('Проход по серии — быстрыми моделями этапа 1; ключевой срез — полным ансамблем.')
    return '\n'.join(t)


def curve_image(S):
    """Картинка: вероятность опухоли и площадь контура по срезам."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = S['rows']
    n = [r['n'] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 2.6), dpi=110)
    ax.plot(n, [r['p_tumor'] for r in rows], color='#d62839', lw=2, label='вероятность опухоли')
    ax.set_ylim(0, 1.02); ax.set_xlabel('срез'); ax.set_ylabel('вероятность')
    ax2 = ax.twinx()
    ar = [r['area_mm2'] or r['area_px'] for r in rows]
    ax2.bar(n, ar, color='#e9a23b', alpha=.45, label='площадь контура')
    ax2.set_ylabel('площадь, мм²' if any(r['area_mm2'] for r in rows) else 'площадь, пикс.')
    for r in rows:
        if r['refused']:
            ax.axvspan(r['n'] - .5, r['n'] + .5, color='#8d99ae', alpha=.25, lw=0)
    ax.axvline(S['summary']['key_slice'], color='#3d5a80', ls='--', lw=1)
    fig.text(0.5, 0.01, 'серая заливка — срез не проанализирован; пунктир — ключевой срез', ha='center', fontsize=8, color='#5c677d')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.canvas.draw()
    im = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return im
