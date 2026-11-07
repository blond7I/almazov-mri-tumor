"""Фильтр входа: система отвечает только на срезы МРТ головы, похожие на те, на которых её проверяли.

Три уровня:
1) детерминированные проверки (размер, цвет, пропорции, фон, наличие головы) — отказ с понятной причиной;
2) «непохожесть» на обучающие данные по признакам классификатора (расстояние Махаланобиса; порог — 99.5-й перцентиль валидации);
3) режим МРТ и плоскость (линейная модель на признаках ResNet-50, обучена на ручной разметке): T2/FLAIR -> «вне области применения»,
   а не «опухоли нет».
"""
import json
from pathlib import Path

import numpy as np

REFUSE, WARN = 'refuse', 'warn'


def basic_checks(g, colorfulness, x256, head):
    """g — исходный серый снимок, x256 — после prep(), head — маска головы. Возвращает список (уровень, код, текст)."""
    out = []
    h, w = g.shape
    s = max(h, w)
    y0, x0 = int(round((s - h) / 2 / s * 256)), int(round((s - w) / 2 / s * 256))
    head = head[y0:256 - y0 or None, x0:256 - x0 or None]          # без чёрных полей, добавленных prep()
    if min(h, w) < 128:
        out.append((REFUSE, 'too_small', f'Слишком маленькое изображение ({w}×{h} пикс.): нужен срез не меньше 128 пикс. по короткой стороне.'))
    if max(h, w) / max(min(h, w), 1) > 2.2:
        out.append((REFUSE, 'aspect', 'Неподходящие пропорции изображения: похоже не на срез, а на коллаж, документ или скриншот.'))
    if colorfulness > 20:
        out.append((REFUSE, 'color', 'Цветное изображение: исходные срезы МРТ — в оттенках серого. Загрузите исходный срез.'))
    elif colorfulness > 8:
        out.append((WARN, 'color', 'Изображение слегка цветное (пометки, наложения?) — результат может быть недостоверен.'))
    b = max(2, min(h, w) // 32)
    border = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
    if np.median(border) > 100:
        out.append((REFUSE, 'light_background', 'Светлый фон по краям: снимок МРТ обычно на тёмном фоне. Возможно, это фото экрана или документ.'))
    frac = float(head.mean())
    if frac < 0.08:
        out.append((REFUSE, 'no_head', 'Не найдено изображение головы (слишком мало ткани на снимке).'))
    elif frac > 0.97:
        out.append((WARN, 'full_frame', 'Изображение заполняет весь кадр — возможно, это фрагмент среза, а не срез целиком.'))
    if max(h, w) <= 256:
        out.append((WARN, 'low_resolution', 'Низкое разрешение (≤ 256 пикс.): точность системы ниже.'))
    return out


class FeatureModels:
    """Признаковые проверки: OOD (Махаланобис) и режим/плоскость (логистическая регрессия). Параметры — в weights/stage2/input_models.npz."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.mu, self.P = z['ood_mu'], z['ood_prec']
        self.thr = float(z['ood_thr'])
        self.pca = z['pca'] if 'pca' in z.files else None
        self.heads = {}
        for k in ['seq', 'plane']:
            if f'{k}_W' in z.files:
                self.heads[k] = (z[f'{k}_W'], z[f'{k}_b'], list(z[f'{k}_classes']), z[f'{k}_mu'], z[f'{k}_sd'])

    def check(self, feat):
        f = feat.astype(np.float64)
        if self.pca is not None:
            f = (f - self.pca[0]) @ self.pca[1:].T
        d = f - self.mu
        dist = float(np.sqrt(max(d @ self.P @ d, 0)))
        res = dict(ood_distance=round(dist, 2), ood_thr=round(self.thr, 2), ood=dist > self.thr)
        for k, (W, b, cls, mu, sd) in self.heads.items():
            z = (feat.astype(np.float64) - mu) / sd
            s = z @ W.T + b
            e = np.exp(s - s.max())
            p = e / e.sum()
            res[k] = cls[int(np.argmax(p))]
            res[k + '_p'] = round(float(p.max()), 3)
            res[k + '_probs'] = {c: round(float(v), 3) for c, v in zip(cls, p)}
        return res


def feature_checks(fm, feat):
    """Превращает результат FeatureModels.check в список (уровень, код, текст)."""
    r = fm.check(feat)
    out = []
    if r['ood']:
        out.append((WARN, 'ood', 'Снимок заметно отличается от тех, на которых система обучалась и проверялась: результат может быть недостоверен.'))
    seq, sp = r.get('seq'), r.get('seq_p', 0)
    if seq in ('T2', 'FLAIR') and sp >= 0.8:
        out.append((REFUSE, 'sequence', f'Похоже на режим {seq} (уверенность {sp:.0%}): система проверена на T1 с контрастом. '
                                       'Вывод «опухоли нет» для этого режима не делается.'))
    elif seq and seq != 'T1c' and sp >= 0.6:
        out.append((WARN, 'sequence', f'Режим, вероятно, {seq} ({sp:.0%}), а не T1 с контрастом — результат менее надёжен.'))
    return out, r
