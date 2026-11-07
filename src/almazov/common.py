"""Общие константы и предобработка — одинаковые для обучения, оценки и прототипа."""
import numpy as np
import cv2

# индекс 3 = «опухоли нет»; порядок классов совпадает с обучением этапа 1
CLASSES = ['Глиома', 'Менингиома', 'Опухоль гипофиза', 'Опухоли нет']
NO_TUMOR = 3
MEAN, STD = 0.449 * 255, 0.226 * 255
SIZE = 256


def prep(g):
    """Серый срез любого размера -> квадрат с чёрными полями -> 256x256 -> контраст по перцентилям 0.5–99.5.

    Ровно та же функция, что на этапе 1 (обучение и демо), менять нельзя: веса обучены на ней.
    """
    h, w = g.shape
    s = max(h, w)
    if h != w:
        c = np.zeros((s, s), g.dtype)
        c[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = g
        g = c
    g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA if s > SIZE else cv2.INTER_LINEAR).astype(np.float32)
    lo, hi = np.percentile(g, 0.5), np.percentile(g, 99.5)
    return (np.clip((g - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def prep_geometry(h, w):
    """Как пиксели исходного снимка соотносятся с сеткой 256x256 после prep().

    Возвращает (scale, off_y, off_x): пиксель (y, x) сетки 256 соответствует пикселю исходного снимка
    ((y + 0.5) * scale - 0.5 - off_y, (x + 0.5) * scale - 0.5 - off_x); площадь 1 пикселя сетки = scale**2 исходных.
    """
    s = max(h, w)
    return s / SIZE, (s - h) // 2, (s - w) // 2


def d8(x, t):
    """8 поворотов/отражений квадрата (t = 0..7)."""
    y = np.rot90(x, t // 2)
    return np.fliplr(y) if t % 2 else y


def small(arr, s=64):
    if len(arr) == 0:
        return np.zeros((0, s, s), np.float32)
    return np.stack([cv2.resize(a, (s, s), interpolation=cv2.INTER_AREA) for a in arr]).astype(np.float32)


def z1(v):
    v = v.ravel() - v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def clean_mask(pm, thr=0.5, min_area=30):
    """Правило прототипа этапа 1: порог по вероятности и выброс пятен меньше min_area пикселей (сетка 256)."""
    m = (pm > thr).astype(np.uint8)
    k, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for c in range(1, k):
        if st[c, cv2.CC_STAT_AREA] < min_area:
            m[lab == c] = 0
    return m


def dice(p, m):
    """Dice по каждому срезу; p, m — булевы (n, H, W). Пустой и пустой = 1."""
    inter = (p & m).sum((1, 2))
    den = p.sum((1, 2)) + m.sum((1, 2))
    return (2 * inter + 1e-6) / (den + 1e-6)
