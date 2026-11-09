"""На что опирается классификатор: тепловая карта (Grad-CAM) и проверка «сотрём область — что изменится».

Обе вещи — ориентир для врача и проверка самой системы, а не доказательство:
- Grad-CAM последнего блока ResNet-50 — грубая сетка 8×8 (одна клетка = 32 пикселя сетки 256);
- проверка стиранием: найденную (или экспертную) область закрашиваем по окружению (inpainting) и заново считаем
  классификатор. Если уверенность почти не меняется — ответ опирается на контекст среза (уровень среза, окружение,
  масс-эффект), а не только на саму опухоль.
Проверка на тесте с экспертными масками — notebooks/stage2_explain.ipynb -> results/stage2/explain_check.json.
"""
import cv2
import numpy as np
import torch

from .common import CLASSES, NO_TUMOR
from . import nets


def _T(M):
    return float(M.calib.get('temperature', 1.0)) if len(M.cls) > 1 else 1.0


@torch.no_grad()
def probs(M, xs, dev):
    """(n, 256, 256) uint8 -> (n, 4): вероятности ансамбля так же, как в pipeline.analyze (температура, отражение)."""
    L = np.stack([nets.cls_logits(net, xs, dev) for _, net in M.cls])          # (m, n, 2, 4)
    z = L / _T(M)
    e = np.exp(z - z.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)).mean((0, 2))


def gradcam(M, xs, dev, target, head=None):
    """Grad-CAM ансамбля для класса target (int или по снимку), среднее с отражением -> (n, 256, 256) в [0, 1]."""
    xf = xs[:, :, ::-1].copy()
    acc = 0
    for _, net in M.cls:
        c0, _ = nets.cls_gradcam(net, xs, dev, target=target)
        c1, _ = nets.cls_gradcam(net, xf, dev, target=target)
        acc = acc + c0.astype(np.float32) + c1[:, :, ::-1].astype(np.float32)
    out = np.zeros((len(xs), 256, 256), np.float32)
    for i in range(len(xs)):
        c = np.clip(cv2.resize(acc[i], (256, 256), interpolation=cv2.INTER_CUBIC), 0, None)
        if head is not None:
            c *= cv2.dilate(head[i].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        out[i] = c / (c.max() + 1e-8)
    return out


def erase(x256, region, dil=5):
    """Закрасить область по окружению (inpainting Telea), с запасом dil пикселей."""
    r = cv2.dilate(region.astype(np.uint8), np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))
    return cv2.inpaint(x256, r, 5, cv2.INPAINT_TELEA), r


def mirror_control(region, head):
    """Контроль: область той же формы, отражённая на другую половину и не задевающая исходную."""
    c = region[:, ::-1].copy()
    c &= (1 - cv2.dilate(region.astype(np.uint8), np.ones((11, 11), np.uint8)))
    if head is not None:
        c &= head.astype(np.uint8)
    return c


def occlusion(M, x256, region, dev, P0=None):
    """Стираем область -> как изменились вероятности. P0 — вероятности до (если уже посчитаны)."""
    if P0 is None:
        P0 = probs(M, x256[None], dev)[0]
    xe, _ = erase(x256, region)
    P1 = probs(M, xe[None], dev)[0]
    t = int(np.argmax(P0))
    return dict(cls=CLASSES[t], p_before=round(float(P0[t]), 4), p_after=round(float(P1[t]), 4),
                cls_after=CLASSES[int(np.argmax(P1))], p_tumor_before=round(1 - float(P0[NO_TUMOR]), 4),
                p_tumor_after=round(1 - float(P1[NO_TUMOR]), 4))


def occlusion_text(o):
    """Одна фраза для врача по результату стирания найденной области."""
    d = o['p_before'] - o['p_after']
    head = f'Если стереть найденную область, вероятность «{o["cls"]}» {o["p_before"]:.0%} → {o["p_after"]:.0%}'
    if o['cls_after'] != o['cls']:
        return head + f' (ответ меняется на «{o["cls_after"]}»): тип определён по этой области.'
    if d >= 0.15:
        return head + ': ответ заметно опирается на эту область.'
    return (head + ': ответ почти не зависит от самой области — классификатор опирается на контекст среза '
            '(уровень среза, окружение, смещение структур). Тип опухоли — справочно, оценивайте по снимку.')
