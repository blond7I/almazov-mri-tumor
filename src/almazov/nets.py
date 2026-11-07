"""Создание/загрузка моделей и пакетный вывод (классификатор, сегментация, признаки, Grad-CAM)."""
import contextlib
import hashlib
import os

import numpy as np
import torch

from .common import MEAN, STD


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def amp(dev):
    return torch.autocast('cuda') if dev.type == 'cuda' else contextlib.nullcontext()


def make_cls(arch='resnet50', pretrained=False):
    import timm
    return timm.create_model(arch, pretrained=pretrained, num_classes=4)


def make_seg(pretrained=False):
    import segmentation_models_pytorch as smp
    return smp.Unet('resnet34', encoder_weights='imagenet' if pretrained else None, in_channels=1, classes=1)


def load_state(path):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    return {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}


def half_state(sd):
    """Веса в float16 (вдвое меньше файл; для вывода разница пренебрежимо мала)."""
    return {k: (v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu()) for k, v in sd.items()}


def assemble(parts, dst, sha256=None, sha_prefix=None):
    """Склеить файл весов из частей по 20 МБ (так они лежат в репозитории) и проверить контрольную сумму."""
    if not os.path.exists(dst):
        with open(dst, 'wb') as out:
            for p in parts:
                with open(p, 'rb') as f:
                    out.write(f.read())
    h = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
    if sha256 and h != sha256 or sha_prefix and not h.startswith(sha_prefix):
        os.remove(dst)
        raise RuntimeError(f'Контрольная сумма не совпала: {dst}')
    return dst


def to_x(u8, dev, ch=1):
    x = torch.from_numpy((u8.astype(np.float32) - MEAN) / STD)[:, None]
    if ch == 3:
        x = x.repeat(1, 3, 1, 1)
    return x.to(dev, non_blocking=True)


@torch.no_grad()
def cls_logits(net, imgs, dev, bs=64):
    """Логиты (n, 2, 4): [исходный снимок, отражение слева-направо] — для калибровки температурой и ансамблей."""
    net.eval()
    out = []
    for s in range(0, len(imgs), bs):
        x = to_x(imgs[s:s + bs], dev, 3)
        with amp(dev):
            a = net(x).float()
            b = net(torch.flip(x, (3,))).float()
        out.append(torch.stack([a, b], 1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 2, 4), np.float32)


def softmax_tta(L):
    """Вероятности так же, как на этапе 1: среднее softmax по исходному и отражённому снимку."""
    e = np.exp(L - L.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)).mean(1)


@torch.no_grad()
def cls_features(net, imgs, dev, bs=64):
    """Признаки 2048 после глобального пулинга (для фильтра входа и классификатора режима МРТ)."""
    net.eval()
    out = []
    for s in range(0, len(imgs), bs):
        x = to_x(imgs[s:s + bs], dev, 3)
        with amp(dev):
            f = net.forward_head(net.forward_features(x), pre_logits=True).float()
        out.append(f.cpu().half())
    return torch.cat(out).numpy() if out else np.zeros((0, 2048), np.float16)


def cls_gradcam(net, imgs, dev, bs=32):
    """Grad-CAM последнего блока ResNet для предсказанного класса: (n, 8, 8) float16 в [0, 1] и класс."""
    net.eval()
    cams, preds = [], []
    for s in range(0, len(imgs), bs):
        x = to_x(imgs[s:s + bs], dev, 3)
        with torch.enable_grad():
            f = net.forward_features(x).float().detach().requires_grad_(True)
            logits = net.forward_head(f)
            c = logits.argmax(1)
            g, = torch.autograd.grad(logits.gather(1, c[:, None]).sum(), f)
        w = g.mean((2, 3), keepdim=True)
        cam = torch.relu((w * f).sum(1))
        cam = cam / (cam.flatten(1).max(1).values[:, None, None] + 1e-8)
        cams.append(cam.detach().cpu().half())
        preds.append(c.cpu())
    if not cams:
        return np.zeros((0, 8, 8), np.float16), np.zeros(0, int)
    return torch.cat(cams).numpy(), torch.cat(preds).numpy()


@torch.no_grad()
def seg_maps(net, imgs, dev, bs=32, flip='mean'):
    """Карты вероятности опухоли (n, 256, 256) float32.

    flip='mean' — среднее по исходному и отражённому (как на этапе 1); 'both' — (n, 2, H, W) по отдельности.
    """
    net.eval()
    out = []
    for s in range(0, len(imgs), bs):
        x = to_x(imgs[s:s + bs], dev, 1)
        with amp(dev):
            a = torch.sigmoid(net(x).float())
            b = torch.flip(torch.sigmoid(net(torch.flip(x, (3,))).float()), (3,))
        r = torch.cat([a, b], 1) if flip == 'both' else (a + b)[:, 0] / 2
        out.append(r.cpu())
    if not out:
        return np.zeros((0, 2, 256, 256) if flip == 'both' else (0, 256, 256), np.float32)
    return torch.cat(out).numpy()


def to_u8(p):
    return np.round(np.clip(p, 0, 1) * 255).astype(np.uint8)
