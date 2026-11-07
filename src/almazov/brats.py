"""S5 (доп. пункт 3.1): компоненты глиальной опухоли — отёк, некротическое/неконтрастируемое ядро, контрастируемая часть.

Данные: Medical Segmentation Decathlon, Task01_BrainTumour (484 пациента, BraTS 2016–2017; FLAIR, T1, T1gd, T2; разметка:
1 — перифокальный отёк, 2 — неконтрастируемая часть, 3 — контрастируемая часть). Лицензия CC BY-SA 4.0.
Две 2D-модели (U-Net ResNet34) на аксиальных срезах, разбиение по пациентам:
  A) T1gd + FLAIR -> области BraTS: WT (вся опухоль с отёком), TC (ядро), ET (контрастируемая часть);
  B) только T1gd -> те же области — чтобы показать, что отёк (WT) по одному T1 с контрастом надёжно не оценить.
Метрики — Dice по пациенту (3D, по всем срезам тома) на отложенных пациентах.
"""
import glob
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

URL = 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar'   # адрес, который использует MONAI
MIRROR = 'https://huggingface.co/datasets/Sp-data/MSD/resolve/main/Task01_BrainTumour.zip'
MOD = dict(flair=0, t1=1, t1gd=2, t2=3)
REGIONS = ['WT', 'TC', 'ET']


def download(dst_dir, log=print):
    import urllib.request
    d = Path(dst_dir); d.mkdir(parents=True, exist_ok=True)
    tar = d / 'Task01_BrainTumour.tar'
    root = d / 'Task01_BrainTumour'
    if not root.exists():
        if not tar.exists():
            log('Скачиваю MSD Task01_BrainTumour (~7 ГБ)...')
            try:
                urllib.request.urlretrieve(URL, tar)
            except Exception as e:      # запасное зеркало (тот же архив MSD в zip)
                log(f'  основной адрес недоступен ({type(e).__name__}), пробую зеркало Hugging Face')
                tar = d / 'Task01_BrainTumour.zip'
                urllib.request.urlretrieve(MIRROR, tar)
        log('Распаковываю...')
        if tar.suffix == '.zip':
            import zipfile
            with zipfile.ZipFile(tar) as z:
                z.extractall(d)
        else:
            with tarfile.open(tar) as t:
                t.extractall(d)
        tar.unlink()
        if not root.exists():
            hit = glob.glob(str(d / '**' / 'Task01_BrainTumour'), recursive=True)
            if hit:
                root = Path(hit[0])
    return root


def norm(x):
    v = x[x > 0]
    if v.size < 50:
        return np.zeros_like(x, np.uint8)
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    return (np.clip((x - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def to256(a, interp=cv2.INTER_LINEAR):
    """240x240 -> 256x256 полями (как prep: квадрат без искажения), затем без масштабирования."""
    h, w = a.shape
    out = np.zeros((256, 256), a.dtype)
    out[(256 - h) // 2:(256 - h) // 2 + h, (256 - w) // 2:(256 - w) // 2 + w] = a
    return out


def regions(lab):
    """Метки MSD (1 отёк, 2 неконтрастируемое, 3 контрастируемое) -> 3 бинарных канала WT/TC/ET."""
    return np.stack([lab > 0, (lab == 2) | (lab == 3), lab == 3]).astype(np.uint8)


def load_slices(root, n_tumor=24, n_empty=4, seed=0, log=print):
    """По каждому пациенту: случайные срезы с опухолью + несколько без неё (обучение), все срезы с мозгом (оценка)."""
    import nibabel as nib
    imgs = sorted(glob.glob(str(Path(root) / 'imagesTr' / 'BRATS_*.nii.gz')))
    rs = np.random.RandomState(seed)
    pids = [Path(p).name.replace('.nii.gz', '') for p in imgs]
    perm = rs.permutation(len(pids))
    n = len(pids)
    part = np.empty(n, object)
    part[perm[:int(.8 * n)]] = 'train'; part[perm[int(.8 * n):int(.9 * n)]] = 'val'; part[perm[int(.9 * n):]] = 'test'
    data = {k: dict(X=[], Y=[], pid=[]) for k in ['train', 'val', 'test']}
    t0 = time.time()
    for k, (p, pid) in enumerate(zip(imgs, pids)):
        v = nib.load(p).get_fdata(dtype=np.float32)               # 240x240x155x4
        lab = nib.load(p.replace('imagesTr', 'labelsTr')).get_fdata(dtype=np.float32).astype(np.uint8)
        brain = (v[..., MOD['flair']] > 0).reshape(-1, v.shape[2]).sum(0) > 500
        tum = (lab > 0).reshape(-1, v.shape[2]).sum(0) > 30
        z_all = np.where(brain)[0]
        if part[k] == 'train':
            zt, ze = np.where(tum)[0], np.where(brain & ~tum)[0]
            z = np.concatenate([rs.choice(zt, min(n_tumor, len(zt)), replace=False) if len(zt) else [],
                                rs.choice(ze, min(n_empty, len(ze)), replace=False) if len(ze) else []]).astype(int)
        else:
            z = z_all
        for zi in z:
            fl = to256(norm(np.rot90(v[:, :, zi, MOD['flair']])))
            gd = to256(norm(np.rot90(v[:, :, zi, MOD['t1gd']])))
            y = np.stack([to256(r) for r in regions(np.rot90(lab[:, :, zi]))])
            data[part[k]]['X'].append(np.stack([gd, fl])); data[part[k]]['Y'].append(y); data[part[k]]['pid'].append(pid)
        if k % 50 == 0:
            log(f'  {k + 1}/{n} пациентов, {time.time() - t0:.0f} с')
    for k in data:
        data[k] = dict(X=np.stack(data[k]['X']), Y=np.stack(data[k]['Y']), pid=np.array(data[k]['pid']))
    log('Срезов: ' + ', '.join(f'{k} {len(v["X"])}' for k, v in data.items()) +
        f'; пациентов: {sum(part == "train")}/{sum(part == "val")}/{sum(part == "test")}')
    return data


class DS(Dataset):
    def __init__(s, X, Y, ch, aug):
        s.X, s.Y, s.ch, s.aug = X, Y, ch, aug

    def __len__(s):
        return len(s.X)

    def __getitem__(s, i):
        import random
        x = s.X[i][s.ch].astype(np.float32)
        y = s.Y[i].astype(np.float32)
        if s.aug:
            if random.random() < .5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            M_ = cv2.getRotationMatrix2D((128, 128), random.uniform(-15, 15), random.uniform(.9, 1.1))
            x = np.stack([cv2.warpAffine(np.ascontiguousarray(c), M_, (256, 256)) for c in x])
            y = np.stack([cv2.warpAffine(np.ascontiguousarray(c), M_, (256, 256), flags=cv2.INTER_NEAREST) for c in y])
            x = x * random.uniform(.85, 1.15) + random.uniform(-15, 15)
        x = (np.clip(x, 0, 255) - 0.449 * 255) / (0.226 * 255)
        return torch.tensor(np.ascontiguousarray(x)), torch.tensor(np.ascontiguousarray(y))


@torch.no_grad()
def predict(net, X, ch, dev, bs=32):
    net.eval(); out = []
    for s in range(0, len(X), bs):
        x = torch.tensor((X[s:s + bs][:, ch].astype(np.float32) - 0.449 * 255) / (0.226 * 255)).to(dev)
        with torch.autocast('cuda', enabled=dev.type == 'cuda'):
            p = torch.sigmoid(net(x).float()) + torch.flip(torch.sigmoid(net(torch.flip(x, (3,))).float()), (3,))
        out.append(((p / 2).cpu().numpy() * 255).astype(np.uint8))
    return np.concatenate(out)


def patient_dice(P, Y, pids):
    """Dice по пациенту в 3D (по всем его срезам) для WT/TC/ET; пустой эталон и пустой ответ — 1."""
    res = {r: [] for r in REGIONS}
    for u in np.unique(pids):
        m = pids == u
        for j, r in enumerate(REGIONS):
            p, g = P[m, j] > 127, Y[m, j] > 0
            den = p.sum() + g.sum()
            res[r].append(1.0 if den == 0 else 2 * (p & g).sum() / den)
    return {r: dict(mean=float(np.mean(v)), median=float(np.median(v)), n=len(v)) for r, v in res.items()}


def train(data, ch, tag, out, dev, epochs=15, bs=16, lr=3e-4, workers=2, log=print, smoke=False):
    import segmentation_models_pytorch as smp
    torch.manual_seed(0); np.random.seed(0)
    net = smp.Unet('resnet34', encoder_weights=None if smoke else 'imagenet', in_channels=len(ch), classes=3).to(dev)
    tr = data['train']
    bs = max(1, min(bs, len(tr['X']) // 2))
    dl = DataLoader(DS(tr['X'], tr['Y'], ch, True), bs, shuffle=True, num_workers=workers, drop_last=True, pin_memory=dev.type == 'cuda')
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(dl), pct_start=.1)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    dice_l, bce = smp.losses.DiceLoss('multilabel'), nn.BCEWithLogitsLoss()
    best, best_sd = -1, None
    va = data['val']
    for ep in range(epochs):
        net.train(); t0 = time.time(); tl = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                o = net(x).float(); loss = dice_l(o, y) + bce(o, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl += loss.item()
        d = patient_dice(predict(net, va['X'], ch, dev), va['Y'], va['pid'])
        score = np.mean([d[r]['mean'] for r in REGIONS])
        if score > best:
            best, best_sd = score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        log(f'[{tag}] эпоха {ep + 1}/{epochs}: loss {tl / len(dl):.3f}  val Dice WT/TC/ET '
            f'{d["WT"]["mean"]:.3f}/{d["TC"]["mean"]:.3f}/{d["ET"]["mean"]:.3f}  {time.time() - t0:.0f} с')
    net.load_state_dict(best_sd)
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best_sd.items()}, Path(out) / f'w_brats_{tag}.pt')
    te = data['test']
    P = predict(net, te['X'], ch, dev)
    return net, patient_dice(P, te['Y'], te['pid']), P


def run(work_dir, out_dir, smoke=False, epochs=15, log=print):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = download(work_dir, log)
    data = load_slices(root, n_tumor=4 if smoke else 24, n_empty=1 if smoke else 4, log=log)
    if smoke:
        for k in data:
            sel = np.isin(data[k]['pid'], np.unique(data[k]['pid'])[:3])
            data[k] = {kk: vv[sel] for kk, vv in data[k].items()}
        epochs = 1
    R = dict(dataset='MSD Task01_BrainTumour (BraTS 2016–2017), CC BY-SA 4.0', split='по пациентам 80/10/10, seed 0',
             n_slices={k: int(len(v['X'])) for k, v in data.items()}, n_patients={k: int(len(np.unique(v['pid']))) for k, v in data.items()})
    examples = {}
    for tag, ch in [('t1gd_flair', [0, 1]), ('t1gd_only', [0])]:
        net, d, P = train(data, ch, tag, out, dev, epochs=epochs, log=log, smoke=smoke)
        R[tag] = d
        examples[tag] = P
        log(f'[{tag}] ТЕСТ Dice WT/TC/ET: {d["WT"]["mean"]:.3f}/{d["TC"]["mean"]:.3f}/{d["ET"]["mean"]:.3f}')
        del net
    # примеры для слайда и демо: 6 срезов теста с крупной опухолью
    te = data['test']
    area = te['Y'][:, 0].reshape(len(te['Y']), -1).sum(1)
    idx = np.argsort(-area)[::max(1, len(area) // 200)][:6]
    np.savez_compressed(out / 'brats_examples.npz', X=te['X'][idx], Y=te['Y'][idx], pid=te['pid'][idx],
                        P_t1gd_flair=examples['t1gd_flair'][idx], P_t1gd_only=examples['t1gd_only'][idx])
    json.dump(R, open(out / 'brats_results.json', 'w'), ensure_ascii=False, indent=1)
    log(json.dumps({k: R[k] for k in ['t1gd_flair', 't1gd_only']}, ensure_ascii=False))
    return R
