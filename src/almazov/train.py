"""Этап 2, шаг M1: повторы обучения (новые seed) того же рецепта, что на этапе 1, и выгрузка всего,
что нужно для оценки неуверенности, калибровки и проверки — без повторного обучения потом.

Запускается из notebooks/stage2_train.ipynb (Kaggle или Colab). Каждая модель сохраняется сразу после обучения;
при повторном запуске готовые модели пропускаются, недообученная продолжается с последней эпохи.
"""
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .common import MEAN, STD, CLASSES, NO_TUMOR, clean_mask, dice
from . import nets
from .data import mask_for

cv2.setNumThreads(0)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class ClsDS(Dataset):
    """Тот же датасет и те же аугментации, что на этапе 1."""

    def __init__(s, A, LAB, idx, aug):
        s.A, s.LAB, s.idx, s.aug = A, LAB, np.asarray(idx), aug

    def __len__(s):
        return len(s.idx)

    def __getitem__(s, k):
        i = s.idx[k]
        x = s.A[i].astype(np.float32)
        if s.aug:
            if random.random() < .5:
                x = x[:, ::-1]
            M_ = cv2.getRotationMatrix2D((128, 128), random.uniform(-15, 15), random.uniform(.9, 1.1))
            M_[:, 2] += np.random.uniform(-12, 12, 2)
            x = cv2.warpAffine(np.ascontiguousarray(x), M_, (256, 256), borderMode=cv2.BORDER_CONSTANT)
            x = x * random.uniform(.8, 1.2) + random.uniform(-20, 20)
            if random.random() < .3:
                x = cv2.GaussianBlur(x, (3, 3), 0)
            if random.random() < .3:
                x = x + np.random.normal(0, 5, x.shape)
        x = (np.clip(x, 0, 255) - MEAN) / STD
        return torch.tensor(x, dtype=torch.float32)[None].repeat(3, 1, 1), int(s.LAB[i])


class SegDS(Dataset):
    """Срезы Figshare с масками + снимки организаторов с перенесёнными масками + здоровые с пустой маской."""

    def __init__(s, X, M, aug):
        s.X, s.M, s.aug = X, M, aug

    def __len__(s):
        return len(s.X)

    def __getitem__(s, k):
        x = s.X[k].astype(np.float32)
        m = s.M[k].astype(np.float32)
        if s.aug:
            if random.random() < .5:
                x, m = x[:, ::-1], m[:, ::-1]
            if random.random() < .5:
                x, m = x[::-1], m[::-1]
            M_ = cv2.getRotationMatrix2D((128, 128), random.uniform(-20, 20), random.uniform(.85, 1.15))
            x = cv2.warpAffine(np.ascontiguousarray(x), M_, (256, 256))
            m = cv2.warpAffine(np.ascontiguousarray(m), M_, (256, 256), flags=cv2.INTER_NEAREST)
            x = x * random.uniform(.8, 1.2) + random.uniform(-20, 20)
        x = (np.clip(x, 0, 255) - MEAN) / STD
        return torch.tensor(x, dtype=torch.float32)[None], torch.tensor(np.ascontiguousarray(m), dtype=torch.float32)[None]


def _loader(ds, bs, shuffle, workers, dev):
    return DataLoader(ds, bs, shuffle=shuffle, num_workers=workers, drop_last=shuffle, pin_memory=dev.type == 'cuda',
                      persistent_workers=False)


def _scaler(dev):
    return torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')


def train_cls(A, LAB, tr, va, dev, ck_dir, tag, seed, epochs=15, lr=3e-4, bs=64, arch='resnet50', pretrained=True,
              workers=2, log=print):
    """Рецепт классификатора этапа 1 (AdamW, OneCycle, label smoothing 0.05, лучшая эпоха по macro-F1 валидации)."""
    from sklearn.metrics import f1_score
    set_seed(seed)
    m = nets.make_cls(arch, pretrained).to(dev)
    dl = _loader(ClsDS(A, LAB, tr, True), bs, True, workers, dev)
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(dl), pct_start=.15)
    scaler = _scaler(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=.05)
    CK, BEST = Path(ck_dir) / f'{tag}_last.pt', Path(ck_dir) / f'{tag}_best.pt'
    start, best, best_ep, hist = 0, -1.0, -1, []
    if CK.exists():
        c = torch.load(CK, map_location=dev, weights_only=False)
        m.load_state_dict(c['m']); opt.load_state_dict(c['o']); sch.load_state_dict(c['s']); scaler.load_state_dict(c['sc'])
        start, best, best_ep, hist = c['ep'] + 1, c['best'], c['best_ep'], c['hist']
        log(f'[{tag}] продолжаю с эпохи {start + 1}')
    for ep in range(start, epochs):
        m.train(); t0 = time.time(); tl = 0.0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with nets.amp(dev):
                loss = crit(m(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl += loss.item()
        pv = nets.softmax_tta(nets.cls_logits(m, A[va], dev))
        f1 = float(f1_score(LAB[va], pv.argmax(1), average='macro'))
        if f1 > best:
            best, best_ep = f1, ep
            torch.save(m.state_dict(), BEST)
        hist.append(dict(ep=ep + 1, loss=tl / max(len(dl), 1), val_f1=f1, sec=round(time.time() - t0, 1)))
        torch.save(dict(m=m.state_dict(), o=opt.state_dict(), s=sch.state_dict(), sc=scaler.state_dict(), ep=ep,
                        best=best, best_ep=best_ep, hist=hist), CK)
        log(f'[{tag}] эпоха {ep + 1}/{epochs}: loss {hist[-1]["loss"]:.3f}  val macro-F1 {f1:.4f}  лучший {best:.4f}  {hist[-1]["sec"]:.0f} с')
    m.load_state_dict(torch.load(BEST, map_location=dev, weights_only=True))
    return m, dict(val_macro_f1=best, best_epoch=best_ep + 1, epochs=epochs, seed=seed, arch=arch, history=hist), CK


def train_seg(Xtr, Mtr, Xva, Mva, dev, ck_dir, tag, seed, epochs=20, lr=3e-4, bs=16, pretrained=True, workers=2, log=print):
    """Рецепт сегментации этапа 1 (U-Net ResNet34, Dice+BCE, лучшая эпоха по Dice валидации Figshare)."""
    import segmentation_models_pytorch as smp
    set_seed(seed)
    net = nets.make_seg(pretrained).to(dev)
    dl = _loader(SegDS(Xtr, Mtr, True), bs, True, workers, dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(dl), pct_start=.1)
    scaler = _scaler(dev)
    dl_, bce = smp.losses.DiceLoss('binary'), nn.BCEWithLogitsLoss()
    CK, BEST = Path(ck_dir) / f'{tag}_last.pt', Path(ck_dir) / f'{tag}_best.pt'
    start, best, best_ep, hist = 0, -1.0, -1, []
    if CK.exists():
        c = torch.load(CK, map_location=dev, weights_only=False)
        net.load_state_dict(c['m']); opt.load_state_dict(c['o']); sch.load_state_dict(c['s']); scaler.load_state_dict(c['sc'])
        start, best, best_ep, hist = c['ep'] + 1, c['best'], c['best_ep'], c['hist']
        log(f'[{tag}] продолжаю с эпохи {start + 1}')
    for ep in range(start, epochs):
        net.train(); t0 = time.time(); tl = 0.0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with nets.amp(dev):
                o = net(x).float()
                loss = dl_(o, y) + bce(o, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl += loss.item()
        d = float(dice(nets.seg_maps(net, Xva, dev) > .5, Mva > 0).mean())
        if d > best:
            best, best_ep = d, ep
            torch.save(net.state_dict(), BEST)
        hist.append(dict(ep=ep + 1, loss=tl / max(len(dl), 1), val_dice=d, sec=round(time.time() - t0, 1)))
        torch.save(dict(m=net.state_dict(), o=opt.state_dict(), s=sch.state_dict(), sc=scaler.state_dict(), ep=ep,
                        best=best, best_ep=best_ep, hist=hist), CK)
        log(f'[{tag}] эпоха {ep + 1}/{epochs}: loss {hist[-1]["loss"]:.3f}  val Dice {d:.4f}  лучший {best:.4f}  {hist[-1]["sec"]:.0f} с')
    net.load_state_dict(torch.load(BEST, map_location=dev, weights_only=True))
    return net, dict(val_dice=best, best_epoch=best_ep + 1, epochs=epochs, seed=seed, history=hist), CK


# ---------------------------------------------------------------- быстрая оценка (для сверки и сводки)

def cls_metrics(P, Y):
    from sklearn.metrics import f1_score, roc_auc_score
    pr = P.argmax(1); tt = Y != NO_TUMOR
    return dict(n=int(len(Y)), accuracy=float((pr == Y).mean()), macro_f1=float(f1_score(Y, pr, average='macro')),
                glioma_recall=float((pr[Y == 0] == 0).mean()) if (Y == 0).any() else None,
                tumor_sens=float(((pr != NO_TUMOR) & tt).sum() / max(tt.sum(), 1)),
                tumor_spec=float(((pr == NO_TUMOR) & ~tt).sum() / max((~tt).sum(), 1)),
                tumor_auc=float(roc_auc_score(tt, 1 - P[:, NO_TUMOR])) if 0 < tt.sum() < len(Y) else None)


def seg_metrics(maps_fte, M_fte, y_fte, maps_ote, M_ote, has_ote, y_ote):
    """maps_* — uint8 (0..255) или float; метрики теми же правилами, что на этапе 1."""
    f = lambda a: a.astype(np.float32) / 255 if a.dtype == np.uint8 else a
    pf = f(maps_fte)
    d = dice(pf > .5, M_fte > 0)
    r = dict(dice_figshare_new_patients=float(d.mean()), dice_median=float(np.median(d)),
             share_dice_lt_05=float((d < .5).mean()),
             dice_by_class={CLASSES[k]: float(d[y_fte == k].mean()) for k in range(3) if (y_fte == k).any()})
    po = f(maps_ote)
    if has_ote.any():
        r['dice_org_images'] = float(dice(po[has_ote] > .5, M_ote[has_ote] > 0).mean())
    nt = y_ote == NO_TUMOR
    if nt.any():
        r['healthy_false_contour_demo'] = float(np.mean([clean_mask(p).sum() > 0 for p in po[nt]]))
    return r


# ---------------------------------------------------------------- оркестратор этапа 2

def _git_commit(repo):
    try:
        return subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ''


def run_stage2(df, A, fX, fM, fy, fpid, S, out_dir, repo_dir, smoke=False, seg_seeds=(1, 2, 3), cls_seeds=(1, 2),
               seg_epochs=20, cls_epochs=15, pretrained=True, workers=None, cls_bs=64, seg_bs=16, save_images=True, ex_df=None, AX=None, log=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ck = out / 'ckpt'; ck.mkdir(exist_ok=True)
    logf = open(out / 'log.txt', 'a', encoding='utf-8')

    def L(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True); logf.write(time.strftime('%H:%M:%S ') + s + '\n'); logf.flush()
    log = log or L
    dev = nets.get_device()
    workers = workers if workers is not None else min(4, os.cpu_count() or 2)
    T0 = time.time()
    LAB = S['LAB']
    lim = (lambda a, k: a[:k]) if smoke else (lambda a, k: a)
    if smoke:
        seg_seeds, cls_seeds, seg_epochs, cls_epochs = seg_seeds[:1], cls_seeds[:1], 1, 1
        cls_bs, seg_bs = min(cls_bs, 16), min(seg_bs, 8)
    VA, TE = lim(S['VA'], 48), lim(S['TE'], 64)
    s_va, s_te = lim(S['s_va'], 32), lim(S['s_te'], 48)
    log(f'Устройство: {dev} {torch.cuda.get_device_name(0) if dev.type == "cuda" else ""}; режим: {"SMOKE (проверка)" if smoke else "ПОЛНЫЙ"}; папка: {out}')

    # ---------- наборы для оценки и их «эталон»
    M_ova, has_ova = mask_for(S, VA)
    M_ote, has_ote = mask_for(S, TE)
    sets = dict(fva=fX[s_va], fte=fX[s_te], ova=A[VA], ote=A[TE])
    ev = out / 'eval_data.npz'
    if not ev.exists():
        np.savez_compressed(ev, m_fva=fM[s_va], m_fte=fM[s_te], m_ova=M_ova, m_ote=M_ote, has_ova=has_ova, has_ote=has_ote,
                            y_fva=fy[s_va], y_fte=fy[s_te], pid_fva=fpid[s_va], pid_fte=fpid[s_te],
                            idx_fva=s_va, idx_fte=s_te, idx_ova=VA, idx_ote=TE,
                            y_ova=LAB[VA], y_ote=LAB[TE], pid_ova=df.pid.values[VA], pid_ote=df.pid.values[TE],
                            group_ova=df.group.values[VA], group_ote=df.group.values[TE])
        cols = [c for c in df.columns if c != 'path'] + ['path']
        df[cols].to_csv(out / 'meta.csv', index=False, encoding='utf-8')
        np.savez(out / 'index.npz', **{k: S[k] for k in ['TR', 'VA', 'TE', 'TEU', 's_tr', 's_va', 's_te', 'om_idx', 'fpart']})
        log('Сохранены данные для оценки:', ev.name, f'{ev.stat().st_size / 1e6:.0f} МБ')

    # пул снимков для ручной разметки режима МРТ и плоскости: все здоровые теста + стратифицированная выборка
    lp = out / 'label_pool.npz'
    if not lp.exists():
        rs = np.random.RandomState(0)
        pool = list(S['TE'][LAB[S['TE']] == NO_TUMOR])
        for part_idx, k in [(S['VA'], 40), (S['TR'], 60)]:
            for c in range(4):
                cand = part_idx[LAB[part_idx] == c]
                pool += list(rs.choice(cand, min(k, len(cand)), replace=False))
        pool = np.array(lim(np.array(pool), 40))
        np.savez_compressed(lp, idx=pool, y=LAB[pool], part=df.part.values[pool], source=df.source.values[pool],
                            h=df.h.values[pool], w=df.w.values[pool])
        log(f'Пул для разметки режима/плоскости: {len(pool)} снимков')

    # картинки 256x256 (JPEG, качество 95) — для галереи, разметки режима/плоскости и демо; оценка идёт по картам, не по ним
    ij = out / 'images_jpg.zip'
    if save_images and not ij.exists():
        import zipfile
        pool_idx = np.load(lp)['idx']
        with zipfile.ZipFile(ij, 'w', zipfile.ZIP_STORED) as z:
            groups = dict(fva=(sets['fva'], s_va), fte=(sets['fte'], s_te), ova=(sets['ova'], VA), ote=(sets['ote'], TE),
                          pool=(A[pool_idx], pool_idx))
            if ex_df is not None and len(ex_df):
                groups['extra'] = (AX if not smoke else AX[:20], np.arange(len(AX) if not smoke else min(20, len(AX))))
            for g_, (X_, ids_) in groups.items():
                for x_, i_ in zip(X_, ids_):
                    z.writestr(f'{g_}/{int(i_):05d}.jpg', cv2.imencode('.jpg', x_, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
        log(f'Картинки для галереи и разметки: {ij.name} {ij.stat().st_size / 1e6:.0f} МБ')

    summary_path = out / 'summary.json'
    SM = json.load(open(summary_path)) if summary_path.exists() else {}

    def save_summary():
        SM['env'] = dict(device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else '',
                         torch=torch.__version__, python=platform.python_version(), commit=_git_commit(repo_dir),
                         smoke=smoke, minutes_total=round((time.time() - T0) / 60, 1))
        json.dump(SM, open(summary_path, 'w'), ensure_ascii=False, indent=1)

    W = Path(repo_dir) / 'weights'

    # ---------- 0. модели этапа 1: сверка тем же кодом + всё, что нужно дальше
    if not (out / 'cls_stage1.npz').exists():
        man = json.load(open(W / 'cls_models.json'))['models'][0]
        f = nets.assemble([W / p for p in man['parts']], ck / 'cls_stage1.pt', sha256=man['sha256'])
        m = nets.make_cls(man['arch']); m.load_state_dict(nets.load_state(f)); m.to(dev).eval()
        t0 = time.time()
        lva, lte = nets.cls_logits(m, A[VA], dev), nets.cls_logits(m, A[TE], dev)
        cam, cpred = nets.cls_gradcam(m, A[TE], dev)
        feats = nets.cls_features(m, lim(A, 64), dev)
        np.save(out / 'feat_stage1.npy', feats)
        np.savez_compressed(out / 'cam_stage1.npz', cam_te=cam, pred_te=cpred)
        np.savez(out / 'cls_stage1.npz', lva=lva, lte=lte, meta=json.dumps(dict(arch=man['arch'], source='этап 1 (weights/cls_models.json)')))
        r = cls_metrics(nets.softmax_tta(lte), LAB[TE])
        SM['cls_stage1'] = dict(test=r, sec=round(time.time() - t0))
        ok = abs(r['accuracy'] - 0.9237) < 0.004
        log(f'Сверка классификатора этапа 1: точность на тесте {r["accuracy"]:.4f} (на этапе 1 было 0.9237)',
            'OK' if ok or smoke else '<- НЕ СОВПАДАЕТ, пришлите лог')
        SM['cls_stage1']['check_ok'] = bool(ok)
        save_summary(); del m
        if dev.type == 'cuda': torch.cuda.empty_cache()

    if not (out / 'seg_stage1.npz').exists():
        f = nets.assemble([W / f'seg_v3_best.pt.part{i}' for i in range(5)], ck / 'seg_stage1.pt', sha_prefix='ac9d92e985ae')
        net = nets.make_seg(); net.load_state_dict(nets.load_state(f)); net.to(dev).eval()
        t0 = time.time()
        res = {}
        for k, X in sets.items():
            b = nets.seg_maps(net, X, dev, flip='both')
            res[k] = nets.to_u8(b.mean(1)); res[k + '_noflip'] = nets.to_u8(b[:, 0])
        np.savez_compressed(out / 'seg_stage1.npz', **res)
        r = seg_metrics(res['fte'], fM[s_te], fy[s_te], res['ote'], M_ote, has_ote, LAB[TE])
        SM['seg_stage1'] = dict(test=r, sec=round(time.time() - t0))
        ok = abs(r['dice_figshare_new_patients'] - 0.7900) < 0.006
        log(f'Сверка сегментации этапа 1: Dice {r["dice_figshare_new_patients"]:.4f} (было 0.7900), '
            f'на снимках организаторов {r.get("dice_org_images", float("nan")):.4f} (было 0.8188)', 'OK' if ok or smoke else '<- НЕ СОВПАДАЕТ, пришлите лог')
        SM['seg_stage1']['check_ok'] = bool(ok)
        save_summary(); del net
        if dev.type == 'cuda': torch.cuda.empty_cache()

    # ---------- 1. сегментация: новые seed того же рецепта (для ансамбля и карты неуверенности границы)
    items_x, items_m = None, None
    for sd in seg_seeds:
        tag = f'seg_s{sd}'
        if (out / f'{tag}.npz').exists():
            log(f'[{tag}] уже готово — пропускаю'); continue
        if items_x is None:
            tr_n = S['TR'][LAB[S['TR']] == NO_TUMOR]
            qtr = S['q_tr']
            items_x = np.concatenate([fX[S['s_tr']], A[S['om_idx'][qtr]], A[tr_n]])
            items_m = np.concatenate([fM[S['s_tr']], S['OM'][qtr], np.zeros((len(tr_n), 256, 256), np.uint8)])
            if smoke:
                sel = np.random.RandomState(0).choice(len(items_x), 64, replace=False)
                items_x, items_m = items_x[sel], items_m[sel]
            log(f'Обучающих примеров для сегментации: {len(items_x)}')
        t0 = time.time()
        net, info, CK = train_seg(items_x, items_m, fX[s_va], fM[s_va], dev, ck, tag, sd, seg_epochs, bs=seg_bs,
                                  pretrained=pretrained, workers=workers, log=log)
        res = {k: nets.to_u8(nets.seg_maps(net, X, dev)) for k, X in sets.items()}
        torch.save(nets.half_state(net.state_dict()), out / f'w_{tag}.pt')
        np.savez_compressed(out / f'{tag}.npz', **res, meta=json.dumps({k: v for k, v in info.items() if k != 'history'}))
        info['test'] = seg_metrics(res['fte'], fM[s_te], fy[s_te], res['ote'], M_ote, has_ote, LAB[TE])
        info['minutes'] = round((time.time() - t0) / 60, 1)
        SM[tag] = info; save_summary()
        CK.unlink(missing_ok=True); (ck / f'{tag}_best.pt').unlink(missing_ok=True)
        log(f'[{tag}] готово за {info["minutes"]} мин: Dice на тесте {info["test"]["dice_figshare_new_patients"]:.4f}')
        del net
        if dev.type == 'cuda': torch.cuda.empty_cache()
    del items_x, items_m

    # ---------- 2. классификатор: новые seed выбранной архитектуры (ResNet-50)
    for sd in cls_seeds:
        tag = f'cls_s{sd}'
        if (out / f'{tag}.npz').exists():
            log(f'[{tag}] уже готово — пропускаю'); continue
        t0 = time.time()
        tr = S['TR'] if not smoke else np.random.RandomState(0).choice(S['TR'], 64, replace=False)
        m, info, CK = train_cls(A, LAB, tr, VA, dev, ck, tag, sd, cls_epochs, bs=cls_bs, pretrained=pretrained, workers=workers, log=log)
        lva, lte = nets.cls_logits(m, A[VA], dev), nets.cls_logits(m, A[TE], dev)
        cam, cpred = nets.cls_gradcam(m, A[TE], dev)
        torch.save(nets.half_state(m.state_dict()), out / f'w_{tag}.pt')
        np.savez(out / f'{tag}.npz', lva=lva, lte=lte, meta=json.dumps({k: v for k, v in info.items() if k != 'history'}))
        np.savez_compressed(out / f'cam_{tag}.npz', cam_te=cam, pred_te=cpred)
        info['test'] = cls_metrics(nets.softmax_tta(lte), LAB[TE])
        info['minutes'] = round((time.time() - t0) / 60, 1)
        SM[tag] = info; save_summary()
        CK.unlink(missing_ok=True); (ck / f'{tag}_best.pt').unlink(missing_ok=True)
        log(f'[{tag}] готово за {info["minutes"]} мин: точность на тесте {info["test"]["accuracy"]:.4f}')
        del m
        if dev.type == 'cuda': torch.cuda.empty_cache()

    # ---------- 2b. снимки вне разбиения этапа 1 (полный архив организаторов): связи и предсказания всех моделей
    if ex_df is not None and len(ex_df) and not (out / 'extra.npz').exists():
        from .extra import link_extra
        from .common import d8, small, z1
        t0 = time.time()
        sel = np.r_[0:min(20, len(ex_df)), max(20, len(ex_df) - 20):len(ex_df)] if smoke else np.arange(len(ex_df))
        exl, AXl = ex_df.iloc[sel].reset_index(drop=True), AX[sel]
        e = link_extra(df, A, fX, fy, fpid, S, exl, AXl, dev, log=log)
        # маска врача, если снимок — срез пациента Figshare (ориентация подбирается так же, как на этапе 1)
        mx = np.zeros((len(e), 256, 256), np.uint8)
        fs64, ax64 = small(fX[e.fig_i.values[e.fig_i.values >= 0]]), small(AXl[e.fig_i.values >= 0])
        for q, k in enumerate(np.where(e.fig_i.values >= 0)[0]):
            a_ = z1(ax64[q])
            t = int(np.argmax([float(a_ @ z1(d8(fs64[q], t).copy())) for t in range(8)]))
            mx[k] = d8(fM[e.fig_i.values[k]], t).copy()
        res = dict(m=mx, has_m=e.fig_i.values >= 0)
        man = json.load(open(W / 'cls_models.json'))['models'][0]
        cls_w = {'cls_stage1': nets.assemble([W / p for p in man['parts']], ck / 'cls_stage1.pt', sha256=man['sha256'])}
        cls_w.update({f'cls_s{sd}': out / f'w_cls_s{sd}.pt' for sd in cls_seeds if (out / f'w_cls_s{sd}.pt').exists()})
        seg_w = {'seg_stage1': nets.assemble([W / f'seg_v3_best.pt.part{i}' for i in range(5)], ck / 'seg_stage1.pt', sha_prefix='ac9d92e985ae')}
        seg_w.update({f'seg_s{sd}': out / f'w_seg_s{sd}.pt' for sd in seg_seeds if (out / f'w_seg_s{sd}.pt').exists()})
        for k, f in cls_w.items():
            m = nets.make_cls('resnet50'); m.load_state_dict(nets.load_state(f)); m.to(dev).eval()
            res['logits_' + k] = nets.cls_logits(m, AXl, dev)
            del m
        for k, f in seg_w.items():
            net = nets.make_seg(); net.load_state_dict(nets.load_state(f)); net.to(dev).eval()
            res['seg_' + k] = nets.to_u8(nets.seg_maps(net, AXl, dev))
            del net
        if dev.type == 'cuda': torch.cuda.empty_cache()
        np.savez_compressed(out / 'extra.npz', **res)
        e.to_csv(out / 'extra_meta.csv', index=False, encoding='utf-8')
        nw = (e.status == 'новый').values & (e.label.values >= 0)
        ye = e.label.values
        SM['extra'] = dict(n=int(len(e)), status=e.status.value_counts().to_dict(), n_new=int(nw.sum()),
                           n_new_groups=int(e.group[nw].nunique()), minutes=round((time.time() - t0) / 60, 1))
        if nw.any():
            P1 = nets.softmax_tta(res['logits_cls_stage1'][nw])
            SM['extra']['cls_stage1_new'] = cls_metrics(P1, ye[nw])
            ks = [k for k in res if k.startswith('logits_')]
            if len(ks) > 1:
                SM['extra']['cls_ensemble_new'] = cls_metrics(np.mean([nets.softmax_tta(res[k][nw]) for k in ks], 0), ye[nw])
            log(f'Новые снимки ({int(nw.sum())}): точность классификатора этапа 1 {SM["extra"]["cls_stage1_new"]["accuracy"]:.4f}')
        save_summary()

    # ---------- 3. сводка ансамблей (предварительная; полная оценка — потом, по протоколу)
    segs = [k for k in ['seg_stage1'] + [f'seg_s{s}' for s in seg_seeds] if (out / f'{k}.npz').exists()]
    if len(segs) >= 2:
        Z = [np.load(out / f'{k}.npz') for k in segs]
        mf = np.mean([z['fte'].astype(np.float32) / 255 for z in Z], 0)
        mo = np.mean([z['ote'].astype(np.float32) / 255 for z in Z], 0)
        SM['seg_ensemble'] = dict(members=segs, test=seg_metrics(mf, fM[s_te], fy[s_te], mo, M_ote, has_ote, LAB[TE]))
        log(f'Ансамбль сегментации ({len(segs)} моделей): Dice {SM["seg_ensemble"]["test"]["dice_figshare_new_patients"]:.4f}')
    clss = [k for k in ['cls_stage1'] + [f'cls_s{s}' for s in cls_seeds] if (out / f'{k}.npz').exists()]
    if len(clss) >= 2:
        P = np.mean([nets.softmax_tta(np.load(out / f'{k}.npz')['lte']) for k in clss], 0)
        SM['cls_ensemble'] = dict(members=clss, test=cls_metrics(P, LAB[TE]))
        log(f'Ансамбль классификаторов ({len(clss)} моделей): точность {SM["cls_ensemble"]["test"]["accuracy"]:.4f}')
    save_summary()
    for p in ck.glob('*'):
        p.unlink()
    log(f'ВСЁ ГОТОВО за {(time.time() - T0) / 60:.0f} мин. Файлы в {out}')
    logf.close()
    return SM
