"""Загрузка данных организаторов и Figshare на том же разбиении, что на этапе 1 (data/split_v3.csv).

Порядок строк = порядок split_v3.csv = порядок, в котором считались все метрики этапа 1,
поэтому индексы VA/TE совпадают с индексами вероятностей этапа 1.
"""
import io
import os
import glob
import zipfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .common import prep, d8, small, z1

PREFIX = '/content/org/'          # так пути записаны в split_v3.csv (Colab этапа 1)
IMG_EXT = ('.jpg', '.jpeg', '.png')


def _zip_name(info):
    """Имя файла внутри zip так же, как на этапе 1: архив из Windows без UTF-8 -> чиним кириллицу (cp866)."""
    name = info.filename
    if not (info.flag_bits & 0x800):
        try:
            name = name.encode('cp437').decode('cp866')
        except Exception:
            pass
    parts = name.replace('\\', '/').split('/')
    if parts and parts[0].startswith('Almazov_MRI'):
        parts = parts[1:]
    return '/'.join(parts)


def _fix_part(p):
    """Имя папки, распакованной чужим распаковщиком с кодировкой cp437 вместо cp866, -> нормальная кириллица."""
    try:
        q = p.encode('cp437').decode('cp866')
        if any('Ѐ' <= ch <= 'ӿ' for ch in q):
            return q
    except Exception:
        pass
    return p


# полный архив организаторов (12 064 снимка, ноябрь 2026) назван по-английски; наш split_v3.csv — по-русски
NAME_MAP = {'train': 'Обучающая выборка', 'test': 'Тестова выборка', 'тестовая выборка': 'Тестова выборка',
            'glioma': 'Глиома', 'meningioma': 'Менингиома', 'pituitary': 'Гипофиз', 'notumor': 'Опухоли нет',
            'no_tumor': 'Опухоли нет', 'опухоль гипофиза': 'Гипофиз'}
CLASS_OF = {'Глиома': 0, 'Менингиома': 1, 'Гипофиз': 2, 'Опухоли нет': 3}


def key3(rel):
    """Ключ сопоставления: «выборка/класс/файл» без лишних пробелов, в одной юникод-нормализации и по-русски."""
    parts = [unicodedata.normalize('NFC', _fix_part(p)).strip() for p in rel.replace('\\', '/').split('/') if p.strip()]
    parts = parts[-3:]
    return '/'.join([NAME_MAP.get(q.lower(), q) for q in parts[:-1]] + parts[-1:])


def _iter_source(src):
    """(относительный путь, байты) для всех снимков из zip-файла или папки."""
    src = Path(src)
    if src.is_file() and src.suffix.lower() == '.zip':
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith(IMG_EXT):
                    continue
                yield _zip_name(info), z.read(info)
    else:
        for p in sorted(src.rglob('*')):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                yield str(p.relative_to(src)), p.read_bytes()


def find_file(pattern, roots):
    """Первый файл/папка по шаблону glob в списке корней (рекурсивно)."""
    for r in roots:
        if not r or not os.path.exists(r):
            continue
        hits = sorted(glob.glob(os.path.join(r, '**', pattern), recursive=True))
        if hits:
            return hits[0]
    return None


def load_org(split_csv, src, log=print, keep_extra=True):
    """Снимки организаторов -> (df в порядке split_v3.csv, A: n x 256 x 256 uint8 после prep(), ex_df, AX).

    src — архив (Almazov_MRI.zip этапа 1 или полный «Epic and CSCR hospital Dataset.zip») либо папка с распакованным архивом.
    Снимки, которых нет в split_v3.csv (в полном архиве их 2914), возвращаются отдельно (ex_df, AX) — для дополнительной проверки.
    """
    import hashlib
    sv = pd.read_csv(split_csv, keep_default_na=False)
    n = len(sv)
    keys = sv.path.map(lambda s: key3(s[len(PREFIX):] if s.startswith(PREFIX) else s))
    assert keys.is_unique, 'Ключи путей в split_v3.csv не уникальны'
    pos = dict(zip(keys, range(n)))
    A = np.zeros((n, 256, 256), np.uint8)
    hw = np.zeros((n, 2), np.int32)
    got = np.zeros(n, bool)
    ex_rows, ex_imgs = [], []
    for rel, data in _iter_source(src):
        k = key3(rel)
        i = pos.get(k)
        if i is None and not keep_extra:
            continue
        g = np.array(Image.open(io.BytesIO(data)).convert('L'))
        if i is None:
            sp, cl, nm = (k.split('/') + ['', '', ''])[:3] if k.count('/') == 2 else ('', '', k)
            ex_rows.append(dict(rel=rel, key=k, name=nm, split='test' if sp.startswith('Тест') else 'train',
                                label=CLASS_OF.get(cl, -1), h=g.shape[0], w=g.shape[1],
                                pix=hashlib.sha1(g.tobytes()).hexdigest()))
            ex_imgs.append(prep(g))
            continue
        A[i] = prep(g)
        hw[i] = g.shape
        got[i] = True
    miss = int((~got).sum())
    log(f'Снимков сопоставлено с разбиением: {int(got.sum())}/{n}; снимков вне разбиения: {len(ex_rows)}')
    if miss:
        ex = list(keys[~got][:5])
        raise RuntimeError(f'Не нашёл {miss} снимков из split_v3.csv, например: {ex}. '
                           'Нужен архив организаторов (Almazov_MRI.zip или полный архив на 12 064 снимка).')
    df = sv.copy()
    df['h'], df['w'] = hw[:, 0], hw[:, 1]
    df['test_unique'] = df.test_unique.astype(str).eq('True')
    df['pid'] = df.pid.astype(str)
    df['fig_i'] = df.fig_i.astype(int)
    # конфликт меток внутри группы копий — такие снимки исключались из обучения и оценки на этапе 1
    df['label_conflict'] = df.group.map(df.groupby('group').label.nunique() > 1).values
    ex_df = pd.DataFrame(ex_rows, columns=['rel', 'key', 'name', 'split', 'label', 'h', 'w', 'pix'])
    AX = np.stack(ex_imgs) if ex_imgs else np.zeros((0, 256, 256), np.uint8)
    return df, A, ex_df, AX


def load_figshare(npz):
    F = np.load(npz)
    return F['X'], F['M'], F['y'], F['pid'].astype(str)


def build_sets(df, A, fX, fM, fy, fpid, log=print):
    """Индексы обучения/валидации/теста и перенос масок Figshare на снимки организаторов — как на этапе 1."""
    LAB = df.label.values
    part = df.part.values
    use = ~df.label_conflict.values
    S = dict(LAB=LAB,
             TR=np.where((part == 'train') & use)[0], VA=np.where((part == 'val') & use)[0],
             TE=np.where((part == 'test') & use)[0], TEU=np.where(df.test_unique.values & use)[0])
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not (set(df.group[part == a]) & set(df.group[part == b])), (a, b)
    # Figshare: пациент попадает в ту же часть, что и его снимки в данных организаторов
    pid_part = {}
    for p_, pt in zip(df.pid, part):
        if p_:
            assert pid_part.setdefault(p_, pt) == pt
    fpart = np.array([pid_part.get(p_, 'train') for p_ in fpid])
    S['s_tr'], S['s_va'], S['s_te'] = (np.where(fpart == k)[0] for k in ('train', 'val', 'test'))
    # маски на снимках организаторов: та же ориентация (8 вариантов), что у найденного снимка Figshare
    om_idx = np.where(df.pid.values != '')[0]
    fs64 = small(fX)
    a64 = small(A[om_idx])
    OM = np.zeros((len(om_idx), 256, 256), np.uint8)
    for q, i in enumerate(om_idx):
        fi = int(df.fig_i.values[i])
        a_ = z1(a64[q])
        t = int(np.argmax([float(a_ @ z1(d8(fs64[fi], t).copy())) for t in range(8)]))
        OM[q] = d8(fM[fi], t).copy()
    S['om_idx'], S['OM'] = om_idx, OM
    S['q_tr'] = np.array([q for q, i in enumerate(om_idx) if part[i] == 'train'], int)
    S['q_va'] = np.array([q for q, i in enumerate(om_idx) if part[i] == 'val'], int)
    S['q_te'] = np.array([q for q, i in enumerate(om_idx) if part[i] == 'test'], int)
    S['fpart'] = fpart
    log(f"Разбиение: обучение {len(S['TR'])}, валидация {len(S['VA'])}, тест {len(S['TE'])} (уникальных {len(S['TEU'])}); "
        f"Figshare пациентов {len(set(fpid[S['s_tr']]))}/{len(set(fpid[S['s_va']]))}/{len(set(fpid[S['s_te']]))}, "
        f"срезов {len(S['s_tr'])}/{len(S['s_va'])}/{len(S['s_te'])}; маски на снимках организаторов: {len(om_idx)}")
    return S


def mask_for(S, idx):
    """Маски (перенесённые из Figshare) для снимков организаторов idx; has=False, если маски нет."""
    qpos = {int(i): q for q, i in enumerate(S['om_idx'])}
    has = np.array([int(i) in qpos for i in idx])
    M = np.zeros((len(idx), 256, 256), np.uint8)
    for k, i in enumerate(idx):
        if has[k]:
            M[k] = S['OM'][qpos[int(i)]]
    return M, has


FIGSHARE_FILES = [3381290, 3381293, 3381296, 3381302]   # Cheng, DOI 10.6084/m9.figshare.1512427 (CC BY 4.0)


def build_figshare_npz(dst, tmp_dir, log=print):
    """Запасной путь: скачать Figshare с первоисточника и собрать figshare256.npz тем же кодом, что на этапе 1."""
    import urllib.request
    import h5py
    import cv2
    tmp = Path(tmp_dir); (tmp / 'mat').mkdir(parents=True, exist_ok=True)
    for fid in FIGSHARE_FILES:
        z = tmp / f'{fid}.zip'
        if not z.exists():
            log(f'Скачиваю Figshare {fid}...')
            urllib.request.urlretrieve(f'https://ndownloader.figshare.com/files/{fid}', z)
        zipfile.ZipFile(z).extractall(tmp / 'mat')
    files = sorted(glob.glob(str(tmp / 'mat' / '**' / '*.mat'), recursive=True), key=lambda s: int(Path(s).stem))
    X, M, Y, P, IDX = [], [], [], [], []
    for f in files:
        with h5py.File(f, 'r') as h:
            cj = h['cjdata']
            img = np.array(cj['image']).T.astype(np.float32)
            msk = np.array(cj['tumorMask']).T.astype(np.uint8)
            lab = int(np.array(cj['label']).flatten()[0])
            pid = ''.join(chr(int(c)) for c in np.array(cj['PID']).flatten())
        lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
        img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1) * 255
        X.append(cv2.resize(img.astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA))
        M.append(cv2.resize(msk, (256, 256), interpolation=cv2.INTER_NEAREST))
        Y.append(lab); P.append(pid); IDX.append(int(Path(f).stem))
    cheng2ours = {1: 1, 2: 0, 3: 2}     # Cheng: 1 менингиома, 2 глиома, 3 гипофиз
    np.savez_compressed(dst, X=np.stack(X), M=np.stack(M), y=np.array([cheng2ours[v] for v in Y]),
                        pid=np.array(P), idx=np.array(IDX))
    log(f'Figshare собран: {len(X)} срезов')
    return dst


def find_image_root(base, min_images=9000):
    """Папка внутри base (например, /kaggle/input), где лежит распакованный архив снимков (не меньше min_images)."""
    if not os.path.isdir(base):
        return None
    for d in sorted(glob.glob(os.path.join(base, '*'))):
        if os.path.isdir(d) and sum(1 for p in Path(d).rglob('*') if p.suffix.lower() in IMG_EXT) >= min_images:
            return d
    return None
