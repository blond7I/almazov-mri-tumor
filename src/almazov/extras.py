"""Дополнительные проверки этапа 2 — запускаются на Kaggle после обучения (notebooks/stage2_extras.ipynb).

Всё считается тем же прототипом, которым пользуется врач (pipeline.analyze на исходных файлах), а не картами из оценки:
1) normals_by_sequence — здоровые снимки теста по режимам МРТ: отказ «вне области применения», ложная тревога, ложный контур;
2) new_images — 92 по-настоящему новых снимка полного архива: результат по классам и 20 ошибок для просмотра врачом;
3) doctor_cases — набор случаев для сессии с врачом (10 случаев + 3 «обвести с нуля» + DICOM) с ключом ответов;
4) brats_band — полоса неуверенности v2 на чужих данных (T1 с контрастом из BraTS/MSD), расчёт зарегистрирован в протоколе.
"""
import glob
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .common import CLASSES, NO_TUMOR
from . import pipeline
from .data import _iter_source, key3, PREFIX

TUMOR_NAMES = CLASSES[:3]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 3), round(float(c + h), 3)]


def compact(res):
    f = res['findings']
    pred = res['top_type'] if res['tumor'] else CLASSES[NO_TUMOR]
    return dict(refused=bool(res['refused']), tumor=bool(res['tumor']), pred=pred, p_tumor=res['p_tumor'],
                top=res['top_type'], top_p=res['top_type_p'], n_find=len(f), n_zones=len(res['review_zones']),
                band_ratio=round(max([x['band_ratio'] for x in f], default=0.0), 3), flags='|'.join(res['flags']),
                checks='|'.join(c['code'] for c in res['checks']), seq=res['auto_sequence'], seq_p=res['auto_sequence_p'],
                plane=res['auto_plane'])


def dice1(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def read_originals(src, keys):
    """Исходные снимки организаторов (серые, исходный размер) по ключам key3."""
    want, out = set(keys), {}
    for rel, data in _iter_source(src):
        k = key3(rel)
        if k in want:
            out[k] = np.array(Image.open(io.BytesIO(data)).convert('L'))
    return out


def df_key(path):
    return key3(path[len(PREFIX):] if path.startswith(PREFIX) else path)


def save_png(path, g):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), g)


# ---------------------------------------------------------------- 1. здоровые снимки теста по режимам
def normals_by_sequence(M, labels_csv, orig_by_idx, out_rows=None, log=print):
    lab = pd.read_csv(labels_csv)
    t = lab[(lab.part == 'test') & (lab.label == NO_TUMOR)].copy()
    rows = []
    for _, r in t.iterrows():
        g = orig_by_idx.get(int(r.idx))
        if g is None:
            continue
        c = compact(pipeline.analyze(M, g))
        rows.append(dict(idx=int(r.idx), seq_label=r.seq, **c))
    R = pd.DataFrame(rows)
    if out_rows:
        R.to_csv(out_rows, index=False, encoding='utf-8')
    summ = {}
    for s, d in list(R.groupby('seq_label')) + [('все', R)]:
        n = len(d)
        ans = d[~d.refused]
        fp = int(ans.tumor.sum())
        fc = int((ans.n_find > 0).sum())
        summ[s] = dict(n=n, refused=int(d.refused.sum()), answered=len(ans),
                       said_tumor=fp, said_tumor_ci=wilson(fp, len(ans)),
                       false_contour=fc, false_contour_ci=wilson(fc, len(ans)),
                       review_zones_per_image=round(float(ans.n_zones.mean()), 3) if len(ans) else None,
                       correct_or_refused=int(n - fp), correct_or_refused_share=round((n - fp) / n, 3) if n else None)
    log('Здоровые снимки теста по режимам:', json.dumps(summ, ensure_ascii=False))
    return dict(note='уникальные здоровые снимки теста (по одному на группу копий); режим — разметка ассистента/Figshare, не врача; '
                     'ответ прототипа целиком (фильтр входа + классификатор + контур)', by_sequence=summ)


# ---------------------------------------------------------------- 2. новые снимки полного архива
def new_images(M, extra_meta_csv, orig_by_key, out_dir, n_err=20, log=print):
    e = pd.read_csv(extra_meta_csv, keep_default_na=False)
    e = e[(e.status == 'новый') & (e.label.astype(int) >= 0)].reset_index(drop=True)
    rows, keep = [], {}
    for _, r in e.iterrows():
        g = orig_by_key.get(r.key)
        if g is None:
            continue
        res = pipeline.analyze(M, g)
        c = compact(res)
        y = CLASSES[int(r.label)]
        rows.append(dict(key=r.key, group=r.group, label=y, ok=c['pred'] == y, **c))
        keep[r.key] = (g, res)
    R = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tum = R[R.label != CLASSES[NO_TUMOR]]
    nor = R[R.label == CLASSES[NO_TUMOR]]
    summ = dict(n=len(R), accuracy=round(float(R.ok.mean()), 3), accuracy_ci=wilson(int(R.ok.sum()), len(R)),
                tumor_suspected=int(tum.tumor.sum()), tumors=len(tum),
                type_correct={y: f'{int(d.ok.sum())}/{len(d)}' for y, d in tum.groupby('label')},
                contour_found=int((tum.n_find > 0).sum()),
                normal_false_alarm=int(nor.tumor.sum()), normals=len(nor), refused=int(R.refused.sum()),
                asks_review=int(((R.top_p < M.calib.get('low_conf', 0.6)) | R['flags'].str.contains('uncertain')).sum()))
    # ошибки для просмотра врачом: сначала уверенные ошибки, по одной на группу копий
    err = R[~R.ok].sort_values('top_p', ascending=False).drop_duplicates('group').head(n_err).reset_index(drop=True)
    for i, r in err.iterrows():
        g, res = keep[r.key]
        save_png(out / f'err_{i + 1:02d}.png', g)
        save_png(out / f'err_{i + 1:02d}_system.png', cv2.cvtColor(pipeline.overlay(res), cv2.COLOR_RGB2BGR))
    err.insert(0, 'file', [f'err_{i + 1:02d}.png' for i in range(len(err))])
    err.to_csv(out / 'errors.csv', index=False, encoding='utf-8')
    R.to_csv(out / 'all_new_images.csv', index=False, encoding='utf-8')
    log('Новые снимки:', json.dumps(summ, ensure_ascii=False))
    return summ


# ---------------------------------------------------------------- 3. набор для врача
def read_mat(f):
    import h5py
    with h5py.File(f, 'r') as h:
        cj = h['cjdata']
        img = np.array(cj['image']).T.astype(np.float32)
        msk = np.array(cj['tumorMask']).T.astype(np.uint8)
        lab = {1: 1, 2: 0, 3: 2}[int(np.array(cj['label']).flatten()[0])]
        pid = ''.join(chr(int(c)) for c in np.array(cj['PID']).flatten())
    return img, msk, lab, pid


def to_u8(img):
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    return (np.clip((img - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def write_dicom(path, raw, spacing_mm=0.49, desc='T1 +C (demo)'):
    """Демонстрационный DICOM из среза Figshare: пиксели исходные, геометрия УСЛОВНАЯ (в наборе её нет)."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid, MRImageStorage
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.Modality = 'MR'
    ds.SeriesDescription = desc
    ds.ContrastBolusAgent = 'Gd (demo)'
    ds.PatientName, ds.PatientID = 'ANON', 'DEMO'
    ds.ImageComments = 'Demo: Figshare (Cheng et al., CC BY 4.0) slice; PixelSpacing and orientation are ASSUMED, not from the source'
    ds.PixelSpacing = [spacing_mm, spacing_mm]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.Rows, ds.Columns = raw.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = np.clip(raw, 0, 65535).astype(np.uint16).tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.save_as(str(path), write_like_original=False)


EXPECT = {
    'уверена и права': 'система уверена и права — замеряем скорость принятия',
    'широкая полоса': 'граница неуверенная (широкая жёлтая полоса) — полезна ли полоса, как врач правит границу',
    'тип неуверен': 'тип опухоли определён неуверенно — как врач реагирует на «сомнения»',
    'контура нет': 'классификатор видит опухоль, контура нет — срабатывает ли подсказка «смотрите весь срез»',
    'зона просмотра': 'есть зона дополнительного просмотра — полезны ли зоны',
    'здоровый срез': 'здоровый срез T1 — нет ли ложных находок',
    'ошибка системы': 'система ошибается (тип или контур) — ловит ли врач ошибку, сколько времени на исправление',
    'с нуля': 'врач обводит сам, без системы — время для сравнения',
}


def doctor_cases(M, mats, s_te, fidx, fM, org_normals, org_tumors, out_dir, seed=0, log=print):
    """mats: номер .mat -> путь; s_te/fidx/fM — тест Figshare (индексы, номера файлов, маски 256);
    org_normals / org_tumors: списки (ключ, серый снимок, метка) — снимки организаторов из теста (уникальные)."""
    rs = np.random.RandomState(seed)
    out = Path(out_dir)
    rows, store = [], {}
    t0 = time.time()
    for i in s_te:
        raw, msk, lab, pid = read_mat(mats[int(fidx[i])])
        g = to_u8(raw)
        res = pipeline.analyze(M, g)
        c = compact(res)
        d = dice1(res['arrays']['mask'], fM[i]) if not res['refused'] else 0.0
        rows.append(dict(src='figshare', i=int(i), file=int(fidx[i]), pid=pid, y=lab, label=CLASSES[lab], dice=round(d, 3),
                         area_gt=int(fM[i].sum()), ok=c['pred'] == CLASSES[lab], **c))
        store[('figshare', int(i))] = (g, raw, msk, res)
    for k, g, lab in org_tumors + org_normals:
        res = pipeline.analyze(M, g)
        c = compact(res)
        rows.append(dict(src='org', i=k, file=k, pid='org:' + k, y=lab, label=CLASSES[lab], dice=np.nan, area_gt=0,
                         ok=c['pred'] == CLASSES[lab], **c))
        store[('org', k)] = (g, None, None, res)
    R = pd.DataFrame(rows)
    log(f'Прогон прототипа на кандидатах: {len(R)} снимков за {time.time() - t0:.0f} с')
    used = set()

    def pick(q, n=1, by=None, asc=True, per_class=False):
        d = R[q & ~R.pid.isin(used)]
        if by is not None:
            d = d.sort_values(by, ascending=asc)
        got = []
        for _, r in d.iterrows():
            if r.pid in used or (per_class and r.y in [x.y for x in got]):
                continue
            got.append(r); used.add(r.pid)
            if len(got) == n:
                break
        return got

    fig, org = R.src == 'figshare', R.src == 'org'
    fl = R['flags'].fillna('')
    clean = ~fl.str.contains('boundary_uncertain|type_uncertain|review_zones|presence_uncertain')
    R['med_dev'] = (R.area_gt - R.groupby('y').area_gt.transform('median')).abs()
    sel = []
    sel += [('уверена и права', r) for r in pick(fig & R.ok & (R.top_p >= .9) & (R.p_tumor >= .95) & (R.dice >= .85) & clean,
                                                   3, 'med_dev', per_class=True)]
    sel += [('широкая полоса', r) for r in pick(fig & R.ok & (fl.str.contains('boundary_uncertain') | (R.band_ratio >= .5))
                                                 & R.dice.between(.6, .9), 2, 'band_ratio', asc=False)]
    sel += [('тип неуверен', r) for r in pick(fig & R.tumor & fl.str.contains('type_uncertain') & (R.p_tumor >= .8), 1, 'top_p')]
    c7 = pick((R.y < 3) & fl.str.contains('classifier_tumor_no_contour') & ~R.refused, 1, 'p_tumor', asc=False)
    sel += [('контура нет', r) for r in c7]
    sel += [('зона просмотра', r) for r in pick(fig & R.ok & (R.n_zones > 0) & (R.dice >= .6), 1, 'n_zones', asc=False)]
    sel += [('здоровый срез', r) for r in pick(org & (R.y == 3) & ~R.refused & ~R.tumor & (R.n_find == 0) & (R.n_zones == 0)
                                               & (R.seq == 'T1'), 1, 'p_tumor')]
    sel += [('ошибка системы', r) for r in pick(fig & R.tumor & ~R.ok & (R.top_p >= .7), 1, 'top_p', asc=False)
            or pick(fig & R.tumor & (R.n_find > 0) & (R.dice < .4), 1, 'dice')]
    scratch = [('с нуля', r) for r in pick(fig & R.ok & (R.dice >= .75) & clean, 3, 'med_dev', per_class=True)]
    order = list(rs.permutation(len(sel)))          # врач видит случаи в случайном порядке, категории — только в ключе
    sel = [sel[j] for j in order]
    key, dicom_src = [], None
    for n, (cat, r) in enumerate(sel + scratch, 1):
        is_scr = cat == 'с нуля'
        name = f'scratch_{n - len(sel)}' if is_scr else f'case_{n:02d}'
        g, raw, msk, res = store[(r.src, r.i)]
        if dicom_src is None and cat == 'уверена и права' and r.src == 'figshare':
            dicom_src = (name, raw)
        save_png(out / ('scratch' if is_scr else 'cases') / f'{name}.png', g)
        if msk is not None:
            save_png(out / 'key' / 'masks' / f'{name}_mask.png', msk * 255)
        if not is_scr:
            save_png(out / 'key' / 'system' / f'{name}_system.png', cv2.cvtColor(pipeline.overlay(res), cv2.COLOR_RGB2BGR))
        key.append(dict(file=name, category=cat, expect=EXPECT[cat], true_label=r.label, source=r.src, source_file=r.file,
                        system_pred=r.pred, system_type_p=r.top_p, p_tumor=r.p_tumor, dice_vs_doctor_mask=r.dice,
                        findings=r.n_find, zones=r.n_zones, flags=r['flags'], auto_sequence=r.seq, auto_plane=r.plane))
    K = pd.DataFrame(key)
    K.to_csv(out / 'key' / 'cases_key.csv', index=False, encoding='utf-8')
    # DICOM: первый «уверенный» случай Фигшер — чтобы в заключении появились миллиметры
    if dicom_src is not None:
        write_dicom(out / 'cases' / f'{dicom_src[0]}_DEMO_geometry_assumed.dcm', dicom_src[1])
    R.drop(columns=['med_dev']).to_csv(out / 'key' / 'all_candidates.csv', index=False, encoding='utf-8')
    log('Случаи для врача:', K[['file', 'category', 'true_label', 'system_pred', 'dice_vs_doctor_mask']].to_string(index=False))
    return dict(cases=len(sel), scratch=len(scratch), categories=K.category.value_counts().to_dict(),
                missing=[c for c in EXPECT if c not in set(K.category)])


# ---------------------------------------------------------------- 4. полоса v2 на BraTS (T1 с контрастом)
def _brats_extra(S, refused):
    """Понимает ли система, что данные чужие: сколько срезов она отклонила, на скольких предупредила («ood» — снимок
    не похож на обучение, «sequence» — сомнение в режиме МРТ) и что говорит правило двух сигналов на срезах, на которые
    она ответила. Пометка «low_resolution» стоит на всех срезах BraTS (240x240) и предупреждением о чужих данных не считается."""
    if not len(S):
        return {}
    ck = S.checks.fillna('')
    ood, seqw = ck.str.contains('ood'), ck.str.contains('sequence')
    warn = ood | seqw
    n_all = len(S) + refused
    lv = lambda d: {k: int((d.level == k).sum()) for k in ('found', 'review', 'none')}
    return dict(answered_slices=int(len(S)), ood_warned_slices=int(ood.sum()), sequence_warned_slices=int(seqw.sum()),
                refused_or_warned_share=round(float((refused + warn.sum()) / n_all), 3),
                answered_without_domain_warning=int((~warn).sum()),
                answered_without_domain_warning_share=round(float((~warn).sum() / n_all), 3),
                levels_on_tumor_free_slices=dict(n=int(S['empty'].sum()), **lv(S[S['empty']])),
                levels_on_tc_slices=dict(n=int(S.has_tc.sum()), **lv(S[S.has_tc])),
                levels_on_tumor_free_slices_no_warning=dict(n=int((S['empty'] & ~warn).sum()), **lv(S[S['empty'] & ~warn])),
                levels_on_tc_slices_no_warning=dict(n=int((S.has_tc & ~warn).sum()), **lv(S[S.has_tc & ~warn])))


def brats_band(M, root, step=2, alpha=0.10, seed=0, log=print):
    """Тестовые пациенты того же разбиения, что в brats.py (seed 0, 80/10/10). На вход прототипа — срез T1gd как есть
    (240x240, уже без черепа). Эталон для контура — ядро опухоли TC (видимая на T1c часть: некроз + контрастируемая)."""
    import nibabel as nib
    from .brats import MOD, norm
    imgs = sorted(glob.glob(str(Path(root) / 'imagesTr' / 'BRATS_*.nii.gz')))
    perm = np.random.RandomState(seed).permutation(len(imgs))
    n = len(imgs)
    test = sorted(perm[int(.9 * n):])
    per_pat, sl = [], []
    t0 = time.time()
    for q, k in enumerate(test):
        p = imgs[k]
        v = nib.load(p).get_fdata(dtype=np.float32)
        lab = nib.load(p.replace('imagesTr', 'labelsTr')).get_fdata(dtype=np.float32).astype(np.uint8)
        brain = (v[..., MOD['flair']] > 0).reshape(-1, v.shape[2]).sum(0) > 500
        risks, dices, n_contour, n_tc, refused, fc, n_empty = [], [], 0, 0, 0, 0, 0
        for z in np.where(brain)[0][::step]:
            g = norm(np.rot90(v[:, :, z, MOD['t1gd']]))
            tc = np.rot90((lab[:, :, z] == 2) | (lab[:, :, z] == 3)).astype(np.uint8)
            wt = np.rot90(lab[:, :, z] > 0)
            tc256 = cv2.resize(tc, (256, 256), interpolation=cv2.INTER_NEAREST)
            res = pipeline.analyze(M, g)
            if res['refused']:
                refused += 1
                continue
            m, b = res['arrays']['mask'], res['arrays']['band']
            has_tc = tc256.sum() >= 30
            if not wt.any():
                n_empty += 1
                fc += int(m.any())
            if has_tc:
                n_tc += 1
                dices.append(dice1(m, tc256))
                if m.any():
                    n_contour += 1
                    outer = (m | b).astype(bool)
                    risks.append(float((tc256.astype(bool) & ~outer).sum() / tc256.sum()))
            sl.append(dict(pid=Path(p).name, z=int(z), has_tc=bool(has_tc), empty=bool(not wt.any()), contour=bool(m.any()), refused=False,
                           tumor=bool(res['tumor']), level=res.get('level'), p_tumor=res['p_tumor'], seq=res['auto_sequence'],
                           checks='|'.join(c['code'] for c in res['checks'])))
        per_pat.append(dict(pid=Path(p).name, slices_tc=n_tc, slices_with_contour=n_contour, refused_slices=refused,
                            risk=float(np.mean(risks)) if risks else None, dice_tc=float(np.mean(dices)) if dices else None,
                            empty_slices=n_empty, false_contour_empty=fc))
        if q % 10 == 0:
            log(f'  BraTS: {q + 1}/{len(test)} пациентов, {time.time() - t0:.0f} с')
    P = pd.DataFrame(per_pat)
    r = P.risk.dropna().values
    bs = np.random.RandomState(0)
    ci = np.percentile([bs.choice(r, len(r)).mean() for _ in range(2000)], [2.5, 97.5]) if len(r) else [None, None]
    S = pd.DataFrame(sl)
    out = dict(patients=len(P), slices=int(len(S) + P.refused_slices.sum()), refused_slices=int(P.refused_slices.sum()),
               mean_risk=round(float(r.mean()), 4) if len(r) else None, risk_ci=[round(float(x), 4) for x in ci] if len(r) else None,
               coverage=round(1 - float(r.mean()), 4) if len(r) else None,
               share_patients_risk_gt_alpha=round(float((r > alpha).mean()), 3) if len(r) else None,
               slices_with_tc=int(P.slices_tc.sum()), contour_found_on_tc_slices=round(float(P.slices_with_contour.sum() / max(P.slices_tc.sum(), 1)), 3),
               mean_dice_tc=round(float(P.dice_tc.dropna().mean()), 3),
               false_contour_on_tumor_free_slices=round(float(P.false_contour_empty.sum() / max(P.empty_slices.sum(), 1)), 3),
               auto_sequence_counts=S.seq.value_counts().to_dict() if len(S) else {},
               **_brats_extra(S, int(P.refused_slices.sum())),
               note='чужие данные: другие центры и томографы, срезы без черепа (skull-stripped), 240x240; калибровка и r — из calib.json без изменений')
    log('BraTS, полоса v2:', json.dumps(out, ensure_ascii=False))
    return out, P, S
