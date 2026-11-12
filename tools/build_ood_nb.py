"""Собирает notebooks/stage2_ood.ipynb: предупреждение «снимок не похож на обучение» по ближайшим обучающим снимкам.
Правило приёма записано в docs/stage2_protocol.md до запуска. Kaggle, GPU."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Этап 2 · предупреждение «снимок не похож на обучение» (Kaggle, GPU)
Входы: выход ноутбука обучения (stage2_results.zip) и датасет **briscdataset/brisc2025**. Секрет `GITHUB_TOKEN`. Internet: On.
Прежнее предупреждение (расстояние Махаланобиса) не сработало ни на одном срезе чужого набора BraTS. Здесь — одна заранее
записанная попытка: расстояние до 10-го ближайшего обучающего снимка, порог — 97.5-й перцентиль валидации.
Правило приёма — в `docs/stage2_protocol.md`. Итог — `results/stage2/ood_check.json`; при приёме — новые параметры в
`weights/stage2/input_models.npz`.
""")
code(r"""
import os, sys, subprocess, glob, zipfile, json, shutil, time
from pathlib import Path
REPO_URL, REPO = 'https://github.com/blond7I/almazov-mri-tumor', '/tmp/almazov_repo'
shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, REPO], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.31', 'segmentation-models-pytorch==0.6.0',
                'pydicom', 'nibabel', 'h5py', 'gdown'], check=True)
sys.path.insert(0, REPO + '/src')
import numpy as np, pandas as pd, torch
from almazov import pipeline, brisc, data, nets, brats, explain
from almazov import inputcheck as ic
from almazov.common import prep, NO_TUMOR
INPUT = os.environ.get('ALMAZOV_INPUT', '/kaggle/input')
src = sorted(glob.glob(INPUT + '/**/stage2_results.zip', recursive=True))
assert src, 'Подключите выход ноутбука обучения (Add Input -> Your Work)'
BR = brisc.find_root(INPUT)
assert BR, 'Подключите датасет BRISC 2025 (Add Input -> Datasets -> briscdataset/brisc2025)'
RES, OUT = '/tmp/res', '/kaggle/working/ood_out'
shutil.rmtree(RES, ignore_errors=True); os.makedirs(RES)
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
with zipfile.ZipFile(src[0]) as z:
    for n in ['index.npz', 'extra_meta.csv']:
        z.extract(n, RES)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG = open(OUT + '/ood_log.txt', 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(time.strftime('%H:%M:%S ') + s + '\n'); LOG.flush()
log('device', DEV, 'BRISC', BR)

def _push(files, msg):
    # files: {путь в репозитории: локальный путь}; свежий клон -> копия -> commit -> push
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret('GITHUB_TOKEN').strip()
    R2 = '/tmp/push_repo'; shutil.rmtree(R2, ignore_errors=True)
    subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, R2], check=True)
    for dst, s in files.items():
        d = Path(R2) / dst
        d.parent.mkdir(parents=True, exist_ok=True); shutil.copy(s, d)
    def git(*a):
        r = subprocess.run(['git', '-C', R2, *a], capture_output=True, text=True)
        print((r.stdout + r.stderr).replace(tok, '***')[-800:])
        if r.returncode: raise RuntimeError('git ' + a[0] + ': ошибка')
    git('config', 'user.name', 'blond7I'); git('config', 'user.email', 'blond7I@users.noreply.github.com')
    git('add', '-A'); git('commit', '-q', '-m', msg + '\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>\n'
                                         'Claude-Session: https://claude.ai/code/session_01KWjYFaDqWdYY8y4uMMdgwF')
    git('push', f'https://x-access-token:{tok}@github.com/blond7I/almazov-mri-tumor.git', 'HEAD:main')
    print('Отправлено в GitHub:', list(files))

def push(files, msg):
    # GitHub — удобство, не условие: при любой ошибке результаты остаются в Output этого ноутбука
    for k in range(3):
        try:
            _push(files, msg); return True
        except Exception as e:
            print('Не удалось отправить в GitHub (попытка', k + 1, '):', type(e).__name__, str(e)[:200]); time.sleep(20)
    print('Результаты остаются в Output (вкладка Output ноутбука)'); return False
""")
code(r"""
# данные — тем же путём, что в ноутбуке обучения: архив организаторов (box) + Figshare
DL = '/tmp/almazov_data'; os.makedirs(DL, exist_ok=True)
BOX = 'https://box.almazovcentre.ru/index.php/s/ZD0MIbCoKEBMziJ/download'
org = data.find_file('*Epic*CSCR*.zip', [INPUT])
if not org:
    org = os.path.join(DL, 'Epic_and_CSCR_hospital_Dataset.zip')
    if not (os.path.exists(org) and zipfile.is_zipfile(org)):
        import urllib.request; log('Скачиваю архив организаторов...'); urllib.request.urlretrieve(BOX, org)
fig = data.find_file('figshare256.npz', [INPUT])
if not fig:
    fig = os.path.join(DL, 'figshare256.npz')
    try:
        import gdown; gdown.download(id='1MDAF0FXIeg89_w-G9-nKBv46tmzpY77i', output=fig, quiet=True)
    except Exception as e:
        log('gdown:', e)
    if not (os.path.exists(fig) and os.path.getsize(fig) > 1e6):
        fig = data.build_figshare_npz(fig, DL + '/fig_raw', log=log)
df, A, ex_df, AX = data.load_org(REPO + '/data/split_v3.csv', org, log=log)
fX, fM, fy, fpid = data.load_figshare(fig)
S = data.build_sets(df, A, fX, fM, fy, fpid, log=log)
I = np.load(RES + '/index.npz')
assert np.array_equal(S['TE'], I['TE']) and np.array_equal(S['s_te'], I['s_te']), 'Разбиение не совпало с обучением'
ex_meta = pd.read_csv(RES + '/extra_meta.csv', keep_default_na=False)
assert len(ex_meta) == len(AX)
""")
code(r"""
# признаки: тот же классификатор этапа 1 и та же функция, что в прототипе (pipeline._infer)
M = pipeline.load_models(REPO, cache_dir='/tmp/almazov_cache', device=DEV)
log('Модели:', M.version)
net = M.cls[0][1]
TD = torch.device(DEV)
F = lambda X: nets.cls_features(net, X, TD).astype(np.float32)
t0 = time.time()
fA, fF, fE = F(A), F(fX), F(AX)
dfb, Xb, Mb = brisc.load(BR, log=log)
fB = F(Xb)
log(f'Признаки: организаторы {fA.shape}, Figshare {fF.shape}, полный архив {fE.shape}, BRISC {fB.shape}, {time.time() - t0:.0f} с')

# BraTS: те же тестовые пациенты и срезы, что в extras.brats_band (seed 0, шаг 2)
import nibabel as nib
from almazov.brats import MOD, norm
root = brats.download('/tmp/brats', log=log)
imgs = sorted(glob.glob(str(Path(root) / 'imagesTr' / 'BRATS_*.nii.gz')))
perm = np.random.RandomState(0).permutation(len(imgs))
test = sorted(perm[int(.9 * len(imgs)):])
XT, rows = [], []
for q, k in enumerate(test):
    p = imgs[k]
    v = nib.load(p).get_fdata(dtype=np.float32)
    brain = (v[..., MOD['flair']] > 0).reshape(-1, v.shape[2]).sum(0) > 500
    for zz in np.where(brain)[0][::2]:
        XT.append(prep(np.ascontiguousarray(norm(np.rot90(v[:, :, zz, MOD['t1gd']])))))
        rows.append((Path(p).name, int(zz)))
XT = np.stack(XT); fT = F(XT)
bt = pd.DataFrame(rows, columns=['pid', 'z'])
ans = pd.read_csv(REPO + '/results/stage2/brats_band_slices.csv')[['pid', 'z', 'level', 'has_tc', 'empty']]
bt = bt.merge(ans, on=['pid', 'z'], how='left'); bt['answered'] = bt.level.notna()
log(f'BraTS: срезов {len(bt)}, из них система отвечает на {int(bt.answered.sum())} (ожидалось 3281 и 2215)')
assert len(bt) == len(fT) and bt.answered.sum() > 0
""")
code(r"""
# подбор по протоколу: опорный набор = обучение; порог = 97.5-й перцентиль валидации; тест в подборе не участвует
K, NCOMP, Q = 10, 256, 97.5
TR, VA, TEU, s_tr, s_va, s_te = S['TR'], S['VA'], S['TEU'], S['s_tr'], S['s_va'], S['s_te']
ref_raw = np.concatenate([fA[TR], fF[s_tr]])
unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
Rn = unit(ref_raw); mu = Rn.mean(0)
_, _, Vt = np.linalg.svd(Rn - mu, full_matrices=False)
PCA = np.vstack([mu, Vt[:NCOMP]]).astype(np.float32)
REF = ic.knn_embed(ref_raw, PCA).astype(np.float16).astype(np.float32)      # ровно то, что попадёт в прототип
dist = lambda X, k=K: ic.knn_distance(ic.knn_embed(X, PCA), REF, k)
def dist_raw(X, k=K):                                                        # без главных компонент — для сведения
    E, out = unit(X), []
    for s in range(0, len(E), 512):
        out.append(1 - np.partition(E[s:s + 512] @ Rn.T, -k, axis=1)[:, -k])
    return np.concatenate(out) if out else np.zeros(0, np.float32)
fm = M.input_models
def dist_maha(X):                                                            # прежнее предупреждение
    f = (X.astype(np.float64) - fm.pca[0]) @ fm.pca[1:].T - fm.mu
    return np.sqrt(np.clip(np.einsum('ni,ij,nj->n', f, fm.P, f), 0, None))

# наборы для проверки
nw = ((ex_meta.status == 'новый') & (ex_meta.label.astype(int) >= 0)).values
bmap = pd.read_csv(REPO + '/results/stage2/brisc_map.csv', keep_default_na=False)
spl = pd.read_csv(REPO + '/results/stage2/brisc_normals_split.csv').set_index('file').part
bpart = dfb.file.map(spl).fillna('').values
blab = dfb.label.values.astype(int)
opart = dfb.file.map(bmap.set_index('file').our_part).fillna('').values
lab = pd.read_csv(REPO + '/data/sequence_plane_labels.csv', keep_default_na=False)
lab = lab[(lab.part == 'test') & (lab.label.astype(int) == NO_TUMOR)]
SETS = {
    'val': np.concatenate([fA[VA], fF[s_va]]),
    'test': np.concatenate([fA[TEU], fF[s_te]]),
    'test_org_unique': fA[TEU], 'test_figshare': fF[s_te],
    'new_images_92': fE[nw],
    'brisc_holdout_normals': fB[bpart == 'holdout'],
    'brisc_train_c_normals': fB[bpart == 'train_c'],
    'brisc_tumors_our_test': fB[(opart == 'test') & (blab != NO_TUMOR)],
    'brisc_tumors_external': fB[(opart == 'external') & (blab != NO_TUMOR)],
    'brats_answered': fT[bt.answered.values], 'brats_refused': fT[~bt.answered.values], 'brats_all': fT,
}
for sname in ['T1', 'T2', 'FLAIR', 'КТ']:
    SETS['test_normals_' + sname] = fA[lab[lab.seq == sname].idx.values.astype(int)]
log({k: len(v) for k, v in SETS.items()})
assert len(SETS['new_images_92']) == 92 and len(SETS['brisc_holdout_normals']) == 629, 'состав наборов не совпал с опубликованным'

def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(float((c - h) / d), 4), round(float((c + h) / d), 4)]
def table(D, thr):
    out = {}
    for k, d in D.items():
        w = int((d > thr).sum())
        out[k] = dict(n=len(d), warned=w, rate=round(w / max(len(d), 1), 4), ci=wilson(w, len(d)),
                      q05=round(float(np.quantile(d, .05)), 4) if len(d) else None, median=round(float(np.median(d)), 4) if len(d) else None,
                      q95=round(float(np.quantile(d, .95)), 4) if len(d) else None)
    return out

D = {k: dist(v) for k, v in SETS.items()}
THR = float(np.percentile(D['val'], Q))
RATES = table(D, THR)
for k, r in RATES.items(): log(f"  {k:26s} n={r['n']:5d}  предупреждений {r['warned']:5d}  {r['rate']:.1%}  медиана расстояния {r['median']}")

# чувствительность (на решение не влияет)
SENS = {}
for name, fn, q in [('k=5', lambda X: dist(X, 5), Q), ('k=20', lambda X: dist(X, 20), Q), ('перцентиль 95', dist, 95), ('перцентиль 99', dist, 99),
                    ('без главных компонент', dist_raw, Q), ('Махаланобис, 97.5', dist_maha, Q)]:
    d = {k: fn(v) for k, v in SETS.items()}
    t = float(np.percentile(d['val'], q))
    SENS[name] = {k: round(float((v > t).mean()), 4) if len(v) else None for k, v in d.items()}
    log(f"  [{name}] тест {SENS[name]['test']:.1%}, BraTS (отвечает) {SENS[name]['brats_answered']:.1%}, BRISC отложенные здоровые {SENS[name]['brisc_holdout_normals']:.1%}")
DM = {k: dist_maha(v) for k, v in SETS.items()}
MAHA = table(DM, float(fm.thr))
log(f"Прежнее предупреждение (Махаланобис, порог {fm.thr:.2f}): тест {MAHA['test']['rate']:.1%}, BraTS (отвечает) {MAHA['brats_answered']['rate']:.1%}")
""")
code(r"""
# ошибается ли система чаще там, где предупреждение сработало (BRISC — строки прототипа; 92 новых снимка — тип по ансамблю)
dB = dist(fB); wB = pd.Series(dB > THR, index=dfb.file.values)
ERR = {}
for tag, f in [('brisc_our_test', 'brisc_rows_our_test.csv'), ('brisc_external', 'brisc_rows_external.csv')]:
    r = pd.read_csv(REPO + '/results/stage2/' + f, keep_default_na=False)
    r = r[~r.refused.astype(str).eq('True')].copy()
    r['warn'] = r.file.map(wB).astype(bool); r['okb'] = r.ok.astype(str).eq('True')
    r['dice_f'] = pd.to_numeric(r.dice, errors='coerce')
    ERR[tag] = {('warned' if w else 'not_warned'): dict(n=int(len(g)), type_accuracy=round(float(g.okb.mean()), 4) if len(g) else None,
                                                         mean_dice=round(float(g.dice_f.mean()), 4) if g.dice_f.notna().any() else None)
                for w, g in r.groupby('warn')}
    log(tag, ERR[tag])
Pn = explain.probs(M, AX[nw], TD); yn = ex_meta.label.astype(int).values[nw]; wn = D['new_images_92'] > THR
ERR['new_images_92'] = {('warned' if w else 'not_warned'): dict(n=int((wn == w).sum()),
                         type_accuracy=round(float((Pn.argmax(1) == yn)[wn == w].mean()), 4) if (wn == w).any() else None) for w in [True, False]}
log('new_images_92', ERR['new_images_92'])
bw = bt.assign(warn=D['brats_all'] > THR)
ba = bw[bw.answered]
BR_PAT = dict(patients=int(ba.pid.nunique()), median_patient_rate=round(float(ba.groupby('pid').warn.mean().median()), 4),
              patients_with_majority_warned=int((ba.groupby('pid').warn.mean() >= 0.5).sum()),
              warned_on_tc_slices=round(float(ba[ba.has_tc.astype(bool)].warn.mean()), 4),
              warned_on_tumor_free_slices=round(float(ba[ba['empty'].astype(bool)].warn.mean()), 4),
              wrong_none_on_tc_without_warning=int(((ba.level == 'none') & ba.has_tc.astype(bool) & ~ba.warn).sum()),
              wrong_none_on_tc_total=int(((ba.level == 'none') & ba.has_tc.astype(bool)).sum()),
              found_on_tumor_free_without_warning=int(((ba.level == 'found') & ba['empty'].astype(bool) & ~ba.warn).sum()),
              found_on_tumor_free_total=int(((ba.level == 'found') & ba['empty'].astype(bool)).sum()))
log('BraTS по пациентам и по ошибкам:', BR_PAT)

# решение по правилу, записанному до запуска
ok1 = RATES['test']['rate'] <= 0.05
ok2 = RATES['brats_answered']['rate'] >= 0.50
accepted = bool(ok1 and ok2)
log(f"Правило: тест {RATES['test']['rate']:.1%} <= 5% — {'да' if ok1 else 'НЕТ'}; BraTS {RATES['brats_answered']['rate']:.1%} >= 50% — {'да' if ok2 else 'НЕТ'} => "
    + ('ПРИНЯТО' if accepted else 'ОТКЛОНЕНО'))
R = dict(models=M.version, commit=subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip(),
         protocol=dict(features='ResNet-50 этапа 1, 2048 после пулинга, единичная длина', reference='обучение: организаторы TR + Figshare s_tr',
                       n_reference=int(len(REF)), pca_components=NCOMP, k=K, threshold_percentile=Q, threshold_set='валидация: организаторы VA + Figshare s_va',
                       n_validation=int(len(D['val'])), threshold=round(THR, 5)),
         rates=RATES, sensitivity_not_used_for_decision=SENS, old_mahalanobis=dict(threshold=round(float(fm.thr), 3), rates=MAHA),
         errors_where_warned=ERR, brats_detail=BR_PAT,
         decision=dict(rule_test_le_5pct=bool(ok1), rule_brats_ge_50pct=bool(ok2), accepted=accepted,
                       test_rate=RATES['test']['rate'], brats_answered_rate=RATES['brats_answered']['rate']))
json.dump(R, open(OUT + '/ood_check.json', 'w'), ensure_ascii=False, indent=1, default=float)
rows_out = pd.concat([pd.DataFrame(dict(set=k, i=np.arange(len(d)), distance=np.round(d, 5), warn=d > THR)) for k, d in D.items() if k != 'brats_all'])
rows_out.to_csv(OUT + '/ood_rows.csv', index=False)
bw.assign(distance=np.round(D['brats_all'], 5)).to_csv(OUT + '/ood_brats_slices.csv', index=False)
files = {'results/stage2/ood_check.json': OUT + '/ood_check.json', 'results/stage2/ood_rows.csv': OUT + '/ood_rows.csv',
         'results/stage2/ood_brats_slices.csv': OUT + '/ood_brats_slices.csv', 'results/stage2/ood_log.txt': OUT + '/ood_log.txt'}
if accepted:
    z = dict(np.load(REPO + '/weights/stage2/input_models.npz', allow_pickle=True))
    z.update(knn_ref=REF.astype(np.float16), knn_pca=PCA, knn_k=np.int64(K), knn_thr=np.float64(THR))
    np.savez_compressed(OUT + '/input_models.npz', **z)
    f2 = ic.FeatureModels(OUT + '/input_models.npz')                      # сверка: прототип даёт то же решение, что подбор
    chk = [f2.check(v.astype(np.float16)) for v in np.concatenate([SETS['test'][:40], SETS['brats_answered'][:40]])]
    exp = np.concatenate([D['test'][:40], D['brats_answered'][:40]])
    agree = np.mean([c['ood'] == bool(e > THR) for c, e in zip(chk, exp)])
    log(f'Сверка с прототипом: решения совпали в {agree:.0%} из 80, макс. расхождение расстояния {max(abs(c["knn_distance"] - e) for c, e in zip(chk, exp)):.4f}')
    assert agree >= 0.95, 'прототип и подбор расходятся'
    log(f'input_models.npz: {os.path.getsize(OUT + "/input_models.npz") / 1e6:.1f} МБ')
    files['weights/stage2/input_models.npz'] = OUT + '/input_models.npz'
LOG.flush()
push(files, ('Out-of-domain warning by nearest training images: ' + ('accepted' if accepted else 'rejected by the pre-registered rule')
             + f" (test {RATES['test']['rate']:.1%}, BraTS {RATES['brats_answered']['rate']:.1%}; run on Kaggle)"))
print('Готово')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
