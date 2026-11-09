"""Собирает notebooks/stage2_brisc.ipynb: BRISC — проверка текущего прототипа (A) и обучение кандидатов (B)."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Этап 2 · BRISC 2025: проверка прототипа и кандидаты с искажениями качества (Kaggle, GPU)
Входы: выход ноутбука обучения (stage2_results.zip) и датасет **briscdataset/brisc2025**. Секрет `GITHUB_TOKEN`. Internet: On. GPU T4.
Правило принятия кандидатов записано в docs/stage2_protocol.md до запуска. Итоги A — сразу в GitHub; кандидаты — в Output
(stage2b_results.zip, stage2b_weights.zip) для общей оценки по протоколу.
""")
code(r"""
import os, sys, subprocess, glob, zipfile, json, shutil, time
from pathlib import Path
REPO_URL, REPO = 'https://github.com/blond7I/almazov-mri-tumor', '/tmp/almazov_repo'
shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, REPO], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.31', 'segmentation-models-pytorch==0.6.0', 'pydicom', 'h5py', 'gdown'], check=True)
sys.path.insert(0, REPO + '/src')
import numpy as np, pandas as pd, torch
from almazov import pipeline, brisc, data, nets, train
INPUT = os.environ.get('ALMAZOV_INPUT', '/kaggle/input')
src = sorted(glob.glob(INPUT + '/**/stage2_results.zip', recursive=True))
assert src, 'Подключите выход ноутбука обучения (Add Input -> Your Work)'
BR = brisc.find_root(INPUT)
assert BR, 'Подключите датасет BRISC 2025 (Add Input -> Datasets -> briscdataset/brisc2025)'
RES, OUT = '/tmp/res', '/kaggle/working/b'
shutil.rmtree(RES, ignore_errors=True); os.makedirs(RES); os.makedirs(OUT, exist_ok=True)
with zipfile.ZipFile(src[0]) as z:
    for n in ['index.npz', 'extra_meta.csv', 'extra.npz']:
        z.extract(n, RES)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG = open(OUT + '/brisc_log.txt', 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(time.strftime('%H:%M:%S ') + s + '\n'); LOG.flush()
log('device', DEV, 'BRISC', BR)

def push(files, msg):
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret('GITHUB_TOKEN').strip()
    R2 = '/tmp/push_repo'; shutil.rmtree(R2, ignore_errors=True)
    subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, R2], check=True)
    for dst, s in files.items():
        d = Path(R2) / dst
        if os.path.isdir(s):
            shutil.rmtree(d, ignore_errors=True); shutil.copytree(s, d)
        else:
            d.parent.mkdir(parents=True, exist_ok=True); shutil.copy(s, d)
    def git(*a):
        r = subprocess.run(['git', '-C', R2, *a], capture_output=True, text=True)
        print((r.stdout + r.stderr).replace(tok, '***')[-800:])
        if r.returncode: raise RuntimeError('git ' + a[0] + ': ошибка')
    git('config', 'user.name', 'blond7I'); git('config', 'user.email', 'blond7I@users.noreply.github.com')
    git('add', '-A'); git('commit', '-q', '-m', msg + '\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n'
                                         'Claude-Session: https://claude.ai/code/session_01KWjYFaDqWdYY8y4uMMdgwF')
    git('push', f'https://x-access-token:{tok}@github.com/blond7I/almazov-mri-tumor.git', 'HEAD:main')
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
# A) BRISC без переобучения: сопоставление и текущий прототип на нашем тесте и на внешних снимках
dfb, Xb, Mb = brisc.load(BR, log=log)
dfb = brisc.assign(dfb, Xb, A, df, AX, ex_meta, fX, S['fpart'], dev=DEV, log=log)
dfb.drop(columns=['img', 'mask']).assign(has_mask=dfb['mask'] != '').to_csv(OUT + '/brisc_map.csv', index=False)
M = pipeline.load_models(REPO, cache_dir='/tmp/almazov_cache', device=DEV)
log('Модели:', M.version)
RA = dict(models=M.version, mapping=dfb.our_part.value_counts().to_dict(),
          mapping_by_class={str(k): v for k, v in dfb.groupby('label').our_part.value_counts().unstack(fill_value=0).to_dict('index').items()})
for name, sub in [('our_test', ['test']), ('external', ['external']), ('extra_new', ['extra-new'])]:
    rows, RA[name] = brisc.evaluate(M, dfb, sub, log=log)
    rows.to_csv(OUT + f'/brisc_rows_{name}.csv', index=False)
RA['sequence_agreement'] = brisc.sequence_agreement(dfb, REPO + '/data/sequence_plane_labels.csv')
json.dump(RA, open(OUT + '/brisc_eval.json', 'w'), ensure_ascii=False, indent=1, default=float)
push({'results/stage2/brisc_eval.json': OUT + '/brisc_eval.json', 'results/stage2/brisc_map.csv': OUT + '/brisc_map.csv',
      'results/stage2/brisc_rows_our_test.csv': OUT + '/brisc_rows_our_test.csv',
      'results/stage2/brisc_rows_external.csv': OUT + '/brisc_rows_external.csv', 'results/stage2/brisc_log.txt': OUT + '/brisc_log.txt'},
     'BRISC 2025: mapping to our split and current prototype on our test and external images (run on Kaggle)')
""")
code(r"""
# B) кандидаты: тот же рецепт + искажения качества (BigAug); сегментация — ещё и маски BRISC из нашей обучающей части
LAB = S['LAB']
sets = dict(fva=fX[S['s_va']], fte=fX[S['s_te']], ova=A[S['VA']], ote=A[S['TE']])
tr_n = S['TR'][LAB[S['TR']] == 3]
q = S['q_tr']
bt = dfb.index[(dfb.our_part == 'train') & (dfb['mask'] != '') & (dfb.label < 3)].values
bn = dfb.index[(dfb.our_part == 'train') & (dfb.label == 3)].values
Xs = np.concatenate([fX[S['s_tr']], A[S['om_idx'][q]], A[tr_n], Xb[bt], Xb[bn]])
Ms = np.concatenate([fM[S['s_tr']], S['OM'][q], np.zeros((len(tr_n), 256, 256), np.uint8), Mb[bt], np.zeros((len(bn), 256, 256), np.uint8)])
log(f'Сегментация: примеров {len(Xs)} (из них BRISC с маской {len(bt)}, BRISC здоровых {len(bn)})')
CK = '/tmp/ck'; os.makedirs(CK, exist_ok=True)
SB = {}
bsel = {k: dfb.index[dfb.our_part.isin(v)].values for k, v in [('brisc_test', ['test']), ('brisc_val', ['val']), ('brisc_external', ['external'])]}
for sd in (1, 2):
    tag = f'seg_b{sd}'
    t0 = time.time()
    net, info, ck = train.train_seg(Xs, Ms, fX[S['s_va']], fM[S['s_va']], torch.device(DEV), CK, tag, 100 + sd, 20, bs=16,
                                    workers=4, log=log, strong_aug=True)
    res = {k: nets.to_u8(nets.seg_maps(net, X, torch.device(DEV))) for k, X in sets.items()}
    res.update({k: nets.to_u8(nets.seg_maps(net, Xb[ix], torch.device(DEV))) for k, ix in bsel.items()})
    np.savez_compressed(OUT + f'/{tag}.npz', **{k: v for k, v in res.items() if k in sets}, meta=json.dumps({k: v for k, v in info.items() if k != 'history'}))
    np.savez_compressed(OUT + f'/brisc_maps_{tag}.npz', **{k: v for k, v in res.items() if k.startswith('brisc_')})
    torch.save(nets.half_state(net.state_dict()), OUT + f'/w_{tag}.pt')
    SB[tag] = dict(val_dice=info['val_dice'], minutes=round((time.time() - t0) / 60, 1),
                   extra_seg=nets.to_u8(nets.seg_maps(net, AX, torch.device(DEV))))
    log(f'[{tag}] val Dice {info["val_dice"]:.4f}, {SB[tag]["minutes"]} мин'); del net; torch.cuda.empty_cache()
for sd in (1, 2):
    tag = f'cls_b{sd}'
    t0 = time.time()
    m, info, ck = train.train_cls(A, LAB, S['TR'], S['VA'], torch.device(DEV), CK, tag, 100 + sd, 15, bs=64, workers=4, log=log, strong_aug=True)
    lva, lte = nets.cls_logits(m, A[S['VA']], torch.device(DEV)), nets.cls_logits(m, A[S['TE']], torch.device(DEV))
    np.savez(OUT + f'/{tag}.npz', lva=lva, lte=lte, meta=json.dumps({k: v for k, v in info.items() if k != 'history'}))
    torch.save(nets.half_state(m.state_dict()), OUT + f'/w_{tag}.pt')
    SB[tag] = dict(val_macro_f1=info['val_macro_f1'], minutes=round((time.time() - t0) / 60, 1),
                   extra_logits=nets.cls_logits(m, AX, torch.device(DEV)))
    log(f'[{tag}] val macro-F1 {info["val_macro_f1"]:.4f}, {SB[tag]["minutes"]} мин'); del m; torch.cuda.empty_cache()
np.savez_compressed(OUT + '/extra_b.npz', **{f'seg_{k}': v['extra_seg'] for k, v in SB.items() if 'extra_seg' in v},
                    **{f'logits_{k}': v['extra_logits'] for k, v in SB.items() if 'extra_logits' in v})
""")
code(r"""
# сравнение контуров на BRISC (маски врачей): текущий ансамбль против текущего + кандидатов; упаковка для общей оценки
from almazov.common import clean_mask, dice
cur = [n for n, _ in M.seg]
def maps_cur(ix):
    return np.stack([nets.seg_maps(net, Xb[ix], torch.device(DEV)) for _, net in M.seg])        # (m, n, 256, 256)
cmp = {}
for k, ix in bsel.items():
    if not len(ix): continue
    tum = dfb.label.values[ix] < 3
    gt = Mb[ix] > 0
    mc = maps_cur(ix)
    mb = np.stack([np.load(OUT + f'/brisc_maps_seg_b{sd}.npz')[k].astype(np.float32) / 255 for sd in (1, 2)])
    def dsc(P):
        pr = np.stack([clean_mask(p) for p in P.mean(0)]) > 0
        return float(dice(pr[tum], gt[tum]).mean()) if tum.any() else None
    cmp[k] = dict(n_tumor=int(tum.sum()), current=dsc(mc), current_plus_candidates=dsc(np.concatenate([mc, mb])), candidates_only=dsc(mb))
log('BRISC, Dice с контурами врачей:', json.dumps(cmp, ensure_ascii=False))
json.dump(dict(candidates={k: {kk: vv for kk, vv in v.items() if not kk.startswith('extra_')} for k, v in SB.items()}, brisc_dice=cmp),
          open(OUT + '/stage2b_summary.json', 'w'), ensure_ascii=False, indent=1, default=float)
with zipfile.ZipFile('/kaggle/working/stage2b_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in glob.glob(OUT + '/seg_b*.npz') + glob.glob(OUT + '/cls_b*.npz') + [OUT + '/extra_b.npz', OUT + '/stage2b_summary.json', OUT + '/brisc_log.txt']:
        z.write(p, os.path.basename(p))
with zipfile.ZipFile('/kaggle/working/stage2b_weights.zip', 'w', zipfile.ZIP_STORED) as z:
    for p in glob.glob(OUT + '/w_*.pt'):
        z.write(p, os.path.basename(p))
push({'results/stage2/stage2b_summary.json': OUT + '/stage2b_summary.json', 'results/stage2/brisc_log.txt': OUT + '/brisc_log.txt'},
     'BRISC candidates: training summary and Dice against expert masks (run on Kaggle)')
for p in glob.glob(OUT + '/brisc_maps_*.npz') + glob.glob(OUT + '/w_*.pt'): os.remove(p)
print('Готово: stage2b_results.zip, stage2b_weights.zip')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
