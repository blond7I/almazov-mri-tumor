"""Собирает notebooks/stage2_normals.ipynb: эксперимент C — классификатор с новыми здоровыми T1 из BRISC.

Зачем: на здоровых T1 текущая система поднимает ложную тревогу примерно в 11% случаев (наш тест 3 из 21; внешние снимки
BRISC 77 из 725), потому что в обучении здоровые были в основном T2/FLAIR/КТ. Здесь половина внешних здоровых снимков BRISC
(их нет в наших данных) добавляется в обучение как «опухоли нет», вторая половина — отложенная проверка; главная независимая
проверка — здоровые T1 нашего теста (другие источники).
"""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Этап 2 · эксперимент C: здоровые T1 из BRISC в обучении классификатора (Kaggle, GPU)
Входы: выход ноутбука обучения (stage2_results.zip) и датасет briscdataset/brisc2025. Секрет GITHUB_TOKEN. Internet: On. GPU T4.
Правило принятия — docs/stage2_protocol.md (записано до запуска).
""")
code(r"""
import os, sys, subprocess, glob, zipfile, json, shutil, time
from pathlib import Path
REPO_URL, REPO = 'https://github.com/blond7I/almazov-mri-tumor', '/tmp/almazov_repo'
shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, REPO], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.31', 'segmentation-models-pytorch==0.6.0', 'gdown'], check=True)
sys.path.insert(0, REPO + '/src')
import numpy as np, pandas as pd, torch
from PIL import Image
from almazov import data, nets, train, brisc
from almazov.common import prep, small, z1, NO_TUMOR, CLASSES
INPUT = '/kaggle/input'
src = sorted(glob.glob(INPUT + '/**/stage2_results.zip', recursive=True)); assert src
BR = brisc.find_root(INPUT); assert BR
RES, OUT = '/tmp/res', '/kaggle/working/n'
shutil.rmtree(RES, ignore_errors=True); os.makedirs(RES); os.makedirs(OUT, exist_ok=True)
with zipfile.ZipFile(src[0]) as z:
    for n in ['index.npz', 'extra_meta.csv']: z.extract(n, RES)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG = open(OUT + '/normals_log.txt', 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(time.strftime('%H:%M:%S ') + s + '\n'); LOG.flush()

def _push(files, msg):
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret('GITHUB_TOKEN').strip()
    R2 = '/tmp/push_repo'; shutil.rmtree(R2, ignore_errors=True)
    subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, R2], check=True)
    for dst, s in files.items():
        d = Path(R2) / dst; d.parent.mkdir(parents=True, exist_ok=True); shutil.copy(s, d)
    def git(*a):
        r = subprocess.run(['git', '-C', R2, *a], capture_output=True, text=True)
        print((r.stdout + r.stderr).replace(tok, '***')[-800:])
        if r.returncode: raise RuntimeError('git ' + a[0])
    git('config', 'user.name', 'blond7I'); git('config', 'user.email', 'blond7I@users.noreply.github.com')
    git('add', '-A'); git('commit', '-q', '-m', msg + '\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n'
                                         'Claude-Session: https://claude.ai/code/session_01KWjYFaDqWdYY8y4uMMdgwF')
    git('push', f'https://x-access-token:{tok}@github.com/blond7I/almazov-mri-tumor.git', 'HEAD:main')
def push(files, msg):
    for k in range(3):
        try:
            _push(files, msg); return True
        except Exception as e:
            print('GitHub: попытка', k + 1, type(e).__name__, str(e)[:200]); time.sleep(20)
    return False
""")
code(r"""
# данные — как в ноутбуке обучения
DL = '/tmp/almazov_data'; os.makedirs(DL, exist_ok=True)
org = os.path.join(DL, 'Epic_and_CSCR_hospital_Dataset.zip')
if not (os.path.exists(org) and zipfile.is_zipfile(org)):
    import urllib.request; urllib.request.urlretrieve('https://box.almazovcentre.ru/index.php/s/ZD0MIbCoKEBMziJ/download', org)
fig = os.path.join(DL, 'figshare256.npz')
if not os.path.exists(fig):
    fig = data.build_figshare_npz(fig, DL + '/fig_raw', log=log)
df, A, ex_df, AX = data.load_org(REPO + '/data/split_v3.csv', org, log=log)
fX, fM, fy, fpid = data.load_figshare(fig)
S = data.build_sets(df, A, fX, fM, fy, fpid, log=log)
I = np.load(RES + '/index.npz'); assert np.array_equal(S['TE'], I['TE'])
LAB = S['LAB']
# внешние здоровые BRISC (копий в наших данных нет): половина — в обучение, половина — отложенная проверка;
# похожие снимки (корреляция >= 0.9, вероятно один пациент) держим в одной половине
m = pd.read_csv(REPO + '/results/stage2/brisc_map.csv')
ext = m[(m.our_part == 'external') & (m.label == NO_TUMOR)].reset_index(drop=True)
files = {Path(p).stem: p for p in glob.glob(BR + '/classification_task/*/*/*.jpg')}
Xn = np.stack([prep(np.array(Image.open(files[f]).convert('L'))) for f in ext.file])
v = np.stack([z1(a) for a in small(Xn)]); C = v @ v.T
par = list(range(len(Xn)))
def fnd(i):
    while par[i] != i: par[i] = par[par[i]]; i = par[i]
    return i
for i, j in zip(*np.where(np.triu(C, 1) >= 0.9)): par[fnd(i)] = fnd(j)
grp = np.array([fnd(i) for i in range(len(Xn))])
ug = np.unique(grp); rs = np.random.RandomState(0); rs.shuffle(ug)
half = set(ug[:len(ug) // 2])
ntr = np.array([g in half for g in grp]); nho = ~ntr
log(f'Внешние здоровые BRISC: {len(Xn)} (групп {len(ug)}); в обучение {ntr.sum()}, отложено {nho.sum()}')
pd.DataFrame(dict(file=ext.file, group=grp, part=np.where(ntr, 'train_c', 'holdout'))).to_csv(OUT + '/brisc_normals_split.csv', index=False)
""")
code(r"""
# обучение: тот же рецепт + искажения качества (как кандидаты B) + здоровые T1 BRISC; выбор эпохи — по нашей валидации
A2 = np.concatenate([A, Xn[ntr]]); L2 = np.concatenate([LAB, np.full(ntr.sum(), NO_TUMOR)])
tr2 = np.concatenate([S['TR'], len(A) + np.arange(ntr.sum())])
CK = '/tmp/ck'; os.makedirs(CK, exist_ok=True)
NEW = {}
for sd in (1, 2):
    tag = f'cls_n{sd}'
    mdl, info, _ = train.train_cls(A2, L2, tr2, S['VA'], DEV, CK, tag, 200 + sd, 15, bs=64, workers=4, log=log, strong_aug=True)
    NEW[tag] = dict(info={k: v for k, v in info.items() if k != 'history'},
                    lva=nets.cls_logits(mdl, A[S['VA']], DEV), lte=nets.cls_logits(mdl, A[S['TE']], DEV),
                    lho=nets.cls_logits(mdl, Xn[nho], DEV), lax=nets.cls_logits(mdl, AX, DEV))
    np.savez(OUT + f'/{tag}.npz', lva=NEW[tag]['lva'], lte=NEW[tag]['lte'], meta=json.dumps(NEW[tag]['info']))
    torch.save(nets.half_state(mdl.state_dict()), OUT + f'/w_{tag}.pt'); del mdl; torch.cuda.empty_cache()
# текущие модели (этап 1 + этап 2) — на тех же наборах
import json as _j
CUR = {}
W = Path(REPO) / 'weights'
man = _j.load(open(W / 'cls_models.json'))['models'][0]
f = nets.assemble([W / p for p in man['parts']], '/tmp/cls_stage1.pt', sha256=man['sha256'])
mods = [('cls_stage1', f)]
for mm in _j.load(open(W / 'stage2' / 'models.json'))['models']:
    if mm['kind'] == 'cls':
        mods.append((mm['tag'], nets.assemble([W / 'stage2' / p for p in mm['parts']], f'/tmp/{mm["tag"]}.pt', sha256=mm['sha256'])))
for tag, f in mods:
    mdl = nets.make_cls('resnet50'); mdl.load_state_dict(nets.load_state(f)); mdl.to(DEV).eval()
    CUR[tag] = dict(lva=nets.cls_logits(mdl, A[S['VA']], DEV), lte=nets.cls_logits(mdl, A[S['TE']], DEV),
                    lho=nets.cls_logits(mdl, Xn[nho], DEV), lax=nets.cls_logits(mdl, AX, DEV)); del mdl
""")
code(r"""
from sklearn.metrics import f1_score
def P(models, key):
    return np.mean([nets.softmax_tta(m[key]) for m in models], 0)
lab = pd.read_csv(REPO + '/data/sequence_plane_labels.csv')
comp2seq = {df.imgcomp.values[i]: s for i, s in zip(lab.idx, lab.seq)}
TE = S['TE']; yte = LAB[TE]; seq_te = np.array([comp2seq.get(df.imgcomp.values[i], '') for i in TE])
ex_meta = pd.read_csv(RES + '/extra_meta.csv', keep_default_na=False)
nw = ((ex_meta.status == 'новый') & (ex_meta.label.astype(int) >= 0)).values; yax = ex_meta.label.astype(int).values[nw]
def report(models):
    pv, pt, ph, pa = P(models, 'lva'), P(models, 'lte'), P(models, 'lho'), P(models, 'lax')[nw]
    r = dict(val_acc=float((pv.argmax(1) == LAB[S['VA']]).mean()), val_macro_f1=float(f1_score(LAB[S['VA']], pv.argmax(1), average='macro')),
             test_acc=float((pt.argmax(1) == yte).mean()), test_macro_f1=float(f1_score(yte, pt.argmax(1), average='macro')),
             test_tumor_sens=float((pt[yte != 3].argmax(1) != 3).mean()),
             brisc_holdout_false_alarm=float((ph.argmax(1) != 3).mean()), brisc_holdout_n=int(len(ph)),
             new_images_acc=float((pa.argmax(1) == yax).mean()), new_glioma_recall=float((pa[yax == 0].argmax(1) == 0).mean()))
    for s in ['T1', 'T2', 'FLAIR', 'КТ']:
        k = (yte == 3) & (seq_te == s)
        r[f'test_normals_{s}_false_alarm'] = f'{int((pt[k].argmax(1) != 3).sum())}/{int(k.sum())}'
    return r
cur = list(CUR.values()); new = list(NEW.values())
R = dict(note='классификатор без фильтра входа; ансамбль = среднее softmax с отражением; здоровые теста — все копии (не уникальные)',
         split=dict(external_normals=int(len(Xn)), train=int(ntr.sum()), holdout=int(nho.sum())),
         candidates={k: v['info'] for k, v in NEW.items()},
         current=report(cur), candidates_only=report(new), current_plus_candidates=report(cur + new))
log(json.dumps(R, ensure_ascii=False, indent=1))
json.dump(R, open(OUT + '/normals_experiment.json', 'w'), ensure_ascii=False, indent=1, default=float)
np.savez_compressed(OUT + '/extra_n.npz', **{f'logits_{k}': v['lax'] for k, v in NEW.items()})
push({'results/stage2/normals_experiment.json': OUT + '/normals_experiment.json', 'results/stage2/normals_log.txt': OUT + '/normals_log.txt'},
     'Experiment C: BRISC external T1 normals in classifier training (run on Kaggle)')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
