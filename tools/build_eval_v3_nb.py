"""Собирает notebooks/stage2_eval_v3.ipynb: правила принятия B и C (записаны до запусков) -> итоговый ансамбль ->
полная оценка по протоколу -> веса и калибровка в GitHub. Kaggle, без видеокарты."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Этап 2 · итоговый ансамбль: правила принятия B и C → оценка v3 → веса в GitHub
**Входы (Add Input → Your Work):** ноутбук обучения (stage2_results.zip), ноутбук BRISC (stage2b_results.zip, stage2b_weights.zip),
ноутбук эксперимента C (cls_n*.npz, extra_n.npz, w_cls_n*.pt). Секрет `GITHUB_TOKEN`. Internet: On. Видеокарта не нужна.
**Правила (docs/stage2_protocol.md, записаны до запусков):**
- B, сегментация: Dice ансамбля на валидации Figshare растёт ≥ 0.003 и на тесте не падает больше чем на 0.005;
- B, классификация: macro-F1 валидации не падает, точность теста не падает больше чем на 0.005;
- C: решение уже принято по его правилу (results/stage2/normals_experiment.json) — здесь только пересчёт тех же чисел.
Итоговый ансамбль = текущий + принятые кандидаты; на нём — полная оценка (evaluate.run) и новая калибровка.
""")
code(r"""
import os, sys, subprocess, glob, zipfile, shutil, json, time, hashlib
from pathlib import Path
REPO_URL, REPO = 'https://github.com/blond7I/almazov-mri-tumor', '/tmp/almazov_repo'
shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, REPO], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.31', 'segmentation-models-pytorch==0.6.0'], check=True)
sys.path.insert(0, REPO + '/src')
RES, OUT = '/tmp/res', '/kaggle/working/v3'
shutil.rmtree(RES, ignore_errors=True); os.makedirs(RES); os.makedirs(OUT, exist_ok=True)
IN = '/kaggle/input'
def one(pat):
    f = sorted(glob.glob(IN + '/**/' + pat, recursive=True)); return f[0] if f else None
src = one('stage2_results.zip'); assert src, 'нет выхода ноутбука обучения'
zipfile.ZipFile(src).extractall(RES)
srcb = one('stage2b_results.zip'); assert srcb, 'нет выхода ноутбука BRISC (B)'
zipfile.ZipFile(srcb).extractall(RES)
for pat in ['cls_n1.npz', 'cls_n2.npz', 'extra_n.npz']:
    f = one(pat); assert f, 'нет выхода эксперимента C: ' + pat
    shutil.copy(f, RES)
W = {}                                                     # веса кандидатов: имя -> путь
wb = one('stage2b_weights.zip')
if wb:
    with zipfile.ZipFile(wb) as z:
        for n in z.namelist():
            if os.path.basename(n).startswith('w_'):
                z.extract(n, '/tmp/wb'); W[os.path.basename(n)] = '/tmp/wb/' + n
for f in glob.glob(IN + '/**/w_*.pt', recursive=True):
    W.setdefault(os.path.basename(f), f)
print('Веса кандидатов:', sorted(W))
print(sorted(os.listdir(RES)))
LOG = open(OUT + '/v3_log.txt', 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()
""")
code(r"""
def _push(files, msg):
    # files: {путь в репозитории: локальный путь (файл или папка)}; свежий клон -> копия -> commit -> push
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
# решения по правилам
import numpy as np
from almazov import evaluate as ev
from almazov.common import clean_mask, dice, NO_TUMOR
from sklearn.metrics import f1_score
E = np.load(RES + '/eval_data.npz', allow_pickle=True)
CUR_SEG = ['seg_stage1', 'seg_s1', 'seg_s2', 'seg_s3']; CUR_CLS = ['cls_stage1', 'cls_s1', 'cls_s2']
B_SEG = sorted(os.path.basename(p)[:-4] for p in glob.glob(RES + '/seg_b*.npz'))
B_CLS = sorted(os.path.basename(p)[:-4] for p in glob.glob(RES + '/cls_b*.npz'))
N_CLS = ['cls_n1', 'cls_n2']
def seg_dice(members, key, gtkey):
    S = {k: np.load(RES + f'/{k}.npz') for k in members}
    pm, _, _ = ev.stack_maps(S, members, key)
    g = E[gtkey] > 0
    return float(dice(np.stack([clean_mask(x) for x in pm]) > 0, g).mean())
def cls_scores(members):
    L = {k: np.load(RES + f'/{k}.npz') for k in members}
    pv = ev.ens_prob([L[k]['lva'] for k in members]); pt = ev.ens_prob([L[k]['lte'] for k in members])
    yv, yt = E['y_ova'], E['y_ote']
    return dict(val_macro_f1=float(f1_score(yv, pv.argmax(1), average='macro')), test_acc=float((pt.argmax(1) == yt).mean()),
                test_tumor_sens=float((pt[yt != NO_TUMOR].argmax(1) != NO_TUMOR).mean()))
D = dict(rules='docs/stage2_protocol.md', seg={}, cls={})
for name, mem in [('current', CUR_SEG), ('current+B', CUR_SEG + B_SEG)]:
    D['seg'][name] = dict(members=mem, val_dice=seg_dice(mem, 'fva', 'm_fva'), test_dice=seg_dice(mem, 'fte', 'm_fte'))
for name, mem in [('current', CUR_CLS), ('current+B', CUR_CLS + B_CLS), ('current+C', CUR_CLS + N_CLS), ('current+B+C', CUR_CLS + B_CLS + N_CLS)]:
    D['cls'][name] = dict(members=mem, **cls_scores(mem))
s0, s1 = D['seg']['current'], D['seg']['current+B']
adopt_seg_b = (s1['val_dice'] - s0['val_dice'] >= 0.003) and (s1['test_dice'] - s0['test_dice'] >= -0.005)
c0, cb, cn = D['cls']['current'], D['cls']['current+B'], D['cls']['current+C']
adopt_cls_b = (cb['val_macro_f1'] >= c0['val_macro_f1']) and (cb['test_acc'] - c0['test_acc'] >= -0.005)
nx = json.load(open(REPO + '/results/stage2/normals_experiment.json'))
cur, cpc = nx['current'], nx['current_plus_candidates']
adopt_cls_n = (cpc['val_macro_f1'] >= cur['val_macro_f1'] - 0.005 and cpc['test_acc'] >= cur['test_acc'] - 0.005 and
               cpc['test_tumor_sens'] >= cur['test_tumor_sens'] - 0.005 and
               int(cpc['test_normals_T1_false_alarm'].split('/')[0]) <= int(cur['test_normals_T1_false_alarm'].split('/')[0]))
D['decision'] = dict(seg_b=adopt_seg_b, cls_b=adopt_cls_b, cls_n=adopt_cls_n,
                     note='C решён по его правилу на normals_experiment.json (вариант «текущие + кандидаты»); здесь он только применяется')
FINAL_SEG = CUR_SEG + (B_SEG if adopt_seg_b else [])
FINAL_CLS = CUR_CLS + (B_CLS if adopt_cls_b else []) + (N_CLS if adopt_cls_n else [])
D['final'] = dict(seg=FINAL_SEG, cls=FINAL_CLS)
log(json.dumps(D, ensure_ascii=False, indent=1))
json.dump(D, open(OUT + '/v3_decision.json', 'w'), ensure_ascii=False, indent=1)
""")
code(r"""
# полная оценка по протоколу на итоговом ансамбле (+ калибровка под него)
extra = [m for m in FINAL_SEG + FINAL_CLS if m not in CUR_SEG + CUR_CLS]
os.environ['ALMAZOV_EXTRA_MEMBERS'] = ','.join(extra)
# члены, не вошедшие в итог, убираем из папки, чтобы load их не подхватил
for p in glob.glob(RES + '/seg_b*.npz') + glob.glob(RES + '/cls_b*.npz') + glob.glob(RES + '/cls_n*.npz'):
    if os.path.basename(p)[:-4] not in extra: os.remove(p)
from almazov import figures
t0 = time.time()
R, C = ev.run(RES, OUT, OUT + '/calib.json')
log('evaluate.run:', round(time.time() - t0), 'с; члены', R['members'])
log(json.dumps({k: R[k] for k in ['seg_before_after', 'classifier', 'complete_miss_org_test', 'extra_new'] if k in R}, ensure_ascii=False, default=float)[:4000])
try:
    figures.make_all(RES, OUT)
except Exception as e:
    log('figures: ошибка', type(e).__name__, e)
""")
code(r"""
# упаковка весов итогового ансамбля -> weights/stage2 (+ calib.json), проверка прототипа, отправка
CH = 20 * 1024 * 1024
dst = Path(REPO) / 'weights' / 'stage2'
man = json.load(open(dst / 'models.json'))
have = {m['tag'] for m in man['models']}
for tag in extra:
    f = W.get(f'w_{tag}.pt'); assert f, 'нет весов ' + tag
    data = Path(f).read_bytes(); parts = []
    for i in range(0, len(data), CH):
        pn = f'{tag}.pt.part{i // CH}'; (dst / pn).write_bytes(data[i:i + CH]); parts.append(pn)
    if tag not in have:
        man['models'].append(dict(tag=tag, kind='seg' if tag.startswith('seg') else 'cls', parts=parts,
                                  sha256=hashlib.sha256(data).hexdigest(), mb=round(len(data) / 1e6, 1)))
man['note'] = 'Этап 2: новые обучения (seed) + принятые по правилам кандидаты B (BRISC + искажения) и C (здоровые T1 BRISC); float16'
json.dump(man, open(dst / 'models.json', 'w'), ensure_ascii=False, indent=1)
cal = json.load(open(OUT + '/calib.json')); cal['fitted_on'] = 'stage2 v3: ' + ', '.join(FINAL_SEG + FINAL_CLS)
json.dump(cal, open(dst / 'calib.json', 'w'), ensure_ascii=False, indent=1)
import numpy as np, cv2
from almazov import pipeline
M = pipeline.load_models(REPO, cache_dir='/tmp/almazov_cache')
log('Прототип:', M.version)
assert len(M.seg) == len(FINAL_SEG) and len(M.cls) == len(FINAL_CLS), (len(M.seg), len(M.cls))
img = np.zeros((512, 512), np.uint8); cv2.ellipse(img, (256, 256), (200, 230), 0, 0, 360, 110, -1)
r = pipeline.analyze(M, img); log('analyze ok, level =', r.get('level'))
files = {'results/stage2/stage2_eval_v3.json': OUT + '/stage2_eval.json', 'results/stage2/v3_decision.json': OUT + '/v3_decision.json',
         'results/stage2/v3_log.txt': OUT + '/v3_log.txt', 'weights/stage2/models.json': str(dst / 'models.json'),
         'weights/stage2/calib.json': str(dst / 'calib.json')}
for m in man['models']:
    if m['tag'] in extra:
        for pn in m['parts']: files['weights/stage2/' + pn] = str(dst / pn)
files = {k: v for k, v in files.items() if os.path.exists(v)}
LOG.flush()
push(files, 'Stage-2 v3: pre-registered rules applied to candidates B and C, final ensemble evaluated, weights and calibration updated (run on Kaggle)')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
