"""Собирает notebooks/stage2_normals_split.ipynb: какие внешние здоровые снимки BRISC ушли в обучение (эксперимент C),
а какие отложены. Тот же код и seed, что в stage2_normals.ipynb; сверка по числам (518 / 629). Kaggle, без видеокарты."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Этап 2 · эксперимент C: список отложенных здоровых снимков BRISC
Вход: датасет briscdataset/brisc2025. Секрет GITHUB_TOKEN. Internet: On. Видеокарта не нужна.
Зачем: 2 из 5 классификаторов итогового ансамбля обучены на половине внешних здоровых снимков BRISC, поэтому проверять итоговую систему
можно только на отложенной половине. Здесь тем же кодом и seed восстанавливается это деление и сохраняется в репозиторий.
""")
code(r"""
import os, sys, subprocess, glob, json, shutil, time
from pathlib import Path
REPO_URL, REPO = 'https://github.com/blond7I/almazov-mri-tumor', '/tmp/almazov_repo'
shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, REPO], check=True)
sys.path.insert(0, REPO + '/src')
import numpy as np, pandas as pd
from PIL import Image
from almazov import brisc
from almazov.common import prep, small, z1, NO_TUMOR
BR = brisc.find_root('/kaggle/input'); assert BR
OUT = '/kaggle/working/split'; os.makedirs(OUT, exist_ok=True)

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
print('в обучение', int(ntr.sum()), 'отложено', int(nho.sum()), 'групп', len(ug))
want = json.load(open(REPO + '/results/stage2/normals_experiment.json'))['split']
assert int(ntr.sum()) == want['train'] and int(nho.sum()) == want['holdout'], ('деление не совпало с экспериментом C', want)
pd.DataFrame(dict(file=ext.file, group=grp, part=np.where(ntr, 'train_c', 'holdout'))).to_csv(OUT + '/brisc_normals_split.csv', index=False)
push({'results/stage2/brisc_normals_split.csv': OUT + '/brisc_normals_split.csv'},
     'Experiment C: which BRISC external normals went to training and which are held out (same code and seed; counts verified)')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
