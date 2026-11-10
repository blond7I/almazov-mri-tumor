"""Собирает notebooks/stage2_brats.ipynb: проверка итоговой системы на чужих данных BraTS (T1c без черепа) —
полоса, контур и то, сколько таких срезов система сама отклоняет или помечает. Kaggle, GPU."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Этап 2 · чужие данные: BraTS (T1 с контрастом, срезы без черепа)
Вход: выход ноутбука обучения (stage2_results.zip, Add Input → Your Work). Секрет `GITHUB_TOKEN`. Internet: On. GPU T4.
Проверка вне области применения, записана в протоколе до запуска. Главный вопрос не «хорошо ли работает» (нет),
а **понимает ли система, что данные чужие**: сколько срезов она отклоняет, сколько помечает «снимок не похож на обучение»
и что говорит правило двух сигналов на срезах без опухоли. Итог — `results/stage2/extras.json` (ключ `brats_band`).
""")
code(r"""
import os, sys, subprocess, glob, zipfile, json, shutil, time
from pathlib import Path
REPO_URL, REPO = 'https://github.com/blond7I/almazov-mri-tumor', '/tmp/almazov_repo'
shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, REPO], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.31', 'segmentation-models-pytorch==0.6.0',
                'pydicom', 'nibabel', 'h5py'], check=True)
sys.path.insert(0, REPO + '/src')
import numpy as np, pandas as pd, torch
from almazov import pipeline, extras, data
INPUT = os.environ.get('ALMAZOV_INPUT', '/kaggle/input')
src = sorted(glob.glob(INPUT + '/**/stage2_results.zip', recursive=True))
assert src, 'Подключите выход ноутбука обучения (Add Input -> Your Work)'
RES, OUT = '/tmp/res', '/kaggle/working/brats_out'
shutil.rmtree(RES, ignore_errors=True); os.makedirs(RES)
with zipfile.ZipFile(src[0]) as z:
    for n in ['index.npz']:
        z.extract(n, RES)
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEV)
LOG = open(OUT + '/brats_log.txt', 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(time.strftime('%H:%M:%S ') + s + '\n'); LOG.flush()

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
from almazov import brats
R = {'models': M.version}
root = brats.download('/tmp/brats', log=log)
R['brats_band'], P, S = extras.brats_band(M, root, step=2, log=log)
P.to_csv(OUT + '/brats_band_patients.csv', index=False)
S.to_csv(OUT + '/brats_band_slices.csv', index=False)
import json as _j
old = _j.load(open(REPO + '/results/stage2/extras.json')) if os.path.exists(REPO + '/results/stage2/extras.json') else {}
old['brats_band'] = R['brats_band']; old['brats_band_models'] = M.version
_j.dump(old, open(OUT + '/extras.json', 'w'), ensure_ascii=False, indent=1, default=float)
push({'results/stage2/extras.json': OUT + '/extras.json', 'results/stage2/brats_band_patients.csv': OUT + '/brats_band_patients.csv',
      'results/stage2/brats_band_slices.csv': OUT + '/brats_band_slices.csv', 'results/stage2/brats_log.txt': OUT + '/brats_log.txt'},
     'BraTS check on the final ensemble: band, contour, and how often the system itself refuses or warns (run on Kaggle)')
print('Готово')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
