"""Собирает notebooks/stage2_brats.ipynb (Kaggle): модуль «компоненты глиальной опухоли» (S5)."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Этап 2 · компоненты глиальной опухоли: отёк, ядро, контрастируемая часть (команда InTeam)
Данные скачиваются автоматически: Medical Segmentation Decathlon, Task01_BrainTumour (BraTS 2016–2017, CC BY-SA 4.0), ~7 ГБ.
Две модели на срезах, разбиение по пациентам: **T1 с контрастом + FLAIR** и **только T1 с контрастом** — чтобы показать,
что перифокальный отёк по одному T1 с контрастом надёжно не оценить. Время на GPU T4: ~1–1.5 ч.

**Kaggle:** Session options → Accelerator **GPU T4 x2**, Internet **On** → Save Version → **Save & Run All (Commit)**.
Результат: вкладка Output → `stage2_brats.zip`.
""")
code(r"""
# === 1. Код и библиотеки ===
import os, sys, subprocess, shutil, zipfile, glob
REPO_URL, REPO = 'https://github.com/blond7I/almazov-mri-tumor', '/tmp/almazov_repo'
if os.path.exists(REPO): subprocess.run(['git', '-C', REPO, 'pull', '--ff-only', '-q'], check=False)
else: subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, REPO], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.31', 'segmentation-models-pytorch==0.6.0', 'nibabel'], check=True)
sys.path.insert(0, REPO + '/src')
import torch
assert torch.cuda.is_available(), 'Нет GPU: Session options -> Accelerator -> GPU T4 x2'
print('GPU:', torch.cuda.get_device_name(0))
WORK = '/kaggle/temp/msd' if os.path.isdir('/kaggle/temp') else '/tmp/msd'
OUT = '/kaggle/working/brats' if os.path.isdir('/kaggle/working') else os.path.abspath('brats_out')
""")
code(r"""
# === 2. Скачивание, обучение двух моделей, оценка на отложенных пациентах ===
from almazov import brats
R = brats.run(WORK, OUT)
""")
code(r"""
# === 3. Упаковка ===
dst = os.path.join(os.path.dirname(OUT), 'stage2_brats.zip')
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_STORED) as z:
    for p in sorted(glob.glob(OUT + '/*')): z.write(p, os.path.basename(p))
shutil.rmtree(OUT)
print('Готово:', dst, round(os.path.getsize(dst) / 1e6), 'МБ')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
