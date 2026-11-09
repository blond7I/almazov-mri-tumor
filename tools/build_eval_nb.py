"""Собирает notebooks/stage2_eval.ipynb: оценка по протоколу прямо на Kaggle, к себе скачивается только итог (~15 МБ)."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Этап 2 · оценка по протоколу (docs/stage2_protocol.md) — на Kaggle, без видеокарты
Вход: выход ноутбука обучения (Add Input → Your Work → ноутбук обучения). Выход: `stage2_eval.zip` (метрики, калибровка,
графики, галерея, страница разметки режима/плоскости) — небольшой архив.
""")
code(r"""
import os, sys, subprocess, glob, zipfile, shutil, json
REPO_URL, REPO = 'https://github.com/blond7I/almazov-mri-tumor', '/tmp/almazov_repo'
if os.path.exists(REPO): subprocess.run(['git', '-C', REPO, 'pull', '--ff-only', '-q'], check=False)
else: subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, REPO], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.31', 'segmentation-models-pytorch==0.6.0'], check=True)
sys.path.insert(0, REPO + '/src')
src = sorted(glob.glob('/kaggle/input/**/stage2_results.zip', recursive=True))
assert src, 'Подключите выход ноутбука обучения: Add Input -> Your Work -> notebook80e1793962'
RES, OUT = '/tmp/res', '/kaggle/working/eval'
shutil.rmtree(RES, ignore_errors=True); os.makedirs(RES); zipfile.ZipFile(src[0]).extractall(RES)
# кандидаты из ноутбука BRISC (если подключён его выход): их карты и логиты в том же формате -> оценка «текущий + кандидаты»
srcb = sorted(glob.glob('/kaggle/input/**/stage2b_results.zip', recursive=True))
if srcb:
    zipfile.ZipFile(srcb[0]).extractall(RES)
    os.environ['ALMAZOV_EXTRA_MEMBERS'] = ','.join(sorted(os.path.basename(p)[:-4] for p in glob.glob(RES + '/seg_b*.npz') + glob.glob(RES + '/cls_b*.npz')))
    print('Кандидаты:', os.environ['ALMAZOV_EXTRA_MEMBERS'])
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.dirname(src[0]) + '/summary.json') + glob.glob(os.path.dirname(src[0]) + '/log.txt'):
    shutil.copy(f, OUT)
print(sorted(os.listdir(RES)))
""")
code(r"""
from almazov import evaluate, figures
R, C = evaluate.run(RES, OUT, OUT + '/calib.json')
print(json.dumps({k: R[k] for k in ['crc', 'band_vs_ring', 'seg_before_after', 'risk_coverage', 'error_detection_auroc'] if k in R}, ensure_ascii=False, indent=1, default=float)[:5000])
print(figures.make_all(RES, OUT))
""")
code(r"""
# страница разметки режима МРТ и плоскости + всё в один архив
subprocess.run([sys.executable, REPO + '/tools/build_label_page.py', RES + '/images_jpg.zip', RES + '/label_pool.npz', OUT + '/label_page.html'], check=True)
for f in ['meta.csv', 'extra_meta.csv', 'label_pool.npz', 'index.npz']:
    if os.path.exists(os.path.join(RES, f)): shutil.copy(os.path.join(RES, f), OUT)
dst = '/kaggle/working/stage2_eval.zip'
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(glob.glob(OUT + '/*')): z.write(p, os.path.basename(p))
shutil.rmtree(OUT)
print('Готово:', dst, round(os.path.getsize(dst) / 1e6, 1), 'МБ')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
