"""Собирает notebooks/stage2_push_weights.ipynb: веса этапа 2 из выхода Kaggle -> части по 20 МБ -> проверка -> push в GitHub.
Ничего не качается на ноутбук: Kaggle -> GitHub напрямую. Токен GitHub хранится только в Kaggle Secrets и нигде не печатается."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Этап 2 · веса моделей → GitHub (без скачивания на ноутбук)
1. **Токен GitHub** (github.com → Settings → Developer settings → Fine-grained tokens → Generate new token):
   срок 7 дней · Repository access: *Only select repositories* → `almazov-mri-tumor` · Permissions → Repository → **Contents: Read and write**.
2. **Kaggle:** Add-ons → Secrets → Add secret: имя `GITHUB_TOKEN`, значение — токен; галочка «прикреплён к этому блокноту».
3. **Вход:** Add Input → Your Work → ноутбук обучения (тот же, что для оценки). Settings → Internet: On. Видеокарта не нужна.
4. Run All (~5–10 мин). В конце — «Готово: … моделей на GitHub».
""")
code(r"""
import os, sys, subprocess, glob, zipfile, json, hashlib, shutil
from pathlib import Path
REPO_URL, REPO = 'https://github.com/blond7I/almazov-mri-tumor', '/tmp/almazov_repo'
shutil.rmtree(REPO, ignore_errors=True)
subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, REPO], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.31', 'segmentation-models-pytorch==0.6.0'], check=True)
INPUT = os.environ.get('ALMAZOV_INPUT', '/kaggle/input')
items = {}                                                   # имя w_*.pt -> байты
for zp in sorted(glob.glob(INPUT + '/**/stage2_weights.zip', recursive=True)):
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            b = os.path.basename(n)
            if b.startswith('w_') and b.endswith('.pt'): items.setdefault(b, z.read(n))
for p in sorted(glob.glob(INPUT + '/**/w_*.pt', recursive=True)):
    items.setdefault(os.path.basename(p), Path(p).read_bytes())
assert items, 'Не найдены веса: подключите выход ноутбука обучения (Add Input -> Your Work)'
print({k: round(len(v) / 1e6, 1) for k, v in sorted(items.items())}, 'МБ')
""")
code(r"""
# части по 20 МБ + models.json с контрольными суммами (calib.json и input_models.npz остаются как есть)
CH = 20 * 1024 * 1024
dst = Path(REPO) / 'weights' / 'stage2'
dst.mkdir(parents=True, exist_ok=True)
for old in dst.glob('*.pt.part*'): old.unlink()
models = []
for name, data in sorted(items.items()):
    tag = name[2:-3]                                         # w_seg_s1.pt -> seg_s1
    parts = []
    for i in range(0, len(data), CH):
        pn = f'{tag}.pt.part{i // CH}'
        (dst / pn).write_bytes(data[i:i + CH]); parts.append(pn)
    models.append(dict(tag=tag, kind='seg' if tag.startswith('seg') else 'cls', parts=parts,
                       sha256=hashlib.sha256(data).hexdigest(), mb=round(len(data) / 1e6, 1)))
json.dump(dict(models=models, note='Новые обучения этапа 2 (тот же рецепт, другие seed), float16'),
          open(dst / 'models.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(models, ensure_ascii=False, indent=1))
""")
code(r"""
# проверка до отправки: прототип собирает ансамбль из частей и анализирует снимок
import numpy as np, cv2
sys.path.insert(0, REPO + '/src')
from almazov import pipeline
M = pipeline.load_models(REPO, cache_dir='/tmp/almazov_cache')
print(M.version)
n_seg, n_cls = 1 + sum(m['kind'] == 'seg' for m in models), 1 + sum(m['kind'] == 'cls' for m in models)
assert len(M.seg) == n_seg and len(M.cls) == n_cls, (len(M.seg), len(M.cls))
img = np.zeros((512, 512), np.uint8); cv2.ellipse(img, (256, 256), (200, 230), 0, 0, 360, 110, -1)
res = pipeline.analyze(M, img)
print('analyze ok:', sorted(res)[:12])
""")
code(r"""
# отправка в GitHub; токен берётся из Kaggle Secrets и вычищается из любого вывода
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret('GITHUB_TOKEN').strip()
def git(*a):
    r = subprocess.run(['git', '-C', REPO, *a], capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(tok, '***')[-1500:])
    if r.returncode: raise RuntimeError('git ' + a[0] + ': ошибка (см. вывод выше)')
git('config', 'user.name', 'blond7I'); git('config', 'user.email', 'blond7I@users.noreply.github.com')
git('add', 'weights/stage2')
git('commit', '-q', '-m', 'Stage-2 ensemble weights (float16, 20MB parts, sha256 in models.json)\n\n'
    'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n'
    'Claude-Session: https://claude.ai/code/session_01KWjYFaDqWdYY8y4uMMdgwF')
git('push', f'https://x-access-token:{tok}@github.com/blond7I/almazov-mri-tumor.git', 'HEAD:main')
del tok
print('Готово:', len(models), 'моделей на GitHub ->', REPO_URL + '/tree/main/weights/stage2')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
