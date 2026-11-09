"""Собирает notebooks/stage2_explain.ipynb: на что опирается классификатор (Grad-CAM, стирание опухоли) на тесте с экспертными масками."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Этап 2 · на что опирается классификатор (Kaggle)
Вход: выход ноутбука обучения (stage2_results.zip, Add Input → Your Work). Секрет `GITHUB_TOKEN`. Internet: On. GPU — по возможности (на CPU ~40 мин).
На **тестовых срезах Figshare с экспертными масками** (модели их не видели):
1. **Grad-CAM ансамбля** для итогового класса: попадает ли максимум «внимания» в опухоль (pointing game, допуск 16 пикс. сетки 256 —
   половина клетки карты 8×8) и во сколько раз доля внимания в опухоли больше её доли площади головы (обогащение).
   Базовый уровень — случайная точка в голове.
2. **Стирание опухоли**: экспертную маску (+5 пикс.) закрашиваем по окружению и заново считаем классификатор; **контроль** — та же
   форма, отражённая на другую половину. Если после стирания опухоли классификатор по-прежнему уверен — ответ опирается на контекст
   (уровень среза, окружение, масс-эффект), а не только на опухоль.
Правила заданы до запуска (docs/stage2_protocol.md, «Проверка: на что опирается классификатор»). Итог — `results/stage2/explain_check.json`.
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
from almazov import pipeline, explain, data
INPUT = os.environ.get('ALMAZOV_INPUT', '/kaggle/input')
src = sorted(glob.glob(INPUT + '/**/stage2_results.zip', recursive=True))
assert src, 'Подключите выход ноутбука обучения (Add Input -> Your Work)'
RES, OUT = '/tmp/res', '/kaggle/working/explain_out'
shutil.rmtree(RES, ignore_errors=True); os.makedirs(RES)
with zipfile.ZipFile(src[0]) as z:
    for n in ['index.npz', 'images_jpg.zip']:
        z.extract(n, RES)
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEV)
LOG = open(OUT + '/explain_log.txt', 'w', encoding='utf-8')
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
DL = '/tmp/almazov_data'; os.makedirs(DL, exist_ok=True)
fig = os.path.join(DL, 'figshare256.npz')
if not os.path.exists(fig):
    data.build_figshare_npz(fig, DL + '/fig_raw', log=log)
F = np.load(fig); fX, fM, fy = F['X'], F['M'], F['y']
s_te = np.load(RES + '/index.npz')['s_te']
import cv2
with zipfile.ZipFile(RES + '/images_jpg.zip') as z:
    cs = [np.corrcoef(cv2.imdecode(np.frombuffer(z.read(f'fte/{int(i):05d}.jpg'), np.uint8), 0).astype(np.float32).ravel(),
                      fX[i].astype(np.float32).ravel())[0, 1] for i in s_te[::max(1, len(s_te) // 25)]]
log('Сверка порядка Figshare: min corr', round(float(np.min(cs)), 4)); assert np.min(cs) > 0.98
M = pipeline.load_models(REPO, cache_dir='/tmp/almazov_cache', device=DEV)
log('Модели:', M.version, '| тестовых срезов Figshare:', len(s_te))
""")
code(r"""
from almazov.common import CLASSES, NO_TUMOR
from almazov.extras import wilson
dev = torch.device(DEV)
rows = []
t0 = time.time()
B = 16
for s in range(0, len(s_te), B):
    idx = s_te[s:s + B]
    X = fX[idx]; Mk = (fM[idx] > 0).astype(np.uint8)
    H = np.stack([pipeline.head_mask(x) for x in X])
    P0 = explain.probs(M, X, dev)
    t = P0.argmax(1)
    C = explain.gradcam(M, X, dev, t, head=H)
    XE, XC, CTL = [], [], []
    for x, m, h in zip(X, Mk, H):
        XE.append(explain.erase(x, m)[0])
        c = explain.mirror_control(m, h); CTL.append(int(c.sum()))
        XC.append(explain.erase(x, c)[0] if c.any() else x)
    P1 = explain.probs(M, np.stack(XE), dev); P2 = explain.probs(M, np.stack(XC), dev)
    for k, i in enumerate(idx):
        m, h, cam = Mk[k], H[k], C[k]
        md = cv2.dilate(m, np.ones((33, 33), np.uint8))           # допуск 16 пикселей
        y0, x0 = np.unravel_index(int(np.argmax(cam)), cam.shape)
        head_area = max(int(h.sum()), 1)
        rows.append(dict(i=int(i), y=int(fy[i]), pred=int(t[k]), p_top=float(P0[k, t[k]]), p_tumor=1 - float(P0[k, NO_TUMOR]),
                         mask_frac=float((m & h).sum()) / head_area,
                         hit=bool(md[y0, x0]), hit_base=float((md & h).sum()) / head_area,
                         share=float(cam[m > 0].sum() / (cam.sum() + 1e-8)),
                         p_top_erased=float(P1[k, t[k]]), p_tumor_erased=1 - float(P1[k, NO_TUMOR]), pred_erased=int(P1[k].argmax()),
                         p_top_ctrl=float(P2[k, t[k]]), p_tumor_ctrl=1 - float(P2[k, NO_TUMOR]), ctrl_px=CTL[k]))
    if s % (B * 8) == 0:
        log(f'{s + len(idx)}/{len(s_te)} срезов, {time.time() - t0:.0f} с')
D = pd.DataFrame(rows); D.to_csv(OUT + '/explain_rows.csv', index=False)
log('Готово за', round(time.time() - t0), 'с')
""")
code(r"""
def summ(d):
    n = len(d); h = int(d.hit.sum())
    ok = d[d.ctrl_px > 0]
    e = d.share / d.mask_frac.clip(lower=1e-4)
    return dict(n=n,
        pointing_hit=round(h / n, 3), pointing_hit_ci=[round(v, 3) for v in wilson(h, n)], pointing_random=round(float(d.hit_base.mean()), 3),
        attention_share_in_tumor_median=round(float(d.share.median()), 3), tumor_area_share_median=round(float(d.mask_frac.median()), 3),
        enrichment_median=round(float(e.median()), 2),
        tumor_still_detected_after_erase=round(float((d.p_tumor_erased >= 0.5).mean()), 3),
        same_type_after_erase=round(float((d.pred_erased == d.pred).mean()), 3),
        p_top_drop_mean=round(float((d.p_top - d.p_top_erased).mean()), 3),
        p_top_drop_ctrl_mean=round(float((ok.p_top - ok.p_top_ctrl).mean()), 3) if len(ok) else None,
        share_big_drop=round(float(((d.p_top - d.p_top_erased) >= 0.15).mean()), 3))
S = dict(models=M.version, note='тест Figshare, экспертные маски; Grad-CAM ансамбля для итогового класса; стирание = inpainting Telea маски +5 пикс.',
         all=summ(D), by_class={CLASSES[c]: summ(D[D.y == c]) for c in sorted(D.y.unique())},
         correct_only=summ(D[D.pred == D.y]))
json.dump(S, open(OUT + '/explain_check.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False, indent=1))
""")
code(r"""
# иллюстрация для слайда: 3 класса × (снимок с маской, Grad-CAM, после стирания)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pick = []
for c in range(3):
    d = D[(D.y == c) & (D.pred == c)].sort_values('p_top', ascending=False)
    pick += list(d.i.values[:1])
fig, ax = plt.subplots(len(pick), 3, figsize=(9, 3 * len(pick)), dpi=110)
for r, i in enumerate(pick):
    x = fX[i]; m = (fM[i] > 0).astype(np.uint8); row = D[D.i == i].iloc[0]
    cam = explain.gradcam(M, x[None], dev, int(row.pred), head=pipeline.head_mask(x)[None])[0]
    xe = explain.erase(x, m)[0]
    ax[r, 0].imshow(x, cmap='gray'); ax[r, 0].contour(m, [0.5], colors='r', linewidths=1)
    ax[r, 0].set_title(f'{CLASSES[int(row.y)]}: эксперт (красное)', fontsize=9)
    ax[r, 1].imshow(x, cmap='gray'); ax[r, 1].imshow(cam, cmap='inferno', alpha=0.45); ax[r, 1].contour(m, [0.5], colors='c', linewidths=1)
    ax[r, 1].set_title('куда смотрел классификатор', fontsize=9)
    ax[r, 2].imshow(xe, cmap='gray')
    ax[r, 2].set_title(f'опухоль стёрта: «{CLASSES[int(row.pred)]}» {row.p_top:.0%} → {row.p_top_erased:.0%}', fontsize=9)
    for a in ax[r]: a.axis('off')
plt.tight_layout(); plt.savefig(OUT + '/explain_examples.png'); plt.show()
push({'results/stage2/explain_check.json': OUT + '/explain_check.json', 'results/stage2/explain_rows.csv': OUT + '/explain_rows.csv',
      'results/stage2/explain_examples.png': OUT + '/explain_examples.png', 'results/stage2/explain_log.txt': OUT + '/explain_log.txt'},
     'Stage-2 check: what the classifier relies on (Grad-CAM pointing game, tumor erasure with mirrored control) on Figshare test')
print('Готово')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
