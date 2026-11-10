"""Собирает notebooks/stage2_extras.ipynb: дополнительные проверки прототипом + набор для врача, итог сразу в GitHub."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# Этап 2 · дополнительные проверки прототипом + набор для врача (Kaggle, GPU)
Вход: выход ноутбука обучения (Add Input → Your Work). Секрет `GITHUB_TOKEN` прикреплён (Add-ons → Secrets). Internet: On. GPU T4.
Что делает: (1) здоровые снимки теста по режимам МРТ; (2) 92 новых снимка — по классам + 20 ошибок для врача;
(3) набор случаев для сессии с врачом; (4) полоса v2 на BraTS (T1c). Итоги — сразу в репозиторий (`results/stage2/extras.json`, `doctor_kit/`).
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
RES, OUT = '/tmp/res', '/kaggle/working/extras_out'
shutil.rmtree(RES, ignore_errors=True); os.makedirs(RES)
with zipfile.ZipFile(src[0]) as z:
    for n in ['index.npz', 'meta.csv', 'extra_meta.csv', 'images_jpg.zip']:
        z.extract(n, RES)
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEV)
LOG = open(OUT + '/extras_log.txt', 'w', encoding='utf-8')
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
# данные: исходные снимки организаторов (архив с box) + исходные срезы Figshare 512x512 (.mat)
DL = '/tmp/almazov_data'; os.makedirs(DL, exist_ok=True)
BOX = 'https://box.almazovcentre.ru/index.php/s/ZD0MIbCoKEBMziJ/download'
org = data.find_file('*Epic*CSCR*.zip', [INPUT]) or (data.find_image_root(INPUT) if os.path.exists(INPUT) else None)
if not org:
    org = os.path.join(DL, 'Epic_and_CSCR_hospital_Dataset.zip')
    if not (os.path.exists(org) and zipfile.is_zipfile(org)):
        import urllib.request; log('Скачиваю архив организаторов...'); urllib.request.urlretrieve(BOX, org)
fig = data.build_figshare_npz(os.path.join(DL, 'figshare256.npz'), DL + '/fig_raw', log=log) if not os.path.exists(os.path.join(DL, 'figshare256.npz')) else os.path.join(DL, 'figshare256.npz')
F = np.load(fig); fX, fM, fidx = F['X'], F['M'], F['idx']
mats = {int(Path(f).stem): f for f in glob.glob(DL + '/fig_raw/mat/**/*.mat', recursive=True)}
I = np.load(RES + '/index.npz'); s_te = I['s_te']
meta = pd.read_csv(RES + '/meta.csv', keep_default_na=False)
# сверка порядка Figshare с обучением: те же срезы теста, что в images_jpg.zip (fte)
import cv2
with zipfile.ZipFile(RES + '/images_jpg.zip') as z:
    cs = []
    for i in s_te[::max(1, len(s_te) // 25)]:
        a = cv2.imdecode(np.frombuffer(z.read(f'fte/{int(i):05d}.jpg'), np.uint8), 0).astype(np.float32).ravel()
        b = fX[i].astype(np.float32).ravel()
        cs.append(np.corrcoef(a, b)[0, 1])
log('Сверка порядка Figshare (корреляция с картинками обучения): min', round(float(np.min(cs)), 4))
assert np.min(cs) > 0.98, 'Порядок срезов Figshare не совпадает с обучением'
""")
code(r"""
# исходные снимки организаторов, которые нужны: здоровые теста (разметка режима), тест для врача, новые снимки архива
lab = pd.read_csv(REPO + '/data/sequence_plane_labels.csv')
meta['key'] = meta.path.map(extras.df_key)
tst = meta[(meta.part == 'test') & ~meta.label_conflict.astype(str).eq('True')].drop_duplicates('imgcomp')
ex = pd.read_csv(RES + '/extra_meta.csv', keep_default_na=False)
need = set(meta.key[lab.idx.values]) | set(tst.key) | set(ex[ex.status == 'новый'].key)
t0 = time.time(); ORIG = extras.read_originals(org, need)
log(f'Исходных снимков прочитано: {len(ORIG)} из {len(need)} за {time.time() - t0:.0f} с')
M = pipeline.load_models(REPO, cache_dir='/tmp/almazov_cache', device=DEV)
log('Модели:', M.version)
R = dict(models=M.version, commit=subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
""")
code(r"""
# 1) здоровые снимки теста по режимам МРТ
orig_by_idx = {int(i): ORIG[meta.key[int(i)]] for i in lab.idx if meta.key[int(i)] in ORIG}
R['normals_by_sequence'] = extras.normals_by_sequence(M, REPO + '/data/sequence_plane_labels.csv', orig_by_idx, OUT + '/normals_by_sequence.csv', log=log)
# 2) новые снимки полного архива
R['new_images'] = extras.new_images(M, RES + '/extra_meta.csv', ORIG, OUT + '/doctor_kit/errors_new', log=log)
# 3) набор для врача
org_n = [(r.key, ORIG[r.key], 3) for r in tst[tst.label.astype(int) == 3].itertuples() if r.key in ORIG]
org_t = [(r.key, ORIG[r.key], int(r.label)) for r in tst[tst.label.astype(int) < 3].itertuples() if r.key in ORIG]
R['doctor_cases'] = extras.doctor_cases(M, mats, s_te, fidx, fM, org_n, org_t, OUT + '/doctor_kit', log=log)
if os.path.exists(REPO + '/doctor_kit/README.txt'): shutil.copy(REPO + '/doctor_kit/README.txt', OUT + '/doctor_kit/README.txt')
json.dump(R, open(OUT + '/extras.json', 'w'), ensure_ascii=False, indent=1, default=float)
push({'results/stage2/extras.json': OUT + '/extras.json', 'results/stage2/normals_by_sequence.csv': OUT + '/normals_by_sequence.csv',
      'results/stage2/extras_log.txt': OUT + '/extras_log.txt', 'doctor_kit': OUT + '/doctor_kit'},
     'Stage-2 extras: normals by sequence, new images per class, doctor kit (run on Kaggle)')
""")
code(r"""
# 4) полоса v2 на BraTS (T1 с контрастом) — по протоколу (docs/stage2_protocol.md, «Отклонения»)
from almazov import brats
root = brats.download('/tmp/brats', log=log)
R['brats_band'], P, S = extras.brats_band(M, root, step=2, log=log)
P.to_csv(OUT + '/brats_band_patients.csv', index=False)
S.to_csv(OUT + '/brats_band_slices.csv', index=False)
json.dump(R, open(OUT + '/extras.json', 'w'), ensure_ascii=False, indent=1, default=float)
push({'results/stage2/extras.json': OUT + '/extras.json', 'results/stage2/brats_band_patients.csv': OUT + '/brats_band_patients.csv',
      'results/stage2/brats_band_slices.csv': OUT + '/brats_band_slices.csv',
      'results/stage2/extras_log.txt': OUT + '/extras_log.txt'}, 'Stage-2 extras: uncertainty band v2 on BraTS T1c (run on Kaggle)')
print('Готово')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
