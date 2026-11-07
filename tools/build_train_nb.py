"""Собирает notebooks/stage2_train.ipynb (Kaggle/Colab). Вся логика — в src/almazov, ноутбук только вызывает её."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Этап 2 · обучение и выгрузка для оценки неуверенности (команда InTeam)
Что делает: 3 новых обучения сегментации (U-Net ResNet34) и 2 новых обучения классификатора (ResNet-50) **тем же рецептом и на том же
разбиении**, что на этапе 1; сверяет модели этапа 1 тем же кодом; выгружает карты вероятностей, логиты, Grad-CAM и признаки для
оценки неуверенности, калибровки и проверки. Сначала идёт короткая проверка (SMOKE, ~3 мин), потом полный запуск (~1.5–3 ч на GPU).

**Kaggle (основной вариант, можно закрыть браузер):** справа *Session options* → Accelerator **GPU T4 x2**, Internet **On**
→ сверху **Save Version** → **Save & Run All (Commit)** → Save. Через 2–3 часа: страница ноутбука → **Output** →
скачать `stage2_results.zip` и `stage2_weights.zip`.\nДанные ноутбук скачивает сам (полный архив организаторов + Figshare с первоисточника), ничего подключать не нужно.

**Colab (запасной):** Среда выполнения → Сменить среду → **T4 GPU** → **Выполнить все**, разрешить доступ к Google Drive, вкладку не закрывать.
Если Colab оборвался — снова «Выполнить все»: готовые модели пропустятся, начатая продолжится с последней эпохи. Результаты — в Drive, папка `almazov/stage2`.
""")

code(r"""
# === 1. Код из репозитория, библиотеки, GPU ===
import os, sys, subprocess, glob, shutil, json, time, zipfile
KAGGLE, COLAB = os.path.exists('/kaggle/input'), 'google.colab' in sys.modules
REPO_URL, REPO = 'https://github.com/blond7I/almazov-mri-tumor', '/tmp/almazov_repo'
if os.path.exists(REPO):
    subprocess.run(['git', '-C', REPO, 'pull', '--ff-only', '-q'], check=False)
else:
    subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO_URL, REPO], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.31', 'segmentation-models-pytorch==0.6.0', 'gdown', 'h5py'], check=True)
sys.path.insert(0, REPO + '/src')
import torch
assert torch.cuda.is_available(), 'Нет GPU. Kaggle: Session options -> Accelerator -> GPU T4 x2. Colab: Среда выполнения -> T4 GPU.'
print('GPU:', torch.cuda.get_device_name(0), '| код:', subprocess.run(['git', '-C', REPO, 'log', '-1', '--format=%h %s'], capture_output=True, text=True).stdout.strip())
if COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
OUT = '/kaggle/working/stage2' if KAGGLE else ('/content/drive/MyDrive/almazov/stage2' if COLAB else os.path.abspath('stage2_out'))
SMOKE_OUT = '/tmp/stage2_smoke'
print('Результаты будут в:', OUT)
""")

code(r"""
# === 2. Данные: полный архив организаторов (12 064 снимка) + Figshare. Ищем подключённые, иначе скачиваем ===
from almazov import data
BOX = 'https://box.almazovcentre.ru/index.php/s/ZD0MIbCoKEBMziJ/download'     # ссылка организаторов (ноябрь 2026)
GDRIVE = {'Almazov_MRI.zip': '1xv10QLA6Y8JzUZ_dpMurw93w-MZyzetA', 'figshare256.npz': '1MDAF0FXIeg89_w-G9-nKBv46tmzpY77i'}
DL = '/tmp/almazov_data'; os.makedirs(DL, exist_ok=True)
ROOTS = ['/kaggle/input'] if KAGGLE else (['/content/drive/MyDrive'] if COLAB else ['.'])
def is_zip(p): return p and os.path.exists(p) and zipfile.is_zipfile(p)
def from_box():
    dst = os.path.join(DL, 'Epic_and_CSCR_hospital_Dataset.zip')
    if not is_zip(dst):
        print('Скачиваю полный архив организаторов (241 МБ)...')
        try:
            import urllib.request
            urllib.request.urlretrieve(BOX, dst)
        except Exception as e:
            print('  не получилось:', type(e).__name__, e)
    return dst if is_zip(dst) else None
def from_gdrive(name):
    dst = os.path.join(DL, name)
    if not os.path.exists(dst):
        print('Скачиваю', name, 'по ссылке Google Drive...')
        try:
            import gdown
            gdown.download(id=GDRIVE[name], output=dst, quiet=True)
        except Exception as e:
            print('  не получилось:', type(e).__name__, e)
    return dst if os.path.exists(dst) and os.path.getsize(dst) > 1e6 else None
org = (data.find_file('*Epic*CSCR*.zip', ROOTS) or (data.find_image_root('/kaggle/input') if KAGGLE else None) or from_box()
       or data.find_file('Almazov_MRI*.zip', ROOTS) or from_gdrive('Almazov_MRI.zip'))
assert org, ('Нет данных организаторов. Kaggle: справа Add Input -> загрузите архив организаторов; '
             'или включите Internet (Session options), чтобы ноутбук скачал его сам.')
fig = data.find_file('figshare256.npz', ROOTS) or from_gdrive('figshare256.npz') or data.build_figshare_npz(os.path.join(DL, 'figshare256.npz'), DL + '/fig_raw')
print('Архив организаторов:', org, '\nFigshare:', fig)
t0 = time.time()
df, A, ex_df, AX = data.load_org(REPO + '/data/split_v3.csv', org)
fX, fM, fy, fpid = data.load_figshare(fig)
S = data.build_sets(df, A, fX, fM, fy, fpid)
print(f'Данные готовы за {time.time() - t0:.0f} с')
""")

code(r"""
# === 3. Короткая проверка (SMOKE, ~3 мин), затем полный запуск ===
from almazov import train
if not os.path.exists(os.path.join(OUT, 'seg_stage1.npz')):     # при повторном запуске проверку не повторяем
    shutil.rmtree(SMOKE_OUT, ignore_errors=True)
    train.run_stage2(df, A, fX, fM, fy, fpid, S, SMOKE_OUT, REPO, smoke=True, ex_df=ex_df, AX=AX)
    print('\n=== SMOKE OK — запускаю полный прогон ===\n')
SM = train.run_stage2(df, A, fX, fM, fy, fpid, S, OUT, REPO, ex_df=ex_df, AX=AX)
""")

code(r"""
# === 4. Упаковка результатов ===
files = sorted(p for p in glob.glob(os.path.join(OUT, '*')) if os.path.isfile(p))
if KAGGLE:
    def pack(dst, names, method):
        with zipfile.ZipFile(dst, 'w', method) as z:
            for p in names: z.write(p, os.path.basename(p))
        with zipfile.ZipFile(dst) as z: assert z.testzip() is None
        return os.path.getsize(dst) / 1e6
    w = [p for p in files if os.path.basename(p).startswith('w_')]
    r = [p for p in files if p not in w]
    print('stage2_results.zip:', round(pack('/kaggle/working/stage2_results.zip', r, zipfile.ZIP_DEFLATED)), 'МБ')
    print('stage2_weights.zip:', round(pack('/kaggle/working/stage2_weights.zip', w, zipfile.ZIP_STORED)), 'МБ')
    for p in ['summary.json', 'log.txt']: shutil.copy(os.path.join(OUT, p), '/kaggle/working/' + p)
    shutil.rmtree(OUT)
    print('Готово. Скачайте stage2_results.zip и stage2_weights.zip (вкладка Output).')
else:   # Colab: копия крупных файлов частями по 9 МБ (так их можно забрать из Drive автоматически)
    PD = os.path.join(OUT, 'parts'); os.makedirs(PD, exist_ok=True)
    for p in files:
        data_ = open(p, 'rb').read()
        for i in range(0, len(data_), 9_000_000):
            open(os.path.join(PD, f'{os.path.basename(p)}.part{i // 9_000_000:03d}'), 'wb').write(data_[i:i + 9_000_000])
    print('Готово. Результаты в Google Drive: almazov/stage2 — напишите мне.')
print(json.dumps({k: v.get('test') for k, v in SM.items() if isinstance(v, dict) and 'test' in v}, ensure_ascii=False, indent=1)[:4000])
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []},
      "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
