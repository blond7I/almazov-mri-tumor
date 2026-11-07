"""Собирает demo_colab.ipynb (этап 2): рабочее место врача + пакетная проверка, видеокарта не нужна."""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md(r"""
# МРТ головного мозга: контур, сомнения и проект заключения — демо (этап 2, команда InTeam)
**Как запустить:** «Среда выполнения» → «Выполнить все». Видеокарта не нужна, установка 2–3 минуты.
В конце первой ячейки появится ссылка `https://….gradio.live` — это рабочее место врача: загрузите срез, нажмите «Анализ»,
поправьте находки или текст, нажмите «Сохранить» (PDF заключения + журнал + маска).

Исследовательский прототип, не медицинское изделие. Решение принимает врач. Вопросы: Telegram @blond7I.
""")
code(r"""
# === 1. Установка и запуск рабочего места врача ===
import os, subprocess, sys
REPO = 'https://github.com/blond7I/almazov-mri-tumor'
if os.path.exists('repo'):
    subprocess.run(['git', '-C', 'repo', 'pull', '--ff-only', '-q'], check=False)
else:
    subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO, 'repo'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.31', 'segmentation-models-pytorch==0.6.0', 'reportlab'], check=True)
sys.path.insert(0, 'repo/src')
import glob
from almazov import pipeline, app
M = pipeline.load_models('repo', device='cuda' if __import__('torch').cuda.is_available() else 'cpu')
print('Модели:', M.version)
app.build_ui(M, sorted(glob.glob('repo/examples/*.jpg'))[:8]).launch(share=True, debug=False)
""")
code(r"""
# === 2. Пакетная проверка папки или .zip со снимками ===
# Если снимки разложены по папкам классов (glioma/, meningioma/, pituitary/, notumor/ или по-русски), посчитаются точность,
# чувствительность и специфичность. Результат: results_batch/batch_results.csv, картинки с контуром и JSON по каждому снимку.
from almazov import cli
FOLDER = 'repo/examples'          # <- путь к своей папке или .zip
df = cli.run(FOLDER, 'results_batch', 'repo', device='cpu')
df.head(20)
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "colab": {"provenance": []},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print('ok', len(cells))
