"""Проверка офлайн-демо: модели собираются из weights/, каждый пример из examples/ анализируется, заключение строится."""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from almazov import pipeline, report  # noqa: E402

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
t0 = time.time()
M = pipeline.load_models(root)
print(f'Модели: {M.version} ({time.time() - t0:.0f} с)')
assert len(M.seg) == 4 and len(M.cls) == 3, 'Ансамбль собран не полностью: скачайте репозиторий целиком (папка weights/)'
for f in sorted(glob.glob(os.path.join(root, 'examples', '*.jpg'))):
    t = time.time()
    r = pipeline.analyze(M, f)
    txt = report.build(r)["text"]
    print(f'  {os.path.basename(f):18s} -> {r["top_type"] if r["tumor"] else "опухоли нет":17s} '
          f'находок {len(r["findings"])}, {time.time() - t:.1f} с, заключение {len(txt)} знаков')
print('OK: офлайн-демо готово')
