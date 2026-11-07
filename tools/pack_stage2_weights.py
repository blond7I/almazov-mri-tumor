"""stage2_weights.zip (выгрузка Kaggle) -> weights/stage2/: части по 20 МБ + models.json (контрольные суммы) для прототипа."""
import hashlib, json, sys, zipfile
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
models = []
with zipfile.ZipFile(src) as z:
    for name in sorted(n for n in z.namelist() if n.startswith('w_') and n.endswith('.pt')):
        data = z.read(name)
        tag = name[2:-3]                                     # w_seg_s1.pt -> seg_s1
        parts = []
        for i in range(0, len(data), 20 * 1024 * 1024):
            pn = f'{tag}.pt.part{i // (20 * 1024 * 1024)}'
            (dst / pn).write_bytes(data[i:i + 20 * 1024 * 1024]); parts.append(pn)
        models.append(dict(tag=tag, kind='seg' if tag.startswith('seg') else 'cls', parts=parts,
                           sha256=hashlib.sha256(data).hexdigest(), mb=round(len(data) / 1e6, 1)))
json.dump(dict(models=models, note='Новые обучения этапа 2 (тот же рецепт, другие seed), float16'), open(dst / 'models.json', 'w'), indent=1)
print(json.dumps(models, indent=1))
