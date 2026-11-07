"""Синтетические данные той же структуры, что у настоящих (для проверки кода без GPU и без скачивания)."""
import sys, zipfile, io
import numpy as np, pandas as pd, cv2
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sv = pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'split_v3.csv', keep_default_na=False)
rs = np.random.RandomState(0)
with zipfile.ZipFile(out / 'Almazov_MRI.zip', 'w', zipfile.ZIP_STORED) as z:
    for p, lab in zip(sv.path, sv.label):
        h, w = rs.choice([200, 256, 512]), rs.choice([200, 256, 512])
        img = (rs.rand(h // 4, w // 4) * 60).astype(np.uint8)
        img = cv2.resize(img, (w, h))
        if lab != 3: cv2.circle(img, (w // 2, h // 2), min(h, w) // 8, 255, -1)
        ok, b = cv2.imencode('.jpg', img)
        z.writestr('Almazov_MRI/' + p[len('/content/org/'):], b.tobytes())
nfig = 3064
X = (rs.rand(nfig, 256, 256) * 80).astype(np.uint8); M = np.zeros((nfig, 256, 256), np.uint8)
for i in range(nfig):
    cv2.circle(X[i], (128, 128), 30, 255, -1); cv2.circle(M[i], (128, 128), 30, 1, -1)
pid = np.array([f'P{rs.randint(10000)}' for _ in range(nfig)], dtype=object)
for fi, p in zip(sv.fig_i, sv.pid):
    if fi >= 0: pid[fi] = p
y = rs.randint(0, 3, nfig)
np.savez_compressed(out / 'figshare256.npz', X=X, M=M, y=y, pid=pid.astype(str), idx=np.arange(nfig))
print('ok', out)
