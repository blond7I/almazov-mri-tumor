"""Страница ручной разметки режима МРТ и плоскости (публикуется как артефакт с хранилищем `db`).

    python tools/build_label_page.py IMAGES_ZIP_OR_DIR LABEL_POOL_NPZ OUT.html

IMAGES — images_jpg.zip из Kaggle-выгрузки (файлы pool/<индекс>.jpg) или папка с такими файлами.
Разметка сохраняется в хранилище артефакта (коллекция labels, документ = индекс снимка в split_v3.csv),
без сети — в браузере, и досылается, когда сеть появится.
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def load_images(src, idx):
    out = {}
    src = Path(src)
    if src.suffix == '.zip':
        z = zipfile.ZipFile(src)
        names = {Path(n).stem: n for n in z.namelist() if n.startswith('pool/')}
        get = lambda i: z.read(names[f'{i:05d}'])
    else:
        get = lambda i: (src / f'{i:05d}.jpg').read_bytes()
    for i in idx:
        im = Image.open(io.BytesIO(get(int(i)))).convert('L')
        if max(im.size) > 256:
            im.thumbnail((256, 256))
        b = io.BytesIO()
        im.save(b, 'JPEG', quality=86, optimize=True)
        out[int(i)] = base64.b64encode(b.getvalue()).decode()
    return out


PAGE = r'''<title>Разметка режимов МРТ</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap">
<style>
/* Разметочный пульт в духе рабочей станции рентгенолога: тёмный фон без бликов, снимок — главный, кнопки крупные под палец */
:root{
  --bg:#0f1418; --panel:#161d23; --line:#25313b; --fg:#e4ebf1; --muted:#8a9aa8;
  --accent:#8cc8e6; --accent-ink:#0c1a22; --ok:#78c99a; --warn:#e3b65e;
  --ui:"Golos Text",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,Consolas,monospace;
  color-scheme:dark;
}
html,body{height:100%}
body{background:var(--bg);color:var(--fg);font:15px/1.45 var(--ui);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:560px;margin:0 auto;display:flex;flex-direction:column;gap:12px;padding-block:12px 24px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
h1{font-size:17px;font-weight:600;margin:0;text-wrap:balance}
.count{font:500 14px var(--mono);font-variant-numeric:tabular-nums;color:var(--muted)}
.count b{color:var(--fg);font-weight:500}
.bar{height:4px;background:var(--line);border-radius:2px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .2s}
.sync{font-size:12.5px;color:var(--muted);display:flex;align-items:center;gap:6px}
.sync::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--muted)}
.sync.ok::before{background:var(--ok)} .sync.wait::before{background:var(--warn)}
figure{margin:0;position:relative;background:#000;border:1px solid var(--line);border-radius:6px;aspect-ratio:1;max-width:100%;overflow:hidden}
figure img{width:100%;height:100%;object-fit:contain;display:block;image-rendering:auto}
figcaption{position:absolute;left:8px;top:6px;font:500 12px var(--mono);color:var(--muted);background:rgba(0,0,0,.55);padding:2px 6px;border-radius:3px}
.group{display:flex;flex-direction:column;gap:6px}
.label{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));gap:6px}
button{font:500 15px var(--ui);color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:11px 8px;cursor:pointer;min-height:46px;touch-action:manipulation}
button:hover{border-color:var(--accent)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
button kbd{font:500 11px var(--mono);color:inherit;opacity:.55;margin-left:6px}
.nav{display:flex;gap:6px}
.nav button{flex:1;background:transparent}
details{border:1px solid var(--line);border-radius:6px;padding:10px 12px;background:var(--panel)}
summary{cursor:pointer;font-weight:500}
details p{margin:8px 0 0;color:var(--muted);max-width:65ch}
details b{color:var(--fg);font-weight:500}
.done{border:1px solid var(--ok);border-radius:6px;padding:12px;color:var(--fg)}
@media (prefers-reduced-motion:reduce){.bar i{transition:none}}
@media (hover:none),(max-width:520px){button kbd{display:none}}
</style>
<div class="wrap">
  <header>
    <h1>Режим МРТ и плоскость среза</h1>
    <span class="count"><b id="n">0</b> / <span id="total">0</span></span>
  </header>
  <div class="bar"><i id="prog"></i></div>
  <div class="sync" id="sync">Подключаюсь к хранилищу…</div>
  <figure><img id="img" alt="Срез МРТ для разметки"><figcaption id="cap"></figcaption></figure>
  <div class="group"><span class="label">Режим</span>
    <div class="opts" id="seq"></div></div>
  <div class="group"><span class="label">Плоскость</span>
    <div class="opts" id="plane"></div></div>
  <div class="nav">
    <button id="back" type="button">← Назад</button>
    <button id="skip" type="button">Пропустить</button>
    <button id="first" type="button">К неразмеченным</button>
  </div>
  <div class="done" id="done" hidden>Все снимки размечены. Спасибо! Можно закрыть страницу — ответы сохранены.</div>
  <details>
    <summary>Как отличить режим и плоскость</summary>
    <p><b>T2</b>: жидкость (желудочки, борозды вокруг мозга) ярко-белая; серое вещество светлее белого.</p>
    <p><b>FLAIR</b>: похож на T2, но жидкость в желудочках тёмная; отёк и очаги — яркие.</p>
    <p><b>T1 без контраста</b>: жидкость тёмная, белое вещество светлее серого; сосуды и оболочки не светятся.</p>
    <p><b>T1 с контрастом</b>: как T1, но ярко светятся сосуды, оболочки, гипофиз, сосудистые сплетения; опухоль часто светится по краю или целиком.</p>
    <p><b>Аксиальная</b>: вид сверху — овал головы, полушария слева и справа. <b>Сагиттальная</b>: вид сбоку — профиль лица, мозжечок сзади. <b>Корональная</b>: вид спереди — полушария симметрично, височные доли внизу по бокам.</p>
    <p>Не уверен — жми «Не знаю»: честное «не знаю» лучше угаданного ответа.</p>
  </details>
</div>
<script>
const ITEMS = __ITEMS__;
const SEQ = [["T1c","T1 + контраст","1"],["T1","T1","2"],["T2","T2","3"],["FLAIR","FLAIR","4"],["unk","Не знаю","5"]];
const PLANE = [["ax","Аксиальная","Q"],["sag","Сагиттальная","W"],["cor","Корональная","E"],["unk","Не знаю","R"]];
const KEY = "mri-labels-v1";
let labels = {};          // id -> {seq, plane, at}
let pending = {};         // id -> true (ещё не в хранилище)
try { const s = JSON.parse(localStorage.getItem(KEY) || "{}"); labels = s.labels || {}; pending = s.pending || {}; } catch (e) {}
const saveLocal = () => { try { localStorage.setItem(KEY, JSON.stringify({labels, pending})); } catch (e) {} };
let pos = 0, cur = {}, db = null, dbState = "connecting";
const $ = id => document.getElementById(id);
$("total").textContent = ITEMS.length;

function mk(box, list, field) {
  for (const [v, t, k] of list) {
    const b = document.createElement("button");
    b.type = "button"; b.dataset.v = v; b.setAttribute("aria-pressed", "false");
    b.innerHTML = t + "<kbd>" + k + "</kbd>";
    b.addEventListener("click", () => choose(field, v));
    box.appendChild(b);
  }
}
mk($("seq"), SEQ, "seq"); mk($("plane"), PLANE, "plane");

function nDone() { return ITEMS.filter(x => labels[x.i] && labels[x.i].seq && labels[x.i].plane).length; }
function firstOpen(from = 0) {
  for (let k = 0; k < ITEMS.length; k++) { const j = (from + k) % ITEMS.length; const l = labels[ITEMS[j].i]; if (!l || !l.seq || !l.plane) return j; }
  return -1;
}
function render() {
  const it = ITEMS[pos];
  $("img").src = "data:image/jpeg;base64," + it.b;
  $("cap").textContent = "№ " + it.i;
  cur = Object.assign({}, labels[it.i] || {});
  for (const b of $("seq").children) b.setAttribute("aria-pressed", String(cur.seq === b.dataset.v));
  for (const b of $("plane").children) b.setAttribute("aria-pressed", String(cur.plane === b.dataset.v));
  const d = nDone();
  $("n").textContent = d; $("prog").style.width = (100 * d / ITEMS.length) + "%";
  $("done").hidden = d < ITEMS.length;
  syncText();
}
function syncText() {
  const p = Object.keys(pending).length, el = $("sync");
  if (dbState === "ok" && !p) { el.className = "sync ok"; el.textContent = "Всё сохранено в хранилище"; }
  else if (dbState === "none") { el.className = "sync wait"; el.textContent = "Хранилище недоступно: ответы сохраняются только в этом браузере (" + Object.keys(labels).length + ")"; }
  else { el.className = "sync wait"; el.textContent = p ? "Ждут отправки: " + p + " — отправятся, когда будет сеть" : "Подключаюсь к хранилищу…"; }
}
let flushing = false;
async function flush() {
  if (!db || flushing) return;
  flushing = true;
  try {
    for (const id of Object.keys(pending)) {
      try { await db.collection("labels").doc(String(id)).set(labels[id]); delete pending[id]; saveLocal(); dbState = "ok"; }
      catch (e) { dbState = "wait"; break; }
    }
  } finally { flushing = false; syncText(); }
}
function choose(field, v) {
  const it = ITEMS[pos];
  cur[field] = v;
  labels[it.i] = Object.assign({}, labels[it.i] || {}, {[field]: v, at: Date.now()});
  pending[it.i] = true; saveLocal();
  render();
  if (cur.seq && cur.plane) {
    flush();
    setTimeout(() => { const j = firstOpen(pos + 1); if (j >= 0) { pos = j; render(); } }, 220);
  }
}
$("back").addEventListener("click", () => { pos = (pos - 1 + ITEMS.length) % ITEMS.length; render(); });
$("skip").addEventListener("click", () => { pos = (pos + 1) % ITEMS.length; render(); });
$("first").addEventListener("click", () => { const j = firstOpen(0); if (j >= 0) { pos = j; render(); } });
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toUpperCase();
  const s = SEQ.find(x => x[2] === k), p = PLANE.find(x => x[2] === k || x[2] === ({"Й":"Q","Ц":"W","У":"E","К":"R"})[k]);
  if (s) choose("seq", s[0]); else if (p) choose("plane", p[0]);
  else if (e.key === "ArrowLeft") $("back").click(); else if (e.key === "ArrowRight") $("skip").click();
});
window.addEventListener("online", flush);
setInterval(() => { if (Object.keys(pending).length) flush(); }, 20000);

pos = Math.max(0, firstOpen(0)); render();
(async () => {
  const timer = setTimeout(() => { if (dbState === "connecting") { dbState = "wait"; syncText(); } }, 10000);
  try { db = window.claude ? await window.claude.use("db") : null; } catch (e) { db = null; }
  if (!db) { clearTimeout(timer); dbState = "none"; syncText(); return; }
  try {
    const snap = await db.collection("labels").get();
    for (const d of snap.docs) {
      const v = d.data(); const l = labels[d.id];
      if (!l || (v && v.at && (!l.at || v.at >= l.at && !pending[d.id]))) labels[d.id] = v;
    }
    saveLocal(); dbState = "ok"; clearTimeout(timer);
    pos = Math.max(0, firstOpen(0)); render(); flush();
  } catch (e) { clearTimeout(timer); dbState = "wait"; syncText(); }
})();
</script>
'''


def main():
    src, pool_npz, out = sys.argv[1:4]
    z = np.load(pool_npz, allow_pickle=True)
    idx = [int(i) for i in z['idx']]
    imgs = load_images(src, idx)
    rs = np.random.RandomState(7)
    order = rs.permutation(len(idx))           # перемешано: классы не идут подряд, разметка «вслепую»
    items = [dict(i=idx[k], b=imgs[idx[k]]) for k in order]
    html = PAGE.replace('__ITEMS__', json.dumps(items, separators=(',', ':')))
    Path(out).write_text(html, encoding='utf-8')
    print(f'{out}: {len(items)} снимков, {len(html) / 1e6:.1f} МБ')


if __name__ == '__main__':
    main()
