"""Рабочее место врача (Gradio): снимок со слоями рядом с редактируемым проектом заключения.

    python -m almazov.app            # локально: http://127.0.0.1:7860
    python -m almazov.app --share    # временная публичная ссылка (Colab)
"""
import argparse
import copy
import datetime as _dt
import glob
import os
import tempfile
from pathlib import Path

import numpy as np

from .common import CLASSES
from . import pipeline, report, export

SEQ = ['не указан', 'T1 с контрастом', 'T1 без контраста', 'T2', 'FLAIR', 'другой']
PLANE = ['не указана', 'аксиальная', 'сагиттальная', 'корональная']
LAYERS = ['Полоса неуверенной границы', 'Зоны дополнительного просмотра']
FLAG_TEXT = {
    'classifier_tumor_no_contour': 'Классификатор видит опухоль, контура нет — просмотрите весь срез',
    'contour_without_classifier': 'Есть контур, но классификатор опухоли не видит',
    'type_uncertain': 'Тип определён неуверенно',
    'presence_uncertain': 'Наличие опухоли неуверенно',
    'boundary_uncertain': 'Граница неуверенная',
    'review_zones': 'Есть зоны дополнительного просмотра',
    'low_resolution': 'Низкое разрешение снимка',
    'color_image': 'Цветное изображение — возможно, не срез МРТ',
}


def build_ui(M, examples=()):
    import gradio as gr

    def meta_of(seq, plane, study, fname):
        return dict(sequence=None if seq == SEQ[0] else seq, plane=None if plane == PLANE[0] else plane,
                    study_id=study or None, file=fname)

    def render(st, layers):
        r = dict(st['res'], findings=st['findings'], review_zones=st['zones'])
        return pipeline.overlay(r, show_band=LAYERS[0] in layers, show_review=LAYERS[1] in layers, mask=st['mask'])

    def choices(st):
        f = [f'№{x["id"]}: {report._where(x["half"])}, {x["area_pct_head"]}% площади головы' for x in st['findings']]
        z = [f'?{x["id"]}: {report._where(x["half"])}' for x in st['zones']]
        return gr.update(choices=f, value=[]), gr.update(choices=z, value=None)

    def status(st, note=''):
        r = st['res']
        fl = [FLAG_TEXT.get(x, x) for x in r['flags']]
        if r.get('refused'):
            return '🚫 **Анализ не выполнен.** ' + ' '.join(c['text'] for c in r['checks'] if c['level'] == 'refuse')
        s = f'**Модели:** {r["model_version"]}.  **Находок:** {len(st["findings"])}, зон просмотра: {len(st["zones"])}.'
        if fl:
            s += '\n\n⚠️ ' + ' · '.join(fl)
        return s + (f'\n\n{note}' if note else '')

    def on_analyze(img, seq, plane, study, layers, blind=False):
        if img is None:
            raise gr.Error('Загрузите срез МРТ (JPG/PNG).')
        res = pipeline.analyze(M, img, source_name='загруженный снимок')
        st = dict(res=res, mask=res['arrays']['mask'].copy(), findings=copy.deepcopy(res['findings']),
                  zones=copy.deepcopy(res['review_zones']), log=[], started=_dt.datetime.now(), edited=False, blind=bool(blind))
        if blind:   # проверка врачом: «обвести с нуля» без подсказок ИИ (для сравнения времени)
            st['mask'] = np.zeros_like(st['mask']); st['findings'] = []; st['zones'] = []
            st['res'] = dict(res, arrays=dict(res['arrays'], band=np.zeros_like(res['arrays']['band'])))
            st['log'].append(dict(t=0, action='режим без подсказок ИИ'))
            fc, zc = choices(st)
            return (st, render(st, []), {}, '', fc, zc, gr.update(), '**Режим «обвести с нуля»**: подсказки ИИ скрыты. '
                    'Обведите опухоль кистью (красная), затем «Сохранить».', '')
        txt = report.build(res, st['findings'], st['zones'], meta_of(seq, plane, study, None))['text']
        fc, zc = choices(st)
        return (st, render(st, layers), {c: res['probs'][c] for c in CLASSES}, txt, fc, zc,
                gr.update(value=res['calib']['contour_thr']), status(st), '')

    def log(st, action, **kw):
        st['log'].append(dict(t=round((_dt.datetime.now() - st['started']).total_seconds(), 1), action=action, **kw))

    def refresh(st, layers, note=''):
        fc, zc = choices(st)
        return st, render(st, layers), fc, zc, status(st, note)

    def on_thr(thr, st, layers):
        if not st:
            return st, None, gr.update(), gr.update(), ''
        st['mask'] = pipeline.mask_at(st['res'], thr)
        st['findings'] = pipeline.recompute(st['res'], st['mask'], edited_ids=[0] if thr != st['res']['calib']['contour_thr'] else ())
        log(st, 'порог контура', value=thr)
        return refresh(st, layers, 'Порог изменён — нажмите «Пересчитать заключение».')

    def on_delete(sel, st, layers):
        if not st or not sel:
            return refresh(st, layers) if st else (st, None, gr.update(), gr.update(), '')
        ids = [int(s.split(':')[0][1:]) for s in sel]
        import cv2
        k, lab = cv2.connectedComponents(st['mask'])
        for f in st['findings']:
            if f['id'] in ids:
                x, y, w, h = f['bbox256']
                sub = lab[y:y + h, x:x + w]
                vals, cnt = np.unique(sub[sub > 0], return_counts=True)
                if len(vals):
                    st['mask'][lab == vals[np.argmax(cnt)]] = 0
        st['findings'] = pipeline.recompute(st['res'], st['mask'], edited_ids=ids)
        log(st, 'удалены находки', ids=ids)
        return refresh(st, layers, f'Удалено: {", ".join("№" + str(i) for i in ids)}. Нажмите «Пересчитать заключение».')

    def on_promote(z, st, layers):
        if not st or not z:
            return refresh(st, layers) if st else (st, None, gr.update(), gr.update(), '')
        zid = int(z.split(':')[0][1:])
        st['mask'] = (st['mask'] | pipeline.zone_mask(st['res'], zid)).astype(np.uint8)
        st['zones'] = [x for x in st['zones'] if x['id'] != zid]
        st['findings'] = pipeline.recompute(st['res'], st['mask'], edited_ids=[zid])
        log(st, 'зона сделана находкой', zone=zid)
        return refresh(st, layers, f'Зона ?{zid} добавлена как находка. Нажмите «Пересчитать заключение».')

    def editor_value(st, layers):
        return dict(background=render(st, layers), layers=[], composite=None)

    def on_brush(ed, st, layers):
        if not st or ed is None:
            return (st, None, gr.update(), gr.update(), '', gr.update()) if not st else (*refresh(st, layers), gr.update())
        import cv2
        lay = None
        if ed.get('layers'):
            lay = np.asarray(ed['layers'][0])
        elif ed.get('composite') is not None and ed.get('background') is not None:
            diff = np.abs(np.asarray(ed['composite'])[..., :3].astype(int) - np.asarray(ed['background'])[..., :3].astype(int)).sum(2) > 30
            lay = np.concatenate([np.asarray(ed['composite'])[..., :3], (diff * 255).astype(np.uint8)[..., None]], 2)
        if lay is None or lay.shape[-1] < 4 or not (lay[..., 3] > 0).any():
            return (*refresh(st, layers, 'Кисть: изменений нет.'), gr.update())
        a = lay[..., 3] > 0
        r, g_, b = [lay[..., i].astype(int) for i in range(3)]
        add = a & (r > b + 40)
        rem = a & (b > r + 40)
        to256 = lambda m: cv2.resize(m.astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA) > 0
        m = st['mask'].astype(bool)
        m = (m | to256(add)) & ~to256(rem)
        k, lab, stt, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        for c in range(1, k):
            if stt[c, cv2.CC_STAT_AREA] < 10:
                m[lab == c] = False
        st['mask'] = m.astype(np.uint8)
        st['findings'] = pipeline.recompute(st['res'], st['mask'], edited_ids=[0])
        log(st, 'правка кистью', added_px256=int(to256(add).sum()), removed_px256=int(to256(rem).sum()))
        out = refresh(st, layers, 'Контур исправлен кистью. Нажмите «Пересчитать заключение».')
        return (*out, editor_value(st, layers))

    def on_layers(layers, st):
        return render(st, layers) if st else None

    def on_rebuild(st, seq, plane, study):
        if not st:
            return '', ''
        log(st, 'заключение пересчитано')
        return report.build(st['res'], st['findings'], st['zones'], meta_of(seq, plane, study, None))['text'], ''

    def on_check(text, st):
        if not st:
            return ''
        w = report.check_text(text, st['res'], st['findings'])
        return '\n'.join('⚠️ ' + x for x in w) if w else '✅ Текст не противоречит анализу.'

    def on_save(text, st, seq, plane, study, layers):
        if not st:
            raise gr.Error('Сначала выполните анализ.')
        w = report.check_text(text, st['res'], st['findings'])
        log(st, 'сохранено', warnings=len(w))
        d = tempfile.mkdtemp()
        p = os.path.join(d, f'zaklyuchenie_{_dt.datetime.now():%Y%m%d_%H%M%S}.zip')
        export.bundle(p, st['res'], text, st['mask'], render(st, layers), st['log'], dict(meta_of(seq, plane, study, None), blind=st.get('blind', False)),
                      st['findings'], w, st['started'])
        note = ('⚠️ Сохранено с предупреждениями:\n' + '\n'.join('— ' + x for x in w)) if w else '✅ Сохранено.'
        return p, note

    with gr.Blocks(title='МРТ: проект заключения') as ui:
        gr.Markdown('## МРТ головного мозга: контур, сомнения и проект заключения\n'
                    'Исследовательский прототип, не медицинское изделие. **Решение принимает врач.** '
                    'Красное — контур, жёлтое — неуверенная граница (опухоль может доходить до внешнего края), голубые рамки «?» — зоны дополнительного просмотра.')
        st = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5):
                img = gr.Image(type='numpy', label='Срез МРТ (JPG/PNG)', height=260)
                with gr.Row():
                    seq = gr.Dropdown(SEQ, value=SEQ[0], label='Режим МРТ')
                    plane = gr.Dropdown(PLANE, value=PLANE[0], label='Плоскость')
                    study = gr.Textbox(label='Код исследования (обезличенный)', max_lines=1)
                with gr.Row():
                    btn = gr.Button('Анализ', variant='primary', scale=3)
                    blind = gr.Checkbox(label='Обвести с нуля (без ИИ)', value=False, scale=2)
                out = gr.Image(label='Результат', height=520, interactive=False)
                layers = gr.CheckboxGroup(LAYERS, value=LAYERS, label='Слои')
                thr = gr.Slider(0.2, 0.8, value=0.5, step=0.05, label='Порог контура (ниже — шире контур)')
                with gr.Row():
                    fsel = gr.CheckboxGroup([], label='Находки — отметьте лишние')
                    bdel = gr.Button('Удалить отмеченные')
                with gr.Row():
                    zsel = gr.Dropdown([], label='Зона дополнительного просмотра')
                    bpro = gr.Button('Сделать находкой')
                with gr.Accordion('Правка контура кистью: красная — добавить, синяя — убрать', open=False):
                    ed = gr.ImageEditor(type='numpy', canvas_size=(768, 768), fixed_canvas=True, sources=[], layers=False,
                                        transforms=[], height=520, label='Рисуйте поверх снимка',
                                        brush=gr.Brush(colors=['#ff2020', '#2060ff'], default_color='#ff2020', default_size=10,
                                                       color_mode='fixed'))
                    bbr = gr.Button('Применить правку кистью')
                stat = gr.Markdown()
            with gr.Column(scale=5):
                probs = gr.Label(num_top_classes=4, label='Результат ИИ (справочно)')
                text = gr.Textbox(label='Проект заключения — можно править', lines=26)
                with gr.Row():
                    brb = gr.Button('Пересчитать заключение по текущим находкам')
                    bch = gr.Button('Проверить текст')
                warn = gr.Markdown()
                bsave = gr.Button('Сохранить (PDF + журнал + маска)', variant='primary')
                fout = gr.File(label='Архив заключения')
        if examples:
            gr.Examples(list(examples), inputs=img, label='Примеры (тестовая часть, модель их не видела)')
        btn.click(on_analyze, [img, seq, plane, study, layers, blind], [st, out, probs, text, fsel, zsel, thr, stat, warn]).then(
            lambda st_, ly: editor_value(st_, ly) if st_ else None, [st, layers], ed)
        bbr.click(on_brush, [ed, st, layers], [st, out, fsel, zsel, stat, ed])
        thr.release(on_thr, [thr, st, layers], [st, out, fsel, zsel, stat]).then(
            lambda st_, ly: editor_value(st_, ly) if st_ else None, [st, layers], ed)
        bdel.click(on_delete, [fsel, st, layers], [st, out, fsel, zsel, stat]).then(
            lambda st_, ly: editor_value(st_, ly) if st_ else None, [st, layers], ed)
        bpro.click(on_promote, [zsel, st, layers], [st, out, fsel, zsel, stat]).then(
            lambda st_, ly: editor_value(st_, ly) if st_ else None, [st, layers], ed)
        layers.change(on_layers, [layers, st], out)
        brb.click(on_rebuild, [st, seq, plane, study], [text, warn])
        bch.click(on_check, [text, st], warn)
        bsave.click(on_save, [text, st, seq, plane, study, layers], [fout, warn])
    return ui


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--port', type=int, default=7860)
    a = ap.parse_args(argv)
    M = pipeline.load_models(a.repo, device=a.device)
    ex = sorted(glob.glob(os.path.join(a.repo, 'examples', '*.jpg')))[:8]
    print('Модели:', M.version)
    build_ui(M, ex).launch(share=a.share, server_port=a.port, inbrowser=not a.share)


if __name__ == '__main__':
    main()
