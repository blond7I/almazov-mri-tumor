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

    def on_analyze(img, seq, plane, study, layers):
        if img is None:
            raise gr.Error('Загрузите срез МРТ (JPG/PNG).')
        res = pipeline.analyze(M, img, source_name='загруженный снимок')
        st = dict(res=res, mask=res['arrays']['mask'].copy(), findings=copy.deepcopy(res['findings']),
                  zones=copy.deepcopy(res['review_zones']), log=[], started=_dt.datetime.now(), edited=False)
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
        export.bundle(p, st['res'], text, st['mask'], render(st, layers), st['log'], meta_of(seq, plane, study, None),
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
                btn = gr.Button('Анализ', variant='primary')
                out = gr.Image(label='Результат', height=520, interactive=False)
                layers = gr.CheckboxGroup(LAYERS, value=LAYERS, label='Слои')
                thr = gr.Slider(0.2, 0.8, value=0.5, step=0.05, label='Порог контура (ниже — шире контур)')
                with gr.Row():
                    fsel = gr.CheckboxGroup([], label='Находки — отметьте лишние')
                    bdel = gr.Button('Удалить отмеченные')
                with gr.Row():
                    zsel = gr.Dropdown([], label='Зона дополнительного просмотра')
                    bpro = gr.Button('Сделать находкой')
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
        btn.click(on_analyze, [img, seq, plane, study, layers], [st, out, probs, text, fsel, zsel, thr, stat, warn])
        thr.release(on_thr, [thr, st, layers], [st, out, fsel, zsel, stat])
        bdel.click(on_delete, [fsel, st, layers], [st, out, fsel, zsel, stat])
        bpro.click(on_promote, [zsel, st, layers], [st, out, fsel, zsel, stat])
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
