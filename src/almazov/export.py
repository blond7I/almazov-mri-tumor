"""«Сохранить»: ZIP с PDF заключения, JSON-журналом (результат ИИ, правки врача, время, версия моделей) и маской PNG."""
import datetime as _dt
import io
import json
import os
import zipfile

import cv2
import numpy as np


def _font():
    """Шрифт с кириллицей для PDF (DejaVu Sans есть в составе matplotlib)."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    if 'DejaVu' in pdfmetrics.getRegisteredFontNames():
        return 'DejaVu', 'DejaVu-Bold'
    import matplotlib
    d = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
    pdfmetrics.registerFont(TTFont('DejaVu', os.path.join(d, 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', os.path.join(d, 'DejaVuSans-Bold.ttf')))
    return 'DejaVu', 'DejaVu-Bold'


def make_pdf(text, overlay_rgb=None, title='Проект заключения МРТ головного мозга (один срез)'):
    """Текст заключения (разделы — строки ЗАГЛАВНЫМИ) + картинка с контуром -> PDF A4 (bytes)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from xml.sax.saxutils import escape
    reg, bold = _font()
    st_t = ParagraphStyle('t', fontName=bold, fontSize=13, leading=16, spaceAfter=6)
    st_h = ParagraphStyle('h', fontName=bold, fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2)
    st_p = ParagraphStyle('p', fontName=reg, fontSize=9.5, leading=12.5)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                            title=title)
    story = [Paragraph(escape(title), st_t)]
    if overlay_rgb is not None:
        ok, png = cv2.imencode('.png', cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        story += [RLImage(io.BytesIO(png.tobytes()), width=70 * mm, height=70 * mm), Spacer(1, 4)]
    for block in text.strip().split('\n\n'):
        lines = block.strip().split('\n')
        if lines and lines[0].isupper():
            hd = lines[0].capitalize().replace(' ии ', ' ИИ ').replace('мрт', 'МРТ')
            story.append(Paragraph(escape(hd), st_h))
            lines = lines[1:]
        for ln in lines:
            story.append(Paragraph(escape(ln), st_p))
    doc.build(story)
    return buf.getvalue()


def bundle(path, res, final_text, final_mask, overlay_rgb, log, meta, findings, warnings, started_at):
    """Записать ZIP: zaklyuchenie.pdf, journal.json, mask.png (контур врача), overlay.png."""
    from .pipeline import to_json
    now = _dt.datetime.now()
    ai_mask = res['arrays']['mask']
    inter = float((ai_mask & final_mask).sum())
    dice = (2 * inter + 1e-6) / (float(ai_mask.sum() + final_mask.sum()) + 1e-6)
    journal = dict(
        saved_at=now.isoformat(timespec='seconds'), seconds_spent=round((now - started_at).total_seconds()),
        model_version=res.get('model_version'), meta=meta,
        ai_result=to_json(res), final_findings=findings, final_text=final_text,
        doctor_edits=log, text_warnings_at_save=warnings,
        contour_agreement_dice_ai_vs_final=round(dice, 4),
        note='Исследовательский прототип, не медицинское изделие. Заключение требует подписи врача.')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('zaklyuchenie.pdf', make_pdf(final_text, overlay_rgb))
        z.writestr('journal.json', json.dumps(journal, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
        z.writestr('mask.png', cv2.imencode('.png', (final_mask > 0).astype(np.uint8) * 255)[1].tobytes())
        z.writestr('overlay.png', cv2.imencode('.png', cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))[1].tobytes())
    return path
