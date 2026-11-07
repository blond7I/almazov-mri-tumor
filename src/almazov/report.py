"""Проект заключения: детерминированный шаблон (без языковой модели) — каждая фраза выводится из измерений.

Правила (после медицинской рецензии плана):
- размеры по JPG/PNG — только в пикселях исходного снимка и в % площади головы на срезе; миллиметры — только если
  известен размер пикселя (DICOM PixelSpacing);
- сторона — «левая/правая половина изображения», а не полушарие: ориентация JPG неизвестна;
- без «глиобластомы», RANO и степени злокачественности; оговорка о закрытом наборе классов;
- вероятности — отдельным блоком «Результат ИИ (справочно)»;
- в конце — «требует проверки и подписи врача».
"""
import re
import datetime as _dt

from .common import CLASSES, NO_TUMOR

TYPE_TEXT = {'Глиома': 'глиома', 'Менингиома': 'менингиома', 'Опухоль гипофиза': 'опухоль гипофиза'}
TYPE_GEN = {'Глиома': 'глиомы', 'Менингиома': 'менингиомы', 'Опухоль гипофиза': 'опухоли гипофиза'}
HALF_GEN = {'левая': 'левой', 'правая': 'правой', 'центр': 'центральной'}


def _where(half):
    return 'по средней линии изображения' if half == 'центр' else f'в {HALF_GEN.get(half, half)} половине изображения'
VERT_GEN = {'верхняя': 'верхней', 'средняя': 'средней', 'нижняя': 'нижней'}


def _num(x, nd=0):
    s = f'{x:,.{nd}f}'.replace(',', ' ')
    return s.replace('.', ',')


def _pct(p):
    return f'{round(100 * p)}%'


def _size(f, px_mm):
    if px_mm and 'dmax_mm' in f:
        return (f'наибольший размер {_num(f["dmax_mm"], 1)} мм, перпендикулярный ему — {_num(f["dperp_mm"], 1)} мм, '
                f'площадь на срезе {_num(f["area_mm2"], 0)} мм²')
    return (f'наибольший размер {_num(f["dmax_px"])} пикс., перпендикулярный ему — {_num(f["dperp_px"])} пикс. исходного снимка; '
            f'площадь {_num(f["area_px_orig"])} пикс. ({_num(f["area_pct_head"], 1)}% площади головы на срезе)')


def build(res, findings=None, zones=None, meta=None):
    """res — результат pipeline.analyze; findings/zones — после правок врача (по умолчанию — как у системы).

    Возвращает dict: sections (заголовок -> текст), text (всё вместе), facts (что утверждает текст — для проверки).
    """
    meta = meta or {}
    F = res['findings'] if findings is None else findings
    Z = res['review_zones'] if zones is None else zones
    px_mm = res.get('pixel_spacing_mm')
    P = res['probs']
    h, w = res['orig_hw']
    tumor = bool(F) or res['tumor']
    top = res['top_type']
    tumor_types = sorted([c for c in CLASSES[:3]], key=lambda c: -P[c])
    second = tumor_types[1]
    S = {}

    # 1. Исходные данные
    src = [f'Изображение: {meta.get("file") or res.get("source") or "срез МРТ"}, {w}×{h} пикс.',
           f'Размер пикселя: {_num(px_mm, 3) + " мм" if px_mm else "неизвестен (JPG/PNG) — размеры даны в пикселях"}.',
           f'Режим МРТ: {meta.get("sequence") or "не указан"}; плоскость: {meta.get("plane") or "не указана"}.',
           'Объём анализа: один двумерный срез, не исследование целиком.']
    if meta.get('study_id'):
        src.insert(0, f'Исследование: {meta["study_id"]} (обезличено).')
    S['Исходные данные'] = '\n'.join(src)

    # 2. Описание
    d = []
    if F:
        for f in F:
            loc = f'{_where(f["half"])}, в {VERT_GEN.get(f.get("vert", ""), "средней")} трети по высоте'
            line = f'Находка №{f["id"]}: {loc} выделена область повышенной вероятности опухоли; {_size(f, px_mm)}.'
            if f.get('band_ratio', 0) > 0.6:
                line += ' Граница определяется неуверенно (жёлтая полоса на изображении).'
            if f.get('edited'):
                line += ' Контур исправлен врачом.'
            d.append(line)
    else:
        d.append('Областей, которые система выделяет как опухоль, на срезе не найдено.')
    for z in Z:
        d.append(f'Зона дополнительного просмотра ?{z["id"]}: {_where(z["half"])} часть моделей '
                 f'отмечает участок (до {_pct(z["max_prob"])}), контур не построен.')
    S['Описание'] = '\n'.join(d)

    # 3. Заключение (проект)
    if tumor:
        num = ', '.join(f'№{f["id"]}' for f in F) if F else ''
        c = (f'Признаки объёмного образования{" (находка " + num + ")" if num else ""}. '
             f'По характеру изображения наиболее вероятна {TYPE_TEXT[top]} (из трёх распознаваемых типов: глиома, менингиома, опухоль гипофиза).')
        if not F:
            c += ' Контур образования система не построила — требуется просмотр всего среза.'
    else:
        c = ('Признаков опухоли из распознаваемых системой типов (глиома, менингиома, опухоль гипофиза) на срезе не выявлено.')
    S['Заключение (проект)'] = c

    # 4. Сомнения
    fl = set(res['flags'])
    s = []
    if 'type_uncertain' in fl and tumor:
        s.append(f'Тип определён неуверенно: {TYPE_TEXT[top]} {_pct(P[top] / max(sum(P[t] for t in CLASSES[:3]), 1e-6))}, '
                 f'второй вариант — {TYPE_TEXT[second]}.')
    if 'presence_uncertain' in fl:
        s.append(f'Наличие опухоли оценено неуверенно (оценка {_pct(res["p_tumor"])}).')
    if 'classifier_tumor_no_contour' in fl and not F:
        s.append('Классификатор видит признаки опухоли, но контур не построен: просмотрите весь срез.')
    if 'contour_without_classifier' in fl and F:
        s.append('Контур построен, но классификатор признаков опухоли не видит: проверьте выделенный участок.')
    if any(f.get('band_ratio', 0) > 0.6 for f in F):
        s.append('Граница образования неуверенная: внешний край жёлтой полосы — где опухоль ещё может быть.')
    if Z:
        s.append('Есть зоны дополнительного просмотра (голубые рамки «?»).')
    S['Сомнения'] = '\n'.join(s) if s else 'Существенных сомнений система не отметила.'

    # 5. Ограничения (автотриггеры)
    lim = ['Система различает только глиому, менингиому и опухоль гипофиза; метастазы, лимфома, абсцесс и другие образования ею не распознаются.',
           'Контур соответствует видимой на постконтрастном T1 части образования; перифокальная зона (отёк) по T1 не оценивается — для неё нужен FLAIR/T2.',
           'Сторона указана относительно изображения, а не пациента: по JPG/PNG ориентация неизвестна.' if not meta.get('orientation_known') else '']
    if not px_mm:
        lim.append('Размеры в миллиметрах недоступны: в JPG/PNG нет размера пикселя (нужен DICOM).')
    if 'low_resolution' in fl:
        lim.append('Низкое разрешение снимка (≤ 256 пикс.): точность системы ниже.')
    if 'color_image' in fl:
        lim.append('Изображение цветное — возможно, это не исходный срез МРТ; результат может быть недостоверен.')
    seq = (meta.get('sequence') or '').lower().replace(' ', '')
    if seq and not (re.search(r't1.*(c|контраст|\+|gd)', seq) and 'без' not in seq):
        lim.append(f'Режим «{meta.get("sequence")}» вне области применения: система проверена на T1 с контрастом.')
    for x in meta.get('extra_limits', []):
        lim.append(x)
    S['Ограничения'] = '\n'.join(x for x in lim if x)

    # 6. Результат ИИ (справочно)
    S['Результат ИИ (справочно)'] = ('Оценки модели: ' + '; '.join(f'{c.lower()} {_pct(P[c])}' for c in CLASSES) +
                                    f'. Версия моделей: {res.get("model_version", "")}.')
    S['Подпись'] = ('Проект сформирован системой поддержки принятия врачебных решений (исследовательский прототип, не медицинское изделие). '
                    'Требует проверки и подписи врача.\nВрач: ____________________   Дата: ' + _dt.date.today().strftime('%d.%m.%Y'))
    text = '\n\n'.join(f'{k.upper()}\n{v}' for k, v in S.items())
    facts = dict(tumor=tumor, type=top if tumor else None, n_findings=len(F), halves=sorted({f['half'] for f in F}))
    return dict(sections=S, text=text, facts=facts)


# ------------------------------------------------------------- проверка «текст ↔ анализ» после правок врача

_TYPES = {'Глиома': r'глиом', 'Менингиома': r'менингиом', 'Опухоль гипофиза': r'(опухол\w*|аденом\w*)\s+гипофиз|гипофиз\w*\s+(опухол|аденом)'}
_NEG = r'(не\s+выявлен|не\s+обнаружен|не\s+найден|отсутств|без\s+признак|норм[аы]\b)'


def check_text(text, res, findings=None):
    """Предупреждения, если текст врача расходится с анализом. Решение остаётся за врачом — это только подсказка."""
    F = res['findings'] if findings is None else findings
    t = text.lower()
    concl = t.split('заключение', 1)[1] if 'заключение' in t else t
    concl = re.sub(r'\([^)]*\)', ' ', concl.split('сомнения', 1)[0])   # оговорку о закрытом наборе классов не считаем
    warn = []
    named = [c for c, rx in _TYPES.items() if re.search(rx, concl)]
    neg = re.search(_NEG, concl) is not None
    if F and neg and not named:
        warn.append(f'В заключении указано, что опухоли нет, а на изображении {len(F)} находк{"а" if len(F) == 1 else "и"}. '
                    'Удалите находки или исправьте текст.')
    if not F and not neg and named and not res['tumor']:
        warn.append('В заключении названа опухоль, но система её не видит и находок нет — проверьте или отметьте находку.')
    if res['tumor'] and named and res['top_type'] not in named:
        warn.append(f'В заключении: {", ".join(n.lower() for n in named)}; система считает наиболее вероятной: '
                    f'{res["top_type"].lower()} ({_pct(res["probs"][res["top_type"]])}). Это допустимо, если вы уверены.')
    halves = {f['half'] for f in F}
    for word, half in [('правой', 'правая'), ('левой', 'левая'), ('справа', 'правая'), ('слева', 'левая')]:
        if 'центр' in halves:
            break
        if re.search(word, t) and F and half not in halves:
            warn.append(f'В тексте упомянута сторона «{word}», а находки — {", ".join(_where(x) for x in halves)}.')
            break
    if re.search(r'полушари', t):
        warn.append('Упомянуто полушарие: по JPG/PNG ориентация неизвестна, сторона изображения может не совпадать с анатомической.')
    if re.search(r'\d+[.,]?\d*\s*(мм|см)\b', t) and not res.get('pixel_spacing_mm'):
        warn.append('В тексте есть размеры в мм/см, но размер пикселя неизвестен (JPG/PNG): проверьте источник размеров.')
    if re.search(r'глиобластом|rano|grade|степен\w* злокачеств', t):
        warn.append('Указан подтип или степень злокачественности: по одному срезу система это не оценивает.')
    return warn
