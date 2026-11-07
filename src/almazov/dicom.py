"""DICOM-вход: размер пикселя (мм), анатомическая сторона по ориентации, обезличивание.

В заключение и журнал не попадает ничего из персональных данных пациента: имя, идентификатор, дата рождения, учреждение и т. п.
читаются только для того, чтобы их НЕ сохранять (список тегов — по базовому профилю обезличивания DICOM PS3.15, приказ РКН № 140).
"""
import numpy as np

PHI_TAGS = ['PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex', 'PatientAge', 'PatientAddress', 'OtherPatientIDs',
            'InstitutionName', 'InstitutionAddress', 'ReferringPhysicianName', 'PerformingPhysicianName', 'OperatorsName',
            'AccessionNumber', 'StudyID', 'StudyDate', 'StudyTime', 'SeriesDate', 'AcquisitionDate', 'ContentDate',
            'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'DeviceSerialNumber', 'StationName']


def read(path):
    """-> (изображение float32, сведения без персональных данных)."""
    import pydicom
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    if img.ndim == 3:                                   # многокадровый файл: берём средний срез
        img = img[img.shape[0] // 2]
    img = img * float(getattr(ds, 'RescaleSlope', 1) or 1) + float(getattr(ds, 'RescaleIntercept', 0) or 0)
    if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
        img = img.max() - img
    ps = getattr(ds, 'PixelSpacing', None)
    spacing = None
    if ps is not None and len(ps) == 2:
        r, c = float(ps[0]), float(ps[1])
        spacing = round((r + c) / 2, 4) if abs(r - c) / max(r, c) < 0.05 else None   # неквадратные пиксели — мм не даём
    iop = getattr(ds, 'ImageOrientationPatient', None)
    info = dict(
        pixel_spacing_mm=spacing,
        orientation=[float(v) for v in iop] if iop is not None else None,
        plane=plane_of(iop),
        modality=str(getattr(ds, 'Modality', '')),
        series=str(getattr(ds, 'SeriesDescription', ''))[:64],
        sequence_hint=sequence_hint(ds),
        phi_present=[t for t in PHI_TAGS if str(getattr(ds, t, '')).strip()],
    )
    return img, info


def plane_of(iop):
    if iop is None:
        return None
    row, col = np.array(iop[:3], float), np.array(iop[3:], float)
    n = np.abs(np.cross(row, col))
    return ['сагиттальная', 'корональная', 'аксиальная'][int(np.argmax(n))]


def sequence_hint(ds):
    s = ' '.join(str(getattr(ds, t, '')) for t in ['SeriesDescription', 'ProtocolName', 'SequenceName']).lower()
    contrast = bool(str(getattr(ds, 'ContrastBolusAgent', '')).strip()) or any(k in s for k in ['+c', 'post', 'gd', 'contrast', 'mprage+'])
    if 'flair' in s:
        return 'FLAIR'
    if 't2' in s:
        return 'T2'
    if 't1' in s or 'mprage' in s:
        return 'T1 с контрастом' if contrast else 'T1 (контраст не указан)'
    return None


def patient_side(iop, cx_frac):
    """Анатомическая сторона пациента для точки с относительной координатой x на изображении (0 — левый край).

    В DICOM ось +x пациента направлена к его ЛЕВОЙ стороне. Если направление строки изображения (первые 3 числа
    ImageOrientationPatient) идёт к +x, то правая часть изображения — левая сторона пациента (радиологическая конвенция).
    """
    if iop is None:
        return None
    rx = float(iop[0])
    if abs(rx) < 0.5:          # строка изображения идёт не вдоль оси лево-право (например, сагиттальный срез)
        return None
    right_of_image_is_patient_left = rx > 0
    on_right = cx_frac > 0.5
    if abs(cx_frac - 0.5) < 0.06:
        return 'по средней линии'
    return 'слева' if on_right == right_of_image_is_patient_left else 'справа'
