"""Геометрия полигональных масок — общий фундамент авто-QA, слияния и логики связи.

Все пороги в проекте задаются в пикселях для ОПОРНОЙ ширины кадра (по умолчанию
1280 px) и пересчитываются под фактическое разрешение через :func:`scale_px`.
Иначе один и тот же конфиг ведёт себя по-разному на 1280p и 1920p записях.

Модуль намеренно опирается только на OpenCV + NumPy (без shapely): маски приходят
от SAM 3 как полигоны, а все нужные операции выражаются через contour-API OpenCV.
"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

REFERENCE_WIDTH = 1280.0


def as_points(points: Iterable) -> np.ndarray:
    """Приводит полигон к массиву вершин (N, 2) типа int32."""
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    return pts


def scale_px(value_px: float, image_width: int, reference_width: float = REFERENCE_WIDTH) -> float:
    """Пересчитывает порог, заданный для опорной ширины, под фактический кадр.

    Args:
        value_px: Порог в пикселях при ширине кадра ``reference_width``.
        image_width: Фактическая ширина кадра.
        reference_width: Ширина, для которой откалиброван порог.

    Returns:
        Порог в пикселях текущего кадра.
    """
    return float(value_px) * float(image_width) / float(reference_width)


def polygon_area(points: np.ndarray) -> float:
    """Площадь полигона в px^2."""
    return float(abs(cv2.contourArea(as_points(points))))


def polygon_perimeter(points: np.ndarray) -> float:
    """Периметр замкнутого полигона в px."""
    return float(cv2.arcLength(as_points(points), True))


def mean_thickness(points: np.ndarray) -> float:
    """Средняя толщина вытянутого полигона: ``2*S/P``.

    Для длинной ленты (гофрошланг, кабель) отношение удвоенной площади к периметру
    близко к её ширине. Это ключевой признак отстройки ШИРОКОГО шланга от тонкого
    кабеля GPU — по маске он измеряется устойчиво, по bbox — нет.
    """
    return 2.0 * polygon_area(points) / max(polygon_perimeter(points), 1.0)


def elongation(points: np.ndarray) -> float:
    """Вытянутость по минимальному описанному прямоугольнику: длинная/короткая сторона."""
    (_, _), (w, h), _ = cv2.minAreaRect(as_points(points).astype(np.float32))
    long_side, short_side = max(w, h), min(w, h)
    return float(long_side / max(short_side, 1e-6))


def bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    """Ограничивающая рамка полигона в формате (x1, y1, x2, y2)."""
    x, y, w, h = cv2.boundingRect(as_points(points))
    return int(x), int(y), int(x + w), int(y + h)


def point_to_polygon_distance(point: tuple[float, float], polygon: np.ndarray) -> float:
    """Расстояние от точки до полигона (0, если точка внутри или на границе).

    В отличие от «минимума по вершинам», меряет расстояние до ближайшего РЕБРА,
    что важно для длинных сегментов шланга и крупного контура фюзеляжа.
    """
    poly = as_points(polygon)
    signed = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), True)
    return 0.0 if signed >= 0 else float(-signed)


def polygon_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Минимальное расстояние между двумя полигонами (0 при пересечении/касании)."""
    pa, pb = as_points(a), as_points(b)
    ax1, ay1, ax2, ay2 = bbox(pa)
    bx1, by1, bx2, by2 = bbox(pb)
    # Быстрый отсев: если рамки далеко, расстояние между полигонами не меньше зазора рамок.
    gap_x = max(0, max(bx1 - ax2, ax1 - bx2))
    gap_y = max(0, max(by1 - ay2, ay1 - by2))
    if gap_x or gap_y:
        lower_bound = float(np.hypot(gap_x, gap_y))
    else:
        lower_bound = 0.0
    best = float("inf")
    for pt in pa:
        best = min(best, point_to_polygon_distance((pt[0], pt[1]), pb))
        if best <= lower_bound:
            return best
    for pt in pb:
        best = min(best, point_to_polygon_distance((pt[0], pt[1]), pa))
        if best <= lower_bound:
            return best
    return best


def polygon_endpoints(points: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """Два «конца» вытянутого полигона — пара наиболее удалённых точек контура.

    Для гофрошланга это концы рукава: один уходит к юниту, другой — к ВС. Считаем
    по выпуклой оболочке (она мала), поэтому перебор пар дёшев.

    Note:
        Для сильно изогнутого (U-образного) шланга «максимально удалённая пара»
        может лечь не на физические концы. Для перронных сцен рукав тянется от
        тележки к борту почти по прямой, поэтому оценка устойчива; при появлении
        U-образных случаев здесь потребуется скелетизация.
    """
    pts = as_points(points)
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
    if len(hull) < 2:
        p = (int(pts[0][0]), int(pts[0][1]))
        return p, p
    d = np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(axis=2))
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return (int(hull[i][0]), int(hull[i][1])), (int(hull[j][0]), int(hull[j][1]))


def _raster_pair(a: np.ndarray, b: np.ndarray, max_side: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Растеризует два полигона в общий уменьшенный холст (для mask-IoU)."""
    pa, pb = as_points(a), as_points(b)
    x1 = min(pa[:, 0].min(), pb[:, 0].min())
    y1 = min(pa[:, 1].min(), pb[:, 1].min())
    x2 = max(pa[:, 0].max(), pb[:, 0].max())
    y2 = max(pa[:, 1].max(), pb[:, 1].max())
    w, h = max(int(x2 - x1) + 1, 1), max(int(y2 - y1) + 1, 1)
    scale = min(1.0, float(max_side) / float(max(w, h)))
    cw, ch = max(int(w * scale), 1), max(int(h * scale), 1)
    canvas_a = np.zeros((ch, cw), np.uint8)
    canvas_b = np.zeros((ch, cw), np.uint8)
    shift = np.array([x1, y1], np.float32)
    cv2.fillPoly(canvas_a, [((pa - shift) * scale).astype(np.int32)], 1)
    cv2.fillPoly(canvas_b, [((pb - shift) * scale).astype(np.int32)], 1)
    return canvas_a, canvas_b


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU двух полигонов по растеризации в общий холст (устойчиво к вогнутости)."""
    ca, cb = _raster_pair(a, b)
    inter = int(np.count_nonzero(ca & cb))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(ca | cb))
    return float(inter) / float(max(union, 1))


def centroid(points: np.ndarray) -> tuple[float, float]:
    """Центр масс полигона (при вырожденной площади — среднее вершин)."""
    pts = as_points(points)
    m = cv2.moments(pts)
    if abs(m["m00"]) < 1e-6:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def nms_polygons(
    instances: list[dict],
    iou_threshold: float = 0.5,
    score_key: str = "score",
    polygon_key: str = "points",
) -> tuple[list[dict], list[dict]]:
    """Non-maximum suppression по mask-IoU внутри одного класса.

    Args:
        instances: Список инстансов одного класса.
        iou_threshold: Порог подавления.
        score_key: Ключ уверенности (отсутствует -> 1.0).
        polygon_key: Ключ полигона.

    Returns:
        Кортеж (оставленные, подавленные).
    """
    order = sorted(instances, key=lambda d: float(d.get(score_key, 1.0)), reverse=True)
    kept: list[dict] = []
    dropped: list[dict] = []
    for cand in order:
        if any(polygon_iou(cand[polygon_key], k[polygon_key]) >= iou_threshold for k in kept):
            dropped.append(cand)
        else:
            kept.append(cand)
    return kept, dropped


def closest_polygon(
    target: np.ndarray, candidates: list[np.ndarray]
) -> tuple[Optional[int], float]:
    """Индекс ближайшего полигона из списка и расстояние до него."""
    best_i, best_d = None, float("inf")
    for i, cand in enumerate(candidates):
        d = polygon_distance(target, cand)
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d
