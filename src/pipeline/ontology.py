"""Загрузка и валидация онтологии задачи — типизированный доступ к конфигу.

Онтология (`configs/ontology.*.yaml`) — единственный источник правды конвейера:
какие концепты просить у учителя, что считать браком, что учить студенту и как
из детекций собрать факт подключения. Все остальные модули читают её ЧЕРЕЗ этот
загрузчик, чтобы опечатка в YAML падала один раз и понятным сообщением.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ClassSpec:
    """Описание одного класса онтологии."""

    class_id: int
    label: str
    confidence: float = 0.25
    train: bool = True
    stationary: bool = True
    positive_prompts: list[str] = field(default_factory=list)
    derived_from: Optional[str] = None


@dataclass
class PassSpec:
    """Один одноклассовый проход учителя (concept -> целевой класс)."""

    name: str
    prompt: str
    maps_to: str
    confidence: float = 0.25


class Ontology:
    """Типизированная обёртка над YAML-онтологией."""

    def __init__(self, data: dict[str, Any], source: Optional[str] = None) -> None:
        self.raw = data
        self.source = source
        self.classes: list[ClassSpec] = [
            ClassSpec(
                class_id=int(c["class_id"]),
                label=str(c["label"]),
                confidence=float(c.get("confidence", 0.25)),
                train=bool(c.get("train", True)),
                stationary=bool(c.get("stationary", True)),
                positive_prompts=list(c.get("positive_prompts", []) or []),
                derived_from=c.get("derived_from"),
            )
            for c in data.get("classes", [])
        ]
        self.passes: list[PassSpec] = [
            PassSpec(
                name=str(p["name"]),
                prompt=str(p["prompt"]),
                maps_to=str(p["maps_to"]),
                confidence=float(p.get("confidence", 0.25)),
            )
            for p in (data.get("autolabel", {}) or {}).get("passes", [])
        ]
        self._validate()

    # --- конструирование ---

    @classmethod
    def load(cls, path: str | Path) -> "Ontology":
        """Читает YAML-онтологию с диска."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {}, source=str(p))

    # --- доступ к секциям онтологии (сырые словари YAML, пустой при отсутствии) ---

    @property
    def autoqa(self) -> dict[str, Any]:
        return self.raw.get("autoqa", {}) or {}

    @property
    def link(self) -> dict[str, Any]:
        return self.raw.get("link", {}) or {}

    @property
    def connection(self) -> dict[str, Any]:
        return self.raw.get("connection", {}) or {}

    @property
    def events(self) -> dict[str, Any]:
        return self.raw.get("events", {}) or {}

    @property
    def person_split(self) -> dict[str, Any]:
        return self.raw.get("person_split", {}) or {}

    @property
    def student(self) -> dict[str, Any]:
        return self.raw.get("student", {}) or {}

    @property
    def acceptance(self) -> dict[str, Any]:
        return self.raw.get("acceptance", {}) or {}

    @property
    def reference_width(self) -> float:
        """Ширина кадра, для которой откалиброваны все пороги в пикселях."""
        return float(self.autoqa.get("reference_width", 1280))

    @property
    def proximity_px(self) -> float:
        """Дистанция «шланг рядом с юнитом» (опорная ширина кадра)."""
        derived = (self.raw.get("derived", {}) or {}).get("heater_complex", {}) or {}
        return float(
            self.autoqa.get("proximity", {}).get("hose_to_unit_px")
            or derived.get("proximity_px", 250)
        )

    # --- производные списки ---

    def trainable_labels(self) -> list[str]:
        """Классы студента в порядке class_id (индекс = id в YOLO-разметке)."""
        return [c.label for c in sorted(self.classes, key=lambda c: c.class_id) if c.train]

    def class_index(self) -> dict[str, int]:
        """Соответствие {label: индекс в YOLO} (сплошная нумерация обучаемых классов)."""
        return {label: i for i, label in enumerate(self.trainable_labels())}

    def stationary_labels(self) -> set[str]:
        """Классы, для которых осмысленно трек-голосование по времени."""
        return {c.label for c in self.classes if c.stationary}

    def confidence_of(self, label: str, default: float = 0.25) -> float:
        """Минимальная уверенность, ниже которой инстанс класса отбраковывается."""
        for c in self.classes:
            if c.label == label:
                return c.confidence
        return default

    def vest_ranges(self) -> list[tuple[list[int], list[int]]]:
        """HSV-диапазоны жилета в виде [(lower, upper), ...]."""
        ranges = self.person_split.get("hsv_ranges")
        if ranges:
            return [(list(r["lower"]), list(r["upper"])) for r in ranges]
        return [
            (list(self.person_split.get("hsv_lower", [30, 70, 80])),
             list(self.person_split.get("hsv_upper", [75, 255, 255])))
        ]

    # --- проверки ---

    def _validate(self) -> None:
        """Падает с понятным сообщением при рассогласовании онтологии."""
        if not self.classes:
            raise ValueError("Онтология без классов: секция `classes` пуста")
        ids = [c.class_id for c in self.classes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Дублирующиеся class_id в онтологии: {sorted(ids)}")
        labels = {c.label for c in self.classes}
        if len({c.label for c in self.classes}) != len(self.classes):
            raise ValueError("Дублирующиеся label в онтологии")
        # `person` — служебный концепт учителя: он не класс студента, а источник
        # staff/passenger. Поэтому в maps_to он допустим наравне с реальными классами.
        allowed = labels | {"person"}
        for p in self.passes:
            if p.maps_to not in allowed:
                raise ValueError(
                    f"Проход '{p.name}' отображается в неизвестный класс '{p.maps_to}'. "
                    f"Допустимо: {sorted(allowed)}"
                )
        for c in self.classes:
            if c.derived_from and c.derived_from not in allowed:
                raise ValueError(
                    f"Класс '{c.label}' выведен из неизвестного '{c.derived_from}'"
                )

    def __repr__(self) -> str:  # pragma: no cover — диагностика
        return (
            f"Ontology(source={self.source!r}, classes={len(self.classes)}, "
            f"passes={len(self.passes)}, trainable={self.trainable_labels()})"
        )
