from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PublishedReference:
    hcv: int
    scv: float
    source: str = "Pillay & Özcan (2019), Table 18"
    reproduced: bool = False


PUBLISHED_TORONTO_AHH: dict[str, PublishedReference] = {
    "car-f-92": PublishedReference(0, 4.32),
    "car-f-91": PublishedReference(0, 5.16),
    "ear-f-83": PublishedReference(0, 36.52),
    "hec-s-92": PublishedReference(0, 11.87),
    "kfu-s-93": PublishedReference(0, 14.67),
    "lse-f-91": PublishedReference(0, 10.81),
    "pur-s-93": PublishedReference(0, 4.46),
    "rye-s-93": PublishedReference(0, 9.48),
    "sta-f-83": PublishedReference(0, 157.64),
    "tre-s-92": PublishedReference(0, 8.48),
    "uta-s-92": PublishedReference(0, 3.35),
    "ute-s-92": PublishedReference(0, 27.16),
    "yor-f-83": PublishedReference(0, 41.31),
}
