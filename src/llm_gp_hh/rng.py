from __future__ import annotations

import random
import secrets


def resolve_seed(seed: int | None) -> int:
    return secrets.randbits(64) if seed is None else int(seed)


def make_rng(seed: int) -> random.Random:
    return random.Random(seed)
