from __future__ import annotations

import re

MAX_SLUG_LENGTH = 200
SLUG_PATTERN = rf"^[a-z0-9_-]{{1,{MAX_SLUG_LENGTH}}}$"
SLUG_REGEX = re.compile(SLUG_PATTERN)


def validate_slug(value: str, *, name: str = "identifier") -> str:
    """Ensure the provided identifier conforms to the allowed slug pattern."""
    if not value or not SLUG_REGEX.fullmatch(value):
        raise ValueError(f"{name} must match pattern {SLUG_PATTERN}")
    return value


__all__ = ["MAX_SLUG_LENGTH", "SLUG_PATTERN", "validate_slug"]
