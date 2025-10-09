from __future__ import annotations

from pathlib import Path
from typing import Union


class PathTraversalError(ValueError):
    """Raised when a joined path escapes the allowed base directory."""


def _is_relative_to(path: Path, base: Path) -> bool:
    """Backport of Path.is_relative_to for older Python versions."""
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def safe_join(
    base: Path,
    *parts: Union[str, Path],
    must_exist: bool = True,
) -> Path:
    """Safely join untrusted path components to a trusted base directory.

    Args:
        base: Root directory that joined paths must remain within.
        *parts: Additional path components (typically untrusted input).
        must_exist: When True (default) the resulting path must already exist.

    Returns:
        The resolved absolute path.

    Raises:
        FileNotFoundError: If the base path (or resulting path when must_exist=True)
            does not exist.
        PathTraversalError: If the resulting path would escape the base directory.
    """

    try:
        base_resolved = base.resolve(strict=True)
    except FileNotFoundError as exc:  # pragma: no cover - configuration error
        raise FileNotFoundError(f"Base path does not exist: {base}") from exc

    target = base_resolved.joinpath(*parts)

    try:
        resolved = target.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        if must_exist:
            raise
        # When the target does not yet exist we still need an absolute path.
        # resolve(strict=False) already returned as far as possible; reuse it.
        resolved = target.resolve(strict=False)

    if not _is_relative_to(resolved, base_resolved):
        raise PathTraversalError(f"{resolved} escapes base directory {base_resolved}")

    return resolved
