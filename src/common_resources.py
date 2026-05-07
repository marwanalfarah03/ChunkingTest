from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
COMMON_RESOURCES_ROOT = Path(
    os.environ.get(
        "PAP_COMMON_RESOURCES_DIR",
        str(WORKSPACE_ROOT.parent / "common_resources"),
    )
).expanduser()
LEGACY_DOCUMENTS_ROOT = PROJECT_ROOT / "documents"
LEGACY_CHUNKS_MAP_PATH = PROJECT_ROOT / "chunks_map.json"
COMMON_DOCUMENTS_ROOT = COMMON_RESOURCES_ROOT / "documents"
COMMON_CHUNKS_MAP_PATH = COMMON_RESOURCES_ROOT / "chunks_map.json"


def active_documents_root() -> Path:
    if COMMON_DOCUMENTS_ROOT.exists():
        return COMMON_DOCUMENTS_ROOT.resolve()
    return LEGACY_DOCUMENTS_ROOT.resolve()


def active_chunks_map_path() -> Path:
    if COMMON_CHUNKS_MAP_PATH.exists():
        return COMMON_CHUNKS_MAP_PATH.resolve()
    return LEGACY_CHUNKS_MAP_PATH.resolve()


DOCUMENTS_ROOT = active_documents_root()
CHUNKS_MAP_PATH = active_chunks_map_path()


def serialize_workspace_path(path: Path) -> str:
    resolved_path = path.resolve()
    for root in (COMMON_RESOURCES_ROOT, PROJECT_ROOT):
        try:
            return resolved_path.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return str(resolved_path)


def resolve_workspace_path(path_value: str | os.PathLike[str]) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()

    for root in (COMMON_RESOURCES_ROOT, PROJECT_ROOT):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (COMMON_RESOURCES_ROOT / candidate).resolve()


def resolve_documents_root(path_value: str | os.PathLike[str] | None = None) -> Path:
    if path_value is None:
        return DOCUMENTS_ROOT

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()

    if candidate == Path("documents"):
        return DOCUMENTS_ROOT

    for root in (PROJECT_ROOT, COMMON_RESOURCES_ROOT):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (PROJECT_ROOT / candidate).resolve()