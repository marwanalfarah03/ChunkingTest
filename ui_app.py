from __future__ import annotations

import base64
import json
import io
import os
import re
import shutil
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote as _url_quote

from flask import Flask, Response, jsonify, render_template, request, send_file, url_for

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chunking
import classification
import header_inspection
import milvus_rag
import rag_txt_export
import section_json_extraction
from common_resources import CHUNKS_MAP_PATH as SHARED_CHUNKS_MAP_PATH, DOCUMENTS_ROOT as SHARED_DOCUMENTS_ROOT, resolve_workspace_path

DOCUMENTS_ROOT = SHARED_DOCUMENTS_ROOT
ALLOWED_EXTENSIONS = {".docx"}
DEFAULT_CLASSIFICATION_OUTPUT_NAME = "classification_output.json"
DEFAULT_INSPECTION_OUTPUT_NAME = "column_header_inspection.json"
DEFAULT_SECTION_JSON_OUTPUT_NAME = "section_json_output.json"
DOCUMENT_METADATA_NAME = "document_metadata.json"
CHUNKS_MAP_PATH = SHARED_CHUNKS_MAP_PATH

CL_TOKEN_RE = re.compile(r"CL\d{6}")
TB_TOKEN_RE = re.compile(r"<TB\d{6}>")
EM_TOKEN_RE = re.compile(r"<EM\d{6}>")
REFERENCE_TOKEN_RE = re.compile(r"(CL\d{6}|<TB\d{6}>|<EM\d{6}>)")
LEGACY_ASSET_PREFIX_RE = re.compile(r"^EM\d{6}_")
COLOR_TOKEN_RE = re.compile(r"\s*\[#([0-9A-Fa-f]{6})\]\s*")
RTL_CHAR_RE = re.compile(r"[\u0590-\u08ff\ufb1d-\ufdfd\ufe70-\ufefc]")
LTR_CHAR_RE = re.compile(r"[A-Za-z]")
FORMATTING_TAG_RE = re.compile(r"</?(?:strong|em|u)>|<HL:[^>]*>|</HL>", re.IGNORECASE)
INVISIBLE_CHARS_RE = re.compile(r"[​‌‍‎‏‪-‮⁠﻿]")
DOCUMENT_TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{8}_\d{6}(?:_\d+)?$")

WORKFLOW_HEADER_LEVELS = {
    "group_header": 1,
    "subgroup_header": 2,
    "sub_subgroup_header": 3,
    "subsubgroup_header": 3,
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024


@app.after_request
def disable_html_caching(response: Response) -> Response:
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("text/html"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

_jobs: dict[str, "UiJob"] = {}
_active_job_id: str | None = None
_jobs_lock = threading.RLock()

_milvus_jobs: dict[str, "MilvusJob"] = {}
_active_milvus_job_id: str | None = None
_milvus_jobs_lock = threading.RLock()
_department_catalog_lock = threading.RLock()


def static_asset_version(filename: str) -> int:
    try:
        return (PROJECT_ROOT / "static" / filename).stat().st_mtime_ns
    except OSError:
        return 0


@app.context_processor
def template_helpers() -> dict[str, Any]:
    return {
        "static_asset_url": lambda filename: url_for(
            "static",
            filename=filename,
            v=static_asset_version(filename),
        )
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LlmLogger:
    """Thread-safe JSONL log writer for LLM calls."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def log(self, entry: dict[str, Any]) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def doc_dir_to_id(dir_name: str) -> str:
    """Encode a document directory name to a URL-safe identifier."""
    return base64.urlsafe_b64encode(dir_name.encode("utf-8")).decode().rstrip("=")


def doc_id_to_dir_name(doc_id: str) -> str:
    """Decode a URL-safe identifier back to a directory name."""
    padding = (4 - len(doc_id) % 4) % 4
    return base64.urlsafe_b64decode(doc_id + "=" * padding).decode("utf-8")


def doc_id_to_path(doc_id: str) -> Path | None:
    """Resolve a doc_id to its absolute Path, validating it stays under DOCUMENTS_ROOT."""
    try:
        dir_name = doc_id_to_dir_name(doc_id)
    except Exception:
        return None
    candidate = (DOCUMENTS_ROOT / dir_name).resolve()
    try:
        candidate.relative_to(DOCUMENTS_ROOT.resolve())
    except ValueError:
        return None
    if not candidate.is_dir():
        return None
    return candidate


def history_artifact_dir(doc_path: Path) -> Path | None:
    """Return the chunk artifact directory for current and legacy history entries."""
    candidates = [doc_path / chunking.DEFAULT_OUTPUT_DIRNAME, doc_path]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        has_maps = (
            (candidate / chunking.DEFAULT_TABLE_MAP_NAME).exists()
            or (candidate / chunking.DEFAULT_CELL_MAP_NAME).exists()
            or (candidate / chunking.DEFAULT_ASSET_MAP_NAME).exists()
        )
        has_chunks = any(
            p.is_file() and p.suffix.lower() == ".txt" and "_nested_" not in p.name
            for p in candidate.iterdir()
        )
        if has_maps or has_chunks:
            return candidate
    return None


def history_source_docx(doc_path: Path) -> Path | None:
    preferred = doc_path / chunking.DEFAULT_SOURCE_DOCX_NAME
    if preferred.exists():
        return preferred
    docx_files = sorted(doc_path.glob("*.docx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return docx_files[0] if docx_files else None


def history_entry_mtime(doc_path: Path) -> float:
    paths = [
        history_source_docx(doc_path),
        doc_path / DEFAULT_SECTION_JSON_OUTPUT_NAME,
        doc_path / DEFAULT_CLASSIFICATION_OUTPUT_NAME,
        doc_path / "llm_calls.jsonl",
    ]
    artifact_dir = history_artifact_dir(doc_path)
    if artifact_dir is not None:
        paths.append(artifact_dir)
    existing = [p.stat().st_mtime for p in paths if p is not None and p.exists()]
    return max(existing) if existing else doc_path.stat().st_mtime


def history_asset_list(artifact_dir: Path | None) -> list[dict[str, Any]]:
    if artifact_dir is None:
        return []
    asset_map_path = artifact_dir / chunking.DEFAULT_ASSET_MAP_NAME
    if not asset_map_path.exists():
        return []
    try:
        asset_map = load_json_file(asset_map_path)
    except Exception:
        return []
    assets: list[dict[str, str]] = []
    for asset_id, asset in sorted(asset_map.items()):
        if not isinstance(asset, dict):
            continue
        assets.append({
            "id": str(asset_id),
            "kind": str(asset.get("kind") or ""),
            "name": asset_display_name(asset, str(asset_id)),
            "download_name": asset_download_name(asset, str(asset_id)),
            "content_type": str(asset.get("content_type") or ""),
        })
    return assets


def list_history_documents() -> list[dict[str, Any]]:
    """Return metadata for every document directory under DOCUMENTS_ROOT."""
    if not DOCUMENTS_ROOT.exists():
        return []
    chunks_map = load_chunks_map()
    docs: list[dict[str, Any]] = []
    entries = [p for p in DOCUMENTS_ROOT.iterdir() if p.is_dir()]
    for entry in sorted(entries, key=history_entry_mtime, reverse=True):
        if not entry.is_dir():
            continue
        source_docx = history_source_docx(entry)
        artifact_dir = history_artifact_dir(entry)
        has_history_artifacts = any(
            (entry / filename).exists()
            for filename in [
                DEFAULT_CLASSIFICATION_OUTPUT_NAME,
                DEFAULT_INSPECTION_OUTPUT_NAME,
                DEFAULT_SECTION_JSON_OUTPUT_NAME,
                "llm_calls.jsonl",
            ]
        )
        has_rag_txt = (entry / "rag_txt").is_dir()
        if source_docx is None and artifact_dir is None and not has_history_artifacts:
            continue
        if source_docx is None and artifact_dir == entry and not has_history_artifacts and not has_rag_txt:
            continue
        chunk_count = 0
        if artifact_dir is not None:
            chunk_count = sum(
                1 for p in artifact_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".txt" and "_nested_" not in p.name
            )
        has_llm_logs = (entry / "llm_calls.jsonl").exists()
        departments = get_document_departments(entry, chunks_map)
        docs.append({
            "id": doc_dir_to_id(entry.name),
            "name": entry.name,
            "created_at": datetime.fromtimestamp(history_entry_mtime(entry), tz=timezone.utc).isoformat(),
            "departments": departments,
            "has_docx": source_docx is not None,
            "has_classification": (entry / DEFAULT_CLASSIFICATION_OUTPUT_NAME).exists(),
            "has_inspection": (entry / DEFAULT_INSPECTION_OUTPUT_NAME).exists(),
            "has_section_json": (entry / DEFAULT_SECTION_JSON_OUTPUT_NAME).exists(),
            "has_llm_logs": has_llm_logs,
            "has_rag_txt": has_rag_txt,
            "has_assets": bool(history_asset_list(artifact_dir)),
            "chunk_count": chunk_count,
        })
    return docs


def text_quality(value: str) -> int:
    arabic = sum(1 for ch in value if "\u0600" <= ch <= "\u06ff")
    mojibake = sum(value.count(marker) for marker in ("Ø", "Ù", "Ã", "Â", "â"))
    replacement = value.count("�")
    return arabic * 4 - mojibake * 3 - replacement * 8


def repair_text(value: Any) -> str:
    """Repair UTF-8-as-Latin-1 mojibake visible in extracted Arabic docs."""
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    best = text
    for encoding in ("latin1", "cp1252"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if text_quality(candidate) > text_quality(best):
            best = candidate
    return best.replace("\uf0b7", "•").replace("\uf0a7", "▪").replace("\u00a0", " ")


def asset_display_name(asset: dict[str, Any], asset_id: str) -> str:
    display_name = repair_text(asset.get("display_name"))
    if display_name.strip():
        candidate = display_name
    else:
        original_name = repair_text(asset.get("original_name"))
        if original_name.strip():
            candidate = original_name
        else:
            stored_name = repair_text(asset.get("stored_name"))
            candidate = LEGACY_ASSET_PREFIX_RE.sub("", Path(stored_name).name) if stored_name else asset_id
    candidate = re.sub(r"[\r\n]+", " ", candidate).strip()
    return Path(candidate).name or asset_id


def asset_download_name(asset: dict[str, Any], asset_id: str) -> str:
    label = asset_display_name(asset, asset_id)
    if Path(label).suffix:
        return label

    for key in ("original_name", "stored_name"):
        source_name = repair_text(asset.get(key))
        suffix = Path(Path(source_name).name).suffix if source_name else ""
        if suffix:
            safe_label = re.sub(r'[\\/:*?"<>|]+', "_", label).strip() or asset_id
            return f"{safe_label}{suffix}"
    return re.sub(r'[\\/:*?"<>|]+', "_", label).strip() or asset_id


def build_download_content_disposition(filename: str, fallback_stem: str = "asset") -> str:
    suffix = Path(filename).suffix
    ascii_filename = re.sub(r"[^\x20-\x7e]", "_", Path(filename).name)
    ascii_filename = re.sub(r"\s+", " ", ascii_filename).strip(" .")
    ascii_stem = Path(ascii_filename).stem.strip(" ._")
    if not ascii_stem:
        ascii_filename = f"{fallback_stem}{suffix}"
    encoded_filename = _url_quote(filename, safe=" -._~()'!*")
    return f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{encoded_filename}'


def resolve_asset_file_path(artifact_dir: Path, asset: dict[str, Any]) -> Path:
    relative_path = Path(str(asset.get("relative_path") or ""))
    asset_path = (artifact_dir / relative_path).resolve()
    try:
        asset_path.relative_to(artifact_dir.resolve())
    except ValueError as exc:
        raise ValueError("Asset path is invalid.") from exc
    return asset_path


def send_asset_response(artifact_dir: Path, asset: dict[str, Any], asset_id: str) -> Response:
    asset_path = resolve_asset_file_path(artifact_dir, asset)
    if not asset_path.exists():
        raise FileNotFoundError(asset_path)

    content_type = str(asset.get("content_type") or "")
    kind = str(asset.get("kind") or "")
    send_kwargs: dict[str, Any] = {}
    if content_type:
        send_kwargs["mimetype"] = content_type
    if kind == "image" or content_type.startswith("image/"):
        return send_file(asset_path, **send_kwargs)

    download_name = asset_download_name(asset, asset_id)
    send_kwargs["as_attachment"] = True
    send_kwargs["download_name"] = Path(download_name).name or f"asset{Path(download_name).suffix}"
    response = send_file(asset_path, **send_kwargs)
    response.headers["Content-Disposition"] = build_download_content_disposition(download_name)
    return response


def dominant_direction(value: Any) -> str:
    fixed = FORMATTING_TAG_RE.sub("", repair_text(value))
    rtl_count = len(RTL_CHAR_RE.findall(fixed))
    ltr_count = len(LTR_CHAR_RE.findall(fixed))
    if rtl_count == 0:
        return "ltr"
    if ltr_count == 0:
        return "rtl"
    # Arabic SOP text often includes English system names (T24, BPM, KPI).
    # Keep the paragraph RTL unless Latin clearly dominates the content.
    if rtl_count >= max(2, int(ltr_count * 0.45)):
        return "rtl"
    for char in fixed:
        if RTL_CHAR_RE.match(char):
            return "rtl"
        if LTR_CHAR_RE.match(char):
            return "ltr"
    return "ltr"


def direction_class(value: Any) -> str:
    return f"text-{dominant_direction(value)}"


def color_text(hex_color: str | None) -> str:
    if not hex_color or not re.fullmatch(r"#[0-9A-Fa-f]{6}", hex_color):
        return "#243746"
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#ffffff" if luminance < 140 else "#243746"


def extract_display_color(value: Any) -> tuple[str, str | None]:
    fixed = repair_text(value)
    matches = COLOR_TOKEN_RE.findall(fixed)
    color = f"#{matches[-1]}" if matches else None
    label = COLOR_TOKEN_RE.sub(" ", fixed)
    label = re.sub(r"[ \t]+", " ", label)
    label = re.sub(r"\n{3,}", "\n\n", label)
    return label.strip(), color


def section_display(section_id: str) -> dict[str, str]:
    section = classification.SECTION_INDEX.get(section_id)
    if not section:
        return {"id": section_id, "label": "Unclassified", "short_label": "Unclassified", "description": "Needs review"}
    arabic = repair_text(section["arabic"])
    english = section["english"]
    return {
        "id": section_id,
        "label": f"{arabic} · {english}",
        "short_label": english,
        "arabic": arabic,
        "description": english,
    }


def section_options() -> list[dict[str, str]]:
    return [section_display(section["id"]) for section in classification.SECTIONS]


def normalize_department_names(raw_values: Any) -> list[str]:
    if isinstance(raw_values, str):
        values = [raw_values]
    elif isinstance(raw_values, (list, tuple, set)):
        values = list(raw_values)
    else:
        values = []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = re.sub(r"\s+", " ", repair_text(raw_value)).strip(" ,;")
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized


def parse_requested_departments(raw_value: Any) -> list[str]:
    payload = raw_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            payload = []
        else:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = [stripped]
    return normalize_department_names(payload)


def document_name_lookup_candidates(document_name: str) -> list[str]:
    normalized_name = re.sub(r"\s+", " ", repair_text(document_name)).strip()
    if not normalized_name:
        return []
    candidates = [normalized_name]
    stripped_name = DOCUMENT_TIMESTAMP_SUFFIX_RE.sub("", normalized_name)
    if stripped_name and stripped_name != normalized_name:
        candidates.append(stripped_name)
    return candidates


def document_metadata_path(doc_path: Path) -> Path:
    return doc_path / DOCUMENT_METADATA_NAME


def load_chunks_map() -> dict[str, list[str]]:
    with _department_catalog_lock:
        if not CHUNKS_MAP_PATH.exists():
            return {}
        try:
            raw_payload = json.loads(CHUNKS_MAP_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    if not isinstance(raw_payload, dict):
        return {}

    chunks_map: dict[str, list[str]] = {}
    for raw_document_name, raw_departments in raw_payload.items():
        document_name = re.sub(r"\s+", " ", repair_text(raw_document_name)).strip()
        if not document_name:
            continue
        chunks_map[document_name] = normalize_department_names(raw_departments)
    return chunks_map


def update_chunks_map_entry(document_name: str, departments: list[str]) -> None:
    normalized_name = re.sub(r"\s+", " ", repair_text(document_name)).strip()
    normalized_departments = normalize_department_names(departments)
    if not normalized_name or not normalized_departments:
        raise ValueError("Each document needs at least one department.")

    with _department_catalog_lock:
        if CHUNKS_MAP_PATH.exists():
            try:
                raw_payload = json.loads(CHUNKS_MAP_PATH.read_text(encoding="utf-8"))
            except Exception:
                raw_payload = {}
        else:
            raw_payload = {}
        if not isinstance(raw_payload, dict):
            raw_payload = {}
        raw_payload[normalized_name] = normalized_departments
        classification.write_json(CHUNKS_MAP_PATH, raw_payload)


def load_document_metadata(doc_path: Path) -> dict[str, Any]:
    metadata_path = document_metadata_path(doc_path)
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def get_document_departments(doc_path: Path, chunks_map: dict[str, list[str]] | None = None) -> list[str]:
    metadata = load_document_metadata(doc_path)
    metadata_departments = normalize_department_names(metadata.get("departments"))
    if metadata_departments:
        return metadata_departments

    source_map = chunks_map if chunks_map is not None else load_chunks_map()
    for candidate_name in document_name_lookup_candidates(doc_path.name):
        departments = source_map.get(candidate_name)
        if departments:
            return list(departments)
    return []


def save_document_departments(doc_path: Path, departments: Any) -> list[str]:
    normalized_departments = normalize_department_names(departments)
    if not normalized_departments:
        raise ValueError("Each document needs at least one department.")

    classification.write_json(
        document_metadata_path(doc_path),
        {
            "departments": normalized_departments,
            "updated_at": utc_timestamp(),
        },
    )
    update_chunks_map_entry(doc_path.name, normalized_departments)
    return normalized_departments


def department_catalog() -> dict[str, Any]:
    chunks_map = load_chunks_map()
    options: list[str] = []
    seen: set[str] = set()

    def add_departments(values: list[str]) -> None:
        for value in normalize_department_names(values):
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            options.append(value)

    for departments in chunks_map.values():
        add_departments(departments)

    history_departments: dict[str, list[str]] = {}
    if DOCUMENTS_ROOT.exists():
        for entry in sorted((p for p in DOCUMENTS_ROOT.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()):
            source_docx = history_source_docx(entry)
            artifact_dir = history_artifact_dir(entry)
            has_history_artifacts = any(
                (entry / filename).exists()
                for filename in [
                    DEFAULT_CLASSIFICATION_OUTPUT_NAME,
                    DEFAULT_INSPECTION_OUTPUT_NAME,
                    DEFAULT_SECTION_JSON_OUTPUT_NAME,
                    "llm_calls.jsonl",
                ]
            )
            has_rag_txt = (entry / "rag_txt").is_dir()
            if source_docx is None and artifact_dir == entry and not has_history_artifacts and not has_rag_txt:
                continue
            departments = get_document_departments(entry, chunks_map)
            history_departments[entry.name] = departments
            add_departments(departments)

    return {
        "options": options,
        "document_departments": chunks_map,
        "history_departments": history_departments,
    }


def safe_document_stem(filename: str) -> str:
    stem = Path(filename or "uploaded_document").stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" ._")
    return (stem or "uploaded_document")[:90]


def unique_document_dir(stem: str) -> Path:
    DOCUMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    base = DOCUMENTS_ROOT / stem
    if not base.exists():
        return base
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = DOCUMENTS_ROOT / f"{stem}_{suffix}"
    counter = 2
    while candidate.exists():
        candidate = DOCUMENTS_ROOT / f"{stem}_{suffix}_{counter}"
        counter += 1
    return candidate


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def workflow_header_type_for_level(level: int) -> str:
    if level <= 1:
        return "group_header"
    if level == 2:
        return "subgroup_header"
    return "sub_subgroup_header"


def hierarchy_entry_level(section_id: str, entry: Any) -> int | None:
    if not isinstance(entry, dict):
        return None
    entry_type = str(entry.get("type") or "")
    if entry_type == "subsection":
        try:
            return max(1, int(entry.get("level") or 1))
        except (TypeError, ValueError):
            return 1
    if section_id in {"SEC12", "SEC13"} and entry_type in WORKFLOW_HEADER_LEVELS:
        try:
            return max(1, int(entry.get("level") or WORKFLOW_HEADER_LEVELS[entry_type]))
        except (TypeError, ValueError):
            return WORKFLOW_HEADER_LEVELS[entry_type]
    return None


def assign_hierarchy_entry_level(section_id: str, entry: dict[str, Any], level: int) -> None:
    normalized_level = max(1, int(level))
    entry_type = str(entry.get("type") or "")
    if entry_type == "subsection":
        if normalized_level == 1:
            entry.pop("level", None)
        else:
            entry["level"] = normalized_level
        return
    if section_id in {"SEC12", "SEC13"} and entry_type in WORKFLOW_HEADER_LEVELS:
        entry["level"] = normalized_level
        entry["type"] = workflow_header_type_for_level(normalized_level)


def previous_hierarchy_level(section_id: str, entries: list[Any], entry_index: int) -> int:
    for candidate_index in range(entry_index - 1, -1, -1):
        level = hierarchy_entry_level(section_id, entries[candidate_index])
        if level is not None:
            return level
    return 0


def descendant_hierarchy_span(section_id: str, entries: list[Any], entry_index: int, root_level: int) -> range:
    end_index = entry_index + 1
    for candidate_index in range(entry_index + 1, len(entries)):
        candidate_level = hierarchy_entry_level(section_id, entries[candidate_index])
        if candidate_level is not None and candidate_level <= root_level:
            break
        end_index = candidate_index + 1
    return range(entry_index, end_index)


def shift_hierarchy_entries(section_id: str, entries: list[Any], entry_index: int, direction: str) -> tuple[list[Any], bool]:
    if direction not in {"left", "right"}:
        raise ValueError("Hierarchy direction must be 'left' or 'right'.")
    if entry_index < 0 or entry_index >= len(entries):
        raise IndexError("Hierarchy entry index is out of range.")

    updated_entries = [dict(entry) if isinstance(entry, dict) else entry for entry in entries]
    original_level = hierarchy_entry_level(section_id, updated_entries[entry_index])
    if original_level is None:
        raise ValueError("Only hierarchy headers can be shifted.")

    delta = -1 if direction == "left" else 1
    candidate_level = max(1, original_level + delta)
    if delta > 0:
        candidate_level = min(candidate_level, previous_hierarchy_level(section_id, updated_entries, entry_index) + 1)
    if candidate_level == original_level:
        return updated_entries, False

    subtree = descendant_hierarchy_span(section_id, updated_entries, entry_index, original_level)
    level_delta = candidate_level - original_level
    for candidate_index in subtree:
        entry = updated_entries[candidate_index]
        level = hierarchy_entry_level(section_id, entry)
        if level is None or not isinstance(entry, dict):
            continue
        assign_hierarchy_entry_level(section_id, entry, level + level_delta)
    return updated_entries, True


class DocumentRenderer:
    def __init__(
        self,
        *,
        job_id: str,
        document_path: Path,
        artifact_dir: Path,
        table_map: dict[str, Any],
        cell_map: dict[str, Any],
        asset_map: dict[str, Any],
        inspection_payload: dict[str, Any] | None = None,
        asset_url_prefix: str | None = None,
    ) -> None:
        self.job_id = job_id
        self.document_path = document_path
        self.artifact_dir = artifact_dir
        self.table_map = table_map
        self.cell_map = cell_map
        self.asset_map = asset_map
        self.inspection_payload = inspection_payload or {}
        self.asset_url_prefix = asset_url_prefix or f"/api/assets/{escape(job_id)}"

    def render_inline_text(self, value: Any) -> str:
        fixed = INVISIBLE_CHARS_RE.sub("", repair_text(value))
        if not fixed:
            return ""
        colorized = COLOR_TOKEN_RE.sub(
            lambda match: f'<span class="color-chip" style="--chip-color: #{match.group(1)}"></span>',
            fixed,
        )
        html = escape(colorized)
        for tag in ("strong", "em", "u"):
            html = re.sub(rf"&lt;({tag})&gt;", rf"<{tag}>", html, flags=re.IGNORECASE)
            html = re.sub(rf"&lt;/({tag})&gt;", rf"</{tag}>", html, flags=re.IGNORECASE)
        html = re.sub(
            r"&lt;HL:(.*?)&gt;",
            lambda m: f'<a href="{m.group(1).replace("&amp;", "&")}" target="_blank" rel="noopener noreferrer">',
            html,
            flags=re.IGNORECASE,
        )
        html = re.sub(r"&lt;/HL&gt;", "</a>", html, flags=re.IGNORECASE)
        html = html.replace("\n", "<br>")
        return re.sub(
            r"&lt;span class=&quot;color-chip&quot; style=&quot;--chip-color: (#[0-9A-Fa-f]{6})&quot;&gt;&lt;/span&gt;",
            r'<span class="color-chip" style="--chip-color: \1"></span>',
            html,
        )

    def direction_source(self, value: Any, *, depth: int = 0, seen_cells: set[str] | None = None) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(self.direction_source(item, depth=depth, seen_cells=seen_cells) for item in value.values())
        if isinstance(value, list):
            return " ".join(self.direction_source(item, depth=depth, seen_cells=seen_cells) for item in value)

        raw = str(value)
        if depth >= 2:
            return repair_text(raw)
        seen_cells = seen_cells or set()

        parts: list[str] = []
        for part in REFERENCE_TOKEN_RE.split(raw):
            if not part:
                continue
            if CL_TOKEN_RE.fullmatch(part):
                if part in seen_cells:
                    continue
                cell = self.cell_map.get(part)
                if isinstance(cell, dict):
                    parts.append(
                        self.direction_source(
                            cell.get("text") or "",
                            depth=depth + 1,
                            seen_cells={*seen_cells, part},
                        )
                    )
                else:
                    parts.append(part)
            elif TB_TOKEN_RE.fullmatch(part):
                table_id = part[1:-1]
                table_text = " ".join(str(cell.get("text") or "") for cell in self.cells_for_table(table_id))
                parts.append(self.direction_source(table_text, depth=depth + 1, seen_cells=seen_cells))
            elif EM_TOKEN_RE.fullmatch(part):
                asset = self.asset_map.get(part[1:-1])
                if isinstance(asset, dict):
                    parts.append(asset_display_name(asset, part[1:-1]))
                else:
                    parts.append(part)
            else:
                parts.append(repair_text(part))
        return " ".join(parts)

    def direction_attrs(self, value: Any) -> str:
        direction = dominant_direction(value)
        return f'dir="{direction}" class="{direction_class(value)}"'

    def render_rich_text(
        self,
        value: Any,
        *,
        seen_cells: set[str] | None = None,
        seen_tables: set[str] | None = None,
    ) -> str:
        if value is None:
            return '<span class="empty-value">Empty</span>'
        raw = str(value)
        if not raw:
            return ""
        seen_cells = seen_cells or set()
        seen_tables = seen_tables or set()
        parts: list[str] = []
        for part in REFERENCE_TOKEN_RE.split(raw):
            if not part:
                continue
            if CL_TOKEN_RE.fullmatch(part):
                parts.append(self.render_cell_reference(part, seen_cells=seen_cells, seen_tables=seen_tables))
            elif TB_TOKEN_RE.fullmatch(part):
                parts.append(self.render_table_reference(part[1:-1], seen_tables=seen_tables))
            elif EM_TOKEN_RE.fullmatch(part):
                parts.append(self.render_asset_reference(part[1:-1]))
            else:
                parts.append(self.render_inline_text(part))
        source = self.direction_source(raw, seen_cells=seen_cells)
        direction = dominant_direction(source)
        return f'<span class="rich-text text-{direction}" dir="{direction}">' + "".join(parts) + "</span>"

    def render_cell_reference(self, cell_id: str, *, seen_cells: set[str], seen_tables: set[str]) -> str:
        if cell_id in seen_cells:
            return ""
        cell = self.cell_map.get(cell_id)
        if not isinstance(cell, dict):
            return f'<span class="missing-ref">{escape(cell_id)}</span>'
        next_seen = set(seen_cells)
        next_seen.add(cell_id)
        text = str(cell.get("text") or "")
        html = self.render_rich_text(text, seen_cells=next_seen, seen_tables=seen_tables)
        for table_id in cell.get("nested_table_ids") or []:
            if f"<{table_id}>" not in text:
                html += self.render_table_reference(str(table_id), seen_tables=seen_tables)
        return html

    def render_asset_reference(self, asset_id: str) -> str:
        asset = self.asset_map.get(asset_id)
        if not isinstance(asset, dict):
            return f'<span class="asset-missing">Embedded file {escape(asset_id)}</span>'
        label = asset_display_name(asset, asset_id)
        download_name = asset_download_name(asset, asset_id)
        content_type = str(asset.get("content_type") or "")
        url = f"{self.asset_url_prefix}/{escape(asset_id)}"
        if content_type.startswith("image/"):
            return '<figure class="embedded-asset">' f'<img src="{url}" alt="{escape(label)}"></figure>'
        return f'<a class="embedded-file" href="{url}" download="{escape(download_name)}"><span>Embedded file</span><strong>{escape(label)}</strong></a>'

    def cells_for_table(self, table_id: str) -> list[dict[str, Any]]:
        cells: list[dict[str, Any]] = []
        for cell_id, payload in self.cell_map.items():
            if isinstance(payload, dict) and payload.get("table_id") == table_id:
                cells.append({"cell_id": cell_id, **payload})
        return sorted(cells, key=lambda item: (int(item.get("row") or 0), int(item.get("col") or 0), item["cell_id"]))

    def render_table_reference(self, table_id: str, *, seen_tables: set[str] | None = None) -> str:
        return self.render_table(table_id, seen_tables=seen_tables or set(), nested=True)

    def render_table(self, table_id: str, *, seen_tables: set[str] | None = None, nested: bool = False) -> str:
        seen_tables = seen_tables or set()
        if table_id in seen_tables:
            return ""
        table = self.table_map.get(table_id)
        if not isinstance(table, dict):
            return f'<span class="missing-ref">{escape(table_id)}</span>'
        next_seen_tables = set(seen_tables)
        next_seen_tables.add(table_id)
        cells = self.cells_for_table(table_id)
        row_count = int(table.get("row_count") or 0)
        rows: dict[int, list[dict[str, Any]]] = {row_index: [] for row_index in range(1, row_count + 1)}
        for cell in cells:
            rows.setdefault(int(cell.get("row") or 0), []).append(cell)
        table_text = " ".join(self.direction_source(cell.get("text") or "") for cell in cells)
        direction = dominant_direction(table_text)
        classes = f'{"doc-table nested-doc-table" if nested else "doc-table"} table-{direction}'
        html = [f'<div class="table-scroll"><table class="{classes}" dir="ltr" data-content-dir="{direction}">']
        for row_index in sorted(rows):
            row_cells = rows[row_index]
            if not row_cells:
                continue
            html.append("<tr>")
            for cell in row_cells:
                color = cell.get("color")
                styles = []
                if isinstance(color, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
                    styles.append(f"background-color: {color}")
                    styles.append(f"color: {color_text(color)}")
                style_attr = f' style="{"; ".join(styles)}"' if styles else ""
                cell_source = self.direction_source(cell.get("text") or "")
                cell_direction = dominant_direction(cell_source)
                class_attr = f' class="text-{cell_direction}"'
                rowspan = max(1, int(cell.get("rowspan") or 1))
                colspan = max(1, int(cell.get("colspan") or 1))
                cell_html = self.render_rich_text(cell.get("text") or "", seen_cells=set(), seen_tables=next_seen_tables)
                for nested_table_id in cell.get("nested_table_ids") or []:
                    if f"<{nested_table_id}>" not in str(cell.get("text") or ""):
                        cell_html += self.render_table(str(nested_table_id), seen_tables=next_seen_tables, nested=True)
                html.append(
                    f'<td rowspan="{rowspan}" colspan="{colspan}" dir="{cell_direction}"'
                    f'{class_attr}{style_attr}>{cell_html}</td>'
                )
            html.append("</tr>")
        html.append("</table></div>")
        return "".join(html)

    def render_chunk(self, txt_file_name: str, relative_path: str | None) -> str:
        for table_id, table in self.table_map.items():
            if isinstance(table, dict) and table.get("chunk_file_name") == txt_file_name:
                return self.render_table(table_id)
        content = ""
        if relative_path:
            path = resolve_workspace_path(relative_path)
            if path.exists():
                content = path.read_text(encoding="utf-8")
        if not content:
            return '<div class="empty-document">No preview text was found.</div>'
        direction = dominant_direction(self.direction_source(content))
        return f'<div class="text-chunk text-{direction}" dir="{direction}">{self.render_rich_text(content)}</div>'

    def render_heading(self, value: Any, fallback: str) -> str:
        label, color = extract_display_color(value or fallback)
        if not label:
            label = fallback
        style = ""
        if color:
            style = f' style="--section-color: {color}; --section-text: {color_text(color)}"'
        direction = dominant_direction(label)
        return f'<h2 class="section-title text-{direction}" dir="{direction}"{style}>{self.render_inline_text(label)}</h2>'

    def render_section_summary(self, value: Any, fallback: str) -> str:
        label, color = extract_display_color(value or fallback)
        if not label:
            label = fallback
        style = ""
        if color:
            style = f' style="--section-color: {color}; --section-text: {color_text(color)}"'
        direction = dominant_direction(label)
        return (
            f'<summary class="section-title section-summary collapsible-summary text-{direction}" '
            f'dir="{direction}"{style}><span class="collapse-marker" aria-hidden="true"></span>'
            f'<span class="summary-label">{self.render_inline_text(label)}</span></summary>'
        )

    def render_section_article(
        self,
        section_id: str,
        heading: Any,
        fallback_heading: str,
        body: str,
        *,
        collapsible: bool = False,
    ) -> str:
        if not collapsible:
            return '<article class="final-section">' f'{self.render_heading(heading, fallback_heading)}' f'{body}</article>'
        return (
            f'<article class="final-section final-section-collapsible" data-section-id="{escape(section_id)}">'
            '<details class="collapsible-node section-node" open>'
            f'{self.render_section_summary(heading, fallback_heading)}'
            f'<div class="section-collapsible-body">{body}</div>'
            '</details></article>'
        )

    def render_content_card(self, value: Any, label: str | None = None, nested: bool = False) -> str:
        label_html = f'<div class="content-label">{escape(label)}</div>' if label else ""
        nested_class = " nested-content" if nested else ""
        direction = dominant_direction(self.direction_source(value))
        return (
            f'<div class="content-card{nested_class} text-{direction}" dir="{direction}">'
            f'{label_html}{self.render_rich_text(value)}</div>'
        )

    def render_hierarchy_shift_controls(
        self,
        section_id: str,
        entries: list[Any],
        result_index: int,
        entry_index: int,
    ) -> str:
        level = hierarchy_entry_level(section_id, entries[entry_index]) or 1
        max_right_level = previous_hierarchy_level(section_id, entries, entry_index) + 1
        can_left = level > 1
        can_right = level < max_right_level
        return (
            '<span class="hierarchy-shift-controls" role="group" aria-label="Adjust hierarchy depth">'
            f'<button class="hierarchy-shift-button" type="button" data-hierarchy-shift="left" '
            f'data-result-index="{result_index}" data-section-id="{escape(section_id)}" '
            f'data-entry-index="{entry_index}" aria-label="Move this heading left"'
            f'{"" if can_left else " disabled"}><span class="hierarchy-shift-icon" aria-hidden="true">&#8676;</span></button>'
            f'<button class="hierarchy-shift-button" type="button" data-hierarchy-shift="right" '
            f'data-result-index="{result_index}" data-section-id="{escape(section_id)}" '
            f'data-entry-index="{entry_index}" aria-label="Move this heading right"'
            f'{"" if can_right else " disabled"}><span class="hierarchy-shift-icon" aria-hidden="true">&#8677;</span></button>'
            '</span>'
        )

    def render_simple_rows(self, rows: list[Any]) -> str:
        if not rows:
            return ""
        table_source = self.direction_source(rows)
        table_direction = dominant_direction(table_source)
        html = [
            f'<div class="table-scroll"><table class="doc-table generated-table table-{table_direction}" '
            f'dir="ltr" data-content-dir="{table_direction}">'
        ]
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("cells"), list):
                cells = row["cells"]
            elif isinstance(row, list):
                cells = row
            elif isinstance(row, dict):
                cells = [value for key, value in row.items() if key != "type"]
            else:
                cells = [row]
            html.append("<tr>")
            for cell in cells:
                direction = dominant_direction(self.direction_source(cell))
                html.append(f'<td class="text-{direction}" dir="{direction}">{self.render_rich_text(cell)}</td>')
            html.append("</tr>")
        html.append("</table></div>")
        return "".join(html)

    def render_matrix_table(self, data: dict[str, Any]) -> str:
        rows = data.get("rows") or []
        columns = data.get("columns") or []
        if not columns and rows and isinstance(rows[0], dict):
            columns = [key for key in rows[0].keys() if key != "type"]
        if not columns:
            return self.render_simple_rows(rows if isinstance(rows, list) else [])
        direction = dominant_direction(" ".join(repair_text(column) for column in columns))
        html = [
            f'<div class="table-scroll"><table class="doc-table generated-table table-{direction}" '
            f'dir="ltr" data-content-dir="{direction}"><thead><tr>'
        ]
        for column in columns:
            column_direction = dominant_direction(column)
            html.append(f'<th class="text-{column_direction}" dir="{column_direction}">{self.render_inline_text(column)}</th>')
        html.append("</tr></thead><tbody>")
        for row in rows:
            html.append("<tr>")
            for column in columns:
                value = row.get(column) if isinstance(row, dict) else None
                value_direction = dominant_direction(self.direction_source(value))
                html.append(f'<td class="text-{value_direction}" dir="{value_direction}">{self.render_rich_text(value)}</td>')
            html.append("</tr>")
        html.append("</tbody></table></div>")
        return "".join(html)

    def render_hierarchy_entries(self, section_id: str, entries: list[Any], result_index: int) -> str:
        html: list[str] = ['<div class="hierarchy-list hierarchy-collapsible">']
        open_levels: list[int] = []

        def close_to_level(level: int) -> None:
            while open_levels and open_levels[-1] >= level:
                html.append("</div></details>")
                open_levels.pop()

        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                html.append(self.render_content_card(entry, nested=bool(open_levels)))
                continue
            entry_type = str(entry.get("type") or "")
            header_level = hierarchy_entry_level(section_id, entry)
            if entry_type == "subsection" and header_level is not None:
                close_to_level(header_level)
                heading = entry.get("heading") or "Subsection"
                heading_label, heading_color = extract_display_color(heading)
                style = (
                    f' style="--section-color: {heading_color}; --section-text: {color_text(heading_color)}"'
                    if heading_color
                    else ""
                )
                heading_direction = dominant_direction(heading_label)
                controls = self.render_hierarchy_shift_controls(section_id, entries, result_index, entry_index)
                html.append(
                    f'<details class="collapsible-node hierarchy-node hierarchy-subsection-node" '
                    f'data-hierarchy-level="{header_level}" open{style}>'
                    f'<summary class="collapsible-summary hierarchy-subsection-summary text-{heading_direction}" '
                    f'dir="{heading_direction}"><span class="collapse-marker" aria-hidden="true"></span>'
                    f'<span class="summary-label">{self.render_inline_text(heading_label)}</span>'
                    f'{controls}</summary>'
                    '<div class="collapsible-body hierarchy-node-body">'
                )
                if "value" in entry:
                    html.append(self.render_content_card(entry.get("value"), nested=True))
                for value in entry.get("values") or []:
                    html.append(self.render_content_card(value, nested=True))
                open_levels.append(header_level)
            elif entry_type == "content":
                html.append(self.render_content_card(entry.get("value"), nested=bool(open_levels)))
            else:
                label = repair_text(entry_type or "Item")
                html.append(self.render_content_card(entry.get("value") or entry, label=label, nested=bool(open_levels)))
        close_to_level(1)
        html.append("</div>")
        return "".join(html)

    def workflow_columns(self, data: dict[str, Any], entries: list[Any]) -> list[str]:
        columns = data.get("columns") or []
        if len(columns) >= 2:
            return list(columns)
        for entry in entries:
            if isinstance(entry, dict) and entry.get("type") == "step":
                keys = [key for key in entry.keys() if key != "type"]
                if len(keys) >= 2:
                    return list(keys)
        return ["Step", "Owner"]

    def render_workflow_entries(self, section_id: str, data: dict[str, Any], result_index: int) -> str:
        entries = data.get("entries") or []
        columns = self.workflow_columns(data, entries)
        html: list[str] = ['<div class="workflow-view workflow-collapsible">']
        table_open = False
        open_levels: list[int] = []
        pending_steps: list[dict[str, Any]] = []

        def compute_rowspans(steps: list[dict[str, Any]]) -> list[list[int]]:
            # grid[row][col]: >1 = rowspan count, 0 = covered by cell above, 1 = normal
            n = len(steps)
            nc = len(columns)
            grid = [[1] * nc for _ in range(n)]
            for col_idx, col in enumerate(columns):
                row_idx = 0
                while row_idx < n:
                    val = steps[row_idx].get(col)
                    if val and isinstance(val, str) and CL_TOKEN_RE.match(val):
                        span = 1
                        while (row_idx + span < n
                               and steps[row_idx + span].get(col) == val):
                            span += 1
                        if span > 1:
                            grid[row_idx][col_idx] = span
                            for k in range(1, span):
                                grid[row_idx + k][col_idx] = 0
                        row_idx += span
                    else:
                        row_idx += 1
            return grid

        def fill_merged_cells(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Propagate CL IDs into empty cells that the document's rowspan covers.

            The LLM outputs "" for positions covered by a rowspan.  We use the
            cell map's rowspan field to know exactly how many rows each CL ID
            spans, then fill forward so compute_rowspans can detect the merge.
            """
            filled = [dict(step) for step in steps]
            # carry[col_idx] = (cl_id, remaining_rows)
            carry: dict[int, tuple[str, int]] = {}
            for row_idx, step in enumerate(filled):
                for col_idx, col in enumerate(columns):
                    val = step.get(col)
                    if val and isinstance(val, str) and CL_TOKEN_RE.match(val):
                        rowspan = int((self.cell_map.get(val) or {}).get("rowspan") or 1)
                        if rowspan > 1:
                            carry[col_idx] = (val, rowspan - 1)
                        else:
                            carry.pop(col_idx, None)
                    elif not val and col_idx in carry:
                        cl_id, remaining = carry[col_idx]
                        filled[row_idx][col] = cl_id
                        if remaining > 1:
                            carry[col_idx] = (cl_id, remaining - 1)
                        else:
                            del carry[col_idx]
                    else:
                        carry.pop(col_idx, None)
            return filled

        def flush_steps() -> None:
            if not pending_steps:
                return
            steps = fill_merged_cells(pending_steps)
            grid = compute_rowspans(steps)
            for row_idx, step in enumerate(steps):
                fallback_values = [v for k, v in step.items() if k != "type"]
                html.append("<tr>")
                for col_idx, col in enumerate(columns):
                    rs = grid[row_idx][col_idx]
                    if rs == 0:
                        continue
                    value = step.get(col)  # step is already from filled list
                    if not value and col_idx < len(fallback_values):
                        value = fallback_values[col_idx] or None
                    val_direction = dominant_direction(self.direction_source(value))
                    if col_idx == 0:
                        css_cls = "workflow-step"
                    elif col_idx == len(columns) - 1 and len(columns) == 2:
                        css_cls = "workflow-owner"
                    else:
                        css_cls = "workflow-cell"
                    rowspan_attr = f' rowspan="{rs}"' if rs > 1 else ""
                    html.append(
                        f'<td class="{css_cls} text-{val_direction}" dir="{val_direction}"{rowspan_attr}>'
                        f'{self.render_rich_text(value)}</td>'
                    )
                html.append("</tr>")
            pending_steps.clear()

        def close_table() -> None:
            nonlocal table_open
            if table_open:
                flush_steps()
                html.append("</tbody></table></div>")
                table_open = False

        def close_to_level(level: int) -> None:
            close_table()
            while open_levels and open_levels[-1] >= level:
                html.append("</div></details>")
                open_levels.pop()

        def open_node(level: int, heading: Any, entry_index: int) -> None:
            heading_label, heading_color = extract_display_color(heading or "")
            style = (
                f' style="--group-color: {heading_color}; --group-text: {color_text(heading_color)}"'
                if heading_color
                else ""
            )
            heading_direction = dominant_direction(heading_label)
            node_class = "workflow-group-node" if level == 1 else "workflow-subgroup-node"
            summary_class = "workflow-group-summary" if level == 1 else "workflow-subgroup-summary"
            if level >= 3:
                node_class += " workflow-deep-node"
                summary_class += " workflow-deep-summary"
            controls = self.render_hierarchy_shift_controls(section_id, entries, result_index, entry_index)
            html.append(
                f'<details class="collapsible-node workflow-node {node_class}" '
                f'data-hierarchy-level="{level}" open{style}>'
                f'<summary class="collapsible-summary {summary_class} text-{heading_direction}" '
                f'dir="{heading_direction}"><span class="collapse-marker" aria-hidden="true"></span>'
                f'<span class="summary-label">{self.render_inline_text(heading_label)}</span>'
                f'{controls}</summary>'
                '<div class="collapsible-body workflow-node-body">'
            )
            open_levels.append(level)

        def open_table() -> None:
            nonlocal table_open
            if table_open:
                return
            table_direction = dominant_direction(" ".join(repair_text(column) for column in columns))
            html.append(
                f'<div class="table-scroll"><table class="workflow-table table-{table_direction}" '
                f'dir="ltr" data-content-dir="{table_direction}"><thead><tr>'
            )
            for column in columns:
                col_direction = dominant_direction(column)
                html.append(f'<th class="text-{col_direction}" dir="{col_direction}">{self.render_inline_text(column)}</th>')
            html.append("</tr></thead><tbody>")
            table_open = True

        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                close_table()
                html.append(self.render_content_card(entry))
                continue
            entry_type = str(entry.get("type") or "")
            header_level = hierarchy_entry_level(section_id, entry)
            if entry_type in WORKFLOW_HEADER_LEVELS and header_level is not None:
                close_to_level(header_level)
                open_node(header_level, entry.get("heading"), entry_index)
            elif entry_type == "content":
                close_table()
                html.append(self.render_content_card(entry.get("value")))
            elif entry_type == "step":
                open_table()
                pending_steps.append(entry)
            else:
                close_table()
                html.append(self.render_content_card(entry))
        close_table()
        close_to_level(1)
        html.append("</div>")
        return "".join(html)

    def render_sec99_item(self, item: dict[str, Any]) -> str:
        fmt = item.get("format") or "other"
        if fmt == "not_found":
            return ""
        if fmt == "empty":
            return '<div class="empty-document">Empty table</div>'
        heading = item.get("section_heading")
        parts = ['<section class="other-block">']
        if heading:
            label, color = extract_display_color(heading)
            style = f' style="--section-color: {color}; --section-text: {color_text(color)}"' if color else ""
            direction = dominant_direction(label)
            parts.append(f'<h3 class="other-title text-{direction}" dir="{direction}"{style}>{self.render_inline_text(label)}</h3>')
        if fmt == "titled_item":
            parts.append(self.render_content_card(item.get("content")))
        elif fmt == "matrix":
            parts.append(self.render_matrix_table(item))
        elif fmt == "other":
            parts.append(self.render_simple_rows(item.get("rows") or []))
        else:
            parts.append(self.render_generic_value(item))
        parts.append("</section>")
        return "".join(parts)

    def render_generic_value(self, value: Any) -> str:
        if isinstance(value, dict):
            rows = []
            for key, item in value.items():
                if key in {"section_id", "header_cell_ids"}:
                    continue
                key_direction = dominant_direction(key)
                item_direction = dominant_direction(self.direction_source(item))
                rows.append(
                    f'<div class="generic-row text-{item_direction}" dir="{item_direction}">'
                    f'<strong class="text-{key_direction}" dir="{key_direction}">{self.render_inline_text(key)}</strong>'
                    f'<div>{self.render_generic_value(item)}</div></div>'
                )
            return '<div class="generic-block">' + "".join(rows) + "</div>"
        if isinstance(value, list):
            return '<div class="generic-list">' + "".join(self.render_generic_value(item) for item in value) + "</div>"
        return self.render_rich_text(value)

    def render_section_payload(self, section_id: str, payload: Any, result_index: int) -> str:
        section_meta = section_display(section_id)
        fallback_heading = section_meta["label"]
        if isinstance(payload, list):
            body = "".join(self.render_sec99_item(item) for item in payload if isinstance(item, dict))
            if not body:
                body = '<div class="empty-document">No displayable content was produced for this section.</div>'
            return self.render_section_article(section_id, fallback_heading, fallback_heading, body)
        if not isinstance(payload, dict):
            return self.render_section_article(
                section_id,
                fallback_heading,
                fallback_heading,
                self.render_content_card(payload),
            )
        if payload.get("status") == "not_found":
            return ""
        heading = payload.get("section_heading") or fallback_heading
        entries = payload.get("entries")
        rows = payload.get("rows")
        columns = payload.get("columns")
        has_hierarchy_entries = isinstance(entries, list) and section_id in {"SEC11", "SEC12", "SEC13", "SEC18"}
        if isinstance(entries, list) and any(isinstance(item, dict) and item.get("type") in {"group_header", "subgroup_header", "sub_subgroup_header", "subsubgroup_header", "step"} for item in entries):
            body = self.render_workflow_entries(section_id, payload, result_index)
        elif isinstance(entries, list):
            body = self.render_hierarchy_entries(section_id, entries, result_index)
        elif rows is not None and columns:
            body = self.render_matrix_table(payload)
        elif rows is not None:
            body = self.render_simple_rows(rows if isinstance(rows, list) else [])
        elif "content" in payload:
            body = self.render_content_card(payload.get("content"))
        else:
            body = self.render_generic_value(payload)
        return self.render_section_article(
            section_id,
            heading,
            fallback_heading,
            body,
            collapsible=has_hierarchy_entries,
        )

    def table_ids_for_chunk_names(self, chunk_names: list[str]) -> list[str]:
        chunk_lookup: dict[str, str] = {}
        for table_id, table in self.table_map.items():
            if isinstance(table, dict) and table.get("chunk_file_name"):
                chunk_lookup[str(table["chunk_file_name"])] = str(table_id)

        table_ids: list[str] = []
        seen: set[str] = set()
        for chunk_name in chunk_names:
            table_id = chunk_lookup.get(str(chunk_name))
            if table_id and table_id not in seen:
                table_ids.append(table_id)
                seen.add(table_id)
        return table_ids

    def table_rows_for_ids(self, table_ids: list[str]) -> list[dict[str, Any]]:
        table_rows: list[dict[str, Any]] = []
        for table_id in table_ids:
            table = self.table_map.get(table_id)
            if not isinstance(table, dict):
                continue
            column_count = int(table.get("column_count") or 0)
            grouped_rows: dict[int, list[dict[str, Any]]] = {}
            for cell in self.cells_for_table(table_id):
                row_index = int(cell.get("row") or 0)
                if row_index <= 0:
                    continue
                grouped_rows.setdefault(row_index, []).append(
                    {
                        **cell,
                        "row": row_index,
                        "col": int(cell.get("col") or 0),
                        "rowspan": int(cell.get("rowspan") or 1),
                        "colspan": int(cell.get("colspan") or 1),
                    }
                )

            for row_index in sorted(grouped_rows):
                cells = sorted(grouped_rows[row_index], key=lambda c: (int(c.get("col") or 0), c["cell_id"]))
                first_colspan = int(cells[0].get("colspan") or 1) if cells else 1
                table_rows.append(
                    {
                        "table_id": table_id,
                        "row_index": row_index,
                        "column_count": column_count,
                        "cells": cells,
                        "is_full_width": len(cells) == 1 and first_colspan >= max(column_count, 1),
                    }
                )
        return table_rows

    def cell_display_value(self, cell: dict[str, Any]) -> str:
        display_text = str(cell.get("display_text") or "").strip()
        cell_id = str(cell.get("cell_id") or "")
        if display_text and display_text != cell_id:
            return display_text
        return str(cell.get("text") or "").strip()

    def cleaned_label(self, value: Any) -> str:
        label, _ = extract_display_color(value)
        without_tags = FORMATTING_TAG_RE.sub("", repair_text(label))
        without_invisible = INVISIBLE_CHARS_RE.sub("", without_tags)
        return re.sub(r"\s+", " ", without_invisible).strip()

    def normalized_label(self, value: Any) -> str:
        return re.sub(r"[^\w]+", "", self.cleaned_label(value), flags=re.UNICODE).casefold()

    def inspection_result_for(self, section_id: str, chunk_names: list[str]) -> dict[str, Any] | None:
        chunk_set = {str(name) for name in chunk_names}
        for result in self.inspection_payload.get("results") or []:
            if not isinstance(result, dict):
                continue
            if result.get("status") == "skipped" or result.get("inspected_section_id") != section_id:
                continue
            source_files = set(result.get("source_chunk_file_names") or [])
            group_files = set(result.get("table_group_source_chunk_file_names") or [])
            if (source_files | group_files) & chunk_set:
                return result
        return None

    def canonical_columns_for_section(self, section_id: str, column_count: int) -> list[str]:
        schemas = getattr(header_inspection, "HEADER_SECTION_SCHEMAS", {})
        schema = schemas.get(section_id) if isinstance(schemas, dict) else None
        for order in getattr(schema, "canonical_orders", ()) or ():
            if not column_count or len(order) == column_count:
                return [str(column) for column in order]
        return []

    def column_context(
        self,
        section_id: str,
        chunk_names: list[str],
        column_count: int,
    ) -> tuple[list[str], list[str]]:
        inspection_result = self.inspection_result_for(section_id, chunk_names)
        resolution = inspection_result.get("resolution") if isinstance(inspection_result, dict) else None
        columns: list[str] = []
        header_cell_ids: list[str] = []
        if isinstance(resolution, dict):
            columns = [str(column) for column in resolution.get("valid_column_order") or [] if str(column).strip()]
            header_cell_ids = [str(cell_id) for cell_id in resolution.get("actual_header_cell_ids") or []]
        if not columns:
            columns = self.canonical_columns_for_section(section_id, column_count)
        return columns, header_cell_ids

    def row_is_column_header(
        self,
        row: dict[str, Any],
        columns: list[str],
        header_cell_ids: list[str],
    ) -> bool:
        row_cell_ids = {str(cell.get("cell_id") or "") for cell in row.get("cells") or []}
        header_id_set = set(header_cell_ids)
        if header_id_set and header_id_set.issubset(row_cell_ids):
            return True
        if not columns:
            return False
        cells = sorted(row.get("cells") or [], key=lambda c: (int(c.get("col") or 0), str(c.get("cell_id") or "")))
        if len(cells) < len(columns):
            return False
        candidate_labels = [self.normalized_label(self.cell_display_value(cell)) for cell in cells[: len(columns)]]
        expected_labels = [self.normalized_label(column) for column in columns]
        return all(candidate and candidate == expected for candidate, expected in zip(candidate_labels, expected_labels))

    def row_looks_like_section_heading(self, section_id: str, value: Any) -> bool:
        text = self.cleaned_label(value)
        meta = section_display(section_id)
        needles = [
            self.cleaned_label(meta.get("arabic") or ""),
            self.cleaned_label(meta.get("short_label") or ""),
        ]
        return any(needle and needle.casefold() in text.casefold() for needle in needles)

    def workflow_heading_type(self, value: Any) -> str:
        _, color = extract_display_color(value)
        if color and color.upper() in {"#222A35", "#233744", "#243744", "#253744"}:
            return "group_header"
        return "subgroup_header"

    def step_entry_for_row(self, row: dict[str, Any], columns: list[str]) -> dict[str, str] | None:
        cells = sorted(row.get("cells") or [], key=lambda c: (int(c.get("col") or 0), str(c.get("cell_id") or "")))
        if not cells or not any(self.cleaned_label(self.cell_display_value(cell)) for cell in cells):
            return None
        entry: dict[str, str] = {"type": "step"}
        for index, column in enumerate(columns):
            if index < len(cells):
                entry[column] = str(cells[index]["cell_id"])
        return entry if len(entry) > 1 else None

    def infer_columns_from_rows(self, rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        for row in rows:
            if row.get("is_full_width"):
                continue
            cells = sorted(row.get("cells") or [], key=lambda c: (int(c.get("col") or 0), str(c.get("cell_id") or "")))
            labels = [self.cleaned_label(self.cell_display_value(cell)) or str(cell.get("cell_id") or "") for cell in cells]
            if labels:
                return labels, [str(cell["cell_id"]) for cell in cells]
        return [], []

    def build_workflow_table_section_json(
        self,
        section_id: str,
        rows: list[dict[str, Any]],
        columns: list[str],
        header_cell_ids: list[str],
    ) -> dict[str, Any] | None:
        if len(columns) < 2:
            inferred_columns, inferred_header_ids = self.infer_columns_from_rows(rows)
            columns = inferred_columns
            header_cell_ids = header_cell_ids or inferred_header_ids
        if len(columns) < 2:
            return None

        section_heading: str | None = None
        entries: list[dict[str, Any]] = []
        for row in rows:
            if self.row_is_column_header(row, columns, header_cell_ids):
                continue
            cells = sorted(row.get("cells") or [], key=lambda c: (int(c.get("col") or 0), str(c.get("cell_id") or "")))
            if not cells:
                continue
            if row.get("is_full_width") or len(cells) == 1:
                heading = self.cell_display_value(cells[0])
                if not self.cleaned_label(heading):
                    continue
                if section_heading is None and self.row_looks_like_section_heading(section_id, heading):
                    section_heading = heading
                    continue
                entries.append({"type": self.workflow_heading_type(heading), "heading": heading})
                continue
            step_entry = self.step_entry_for_row(row, columns)
            if step_entry:
                entries.append(step_entry)

        if not entries:
            return None
        payload: dict[str, Any] = {
            "section_id": section_id,
            "columns": columns,
            "header_cell_ids": header_cell_ids,
            "entries": entries,
        }
        if section_heading:
            payload["section_heading"] = section_heading
        return payload

    def build_matrix_table_section_json(
        self,
        section_id: str,
        rows: list[dict[str, Any]],
        columns: list[str],
        header_cell_ids: list[str],
    ) -> dict[str, Any] | None:
        if not columns:
            columns, inferred_header_ids = self.infer_columns_from_rows(rows)
            header_cell_ids = header_cell_ids or inferred_header_ids
        if not columns:
            return None

        section_heading: str | None = None
        data_rows: list[dict[str, str]] = []
        for row in rows:
            if self.row_is_column_header(row, columns, header_cell_ids):
                continue
            cells = sorted(row.get("cells") or [], key=lambda c: (int(c.get("col") or 0), str(c.get("cell_id") or "")))
            if not cells:
                continue
            if row.get("is_full_width") or len(cells) == 1:
                value = self.cell_display_value(cells[0])
                if not self.cleaned_label(value):
                    continue
                if section_heading is None and self.row_looks_like_section_heading(section_id, value):
                    section_heading = value
                else:
                    data_rows.append({columns[0]: value})
                continue
            data_row: dict[str, str] = {}
            for index, column in enumerate(columns):
                if index < len(cells):
                    data_row[column] = str(cells[index]["cell_id"])
            if data_row:
                data_rows.append(data_row)

        if not data_rows:
            return None
        payload: dict[str, Any] = {
            "section_id": section_id,
            "header_cell_ids": header_cell_ids,
            "columns": columns,
            "rows": data_rows,
        }
        if section_heading:
            payload["section_heading"] = section_heading
        return payload

    def build_table_section_json(self, section_id: str, chunk_names: list[str]) -> dict[str, Any] | None:
        table_ids = self.table_ids_for_chunk_names(chunk_names)
        if not table_ids:
            return None
        rows = self.table_rows_for_ids(table_ids)
        if not rows:
            return None

        max_column_count = max((int(row.get("column_count") or 0) for row in rows), default=0)
        columns, header_cell_ids = self.column_context(section_id, chunk_names, max_column_count)
        if section_id in {"SEC12", "SEC13"}:
            return self.build_workflow_table_section_json(
                section_id,
                rows,
                columns,
                header_cell_ids,
            )
        return self.build_matrix_table_section_json(section_id, rows, columns, header_cell_ids)

    def render_final_document(self, section_payload: dict[str, Any], display_name: str) -> str:
        results = section_payload.get("results") or []
        pieces = [
            '<div class="final-document">',
            '<header class="final-doc-header">',
            f'<p>Final document</p><h1 class="{direction_class(display_name)}" '
            f'dir="{dominant_direction(display_name)}">{escape(repair_text(display_name))}</h1>',
            '</header>',
        ]
        rendered_count = 0
        for result_index, result in enumerate(results):
            if not isinstance(result, dict) or result.get("status") != "resolved":
                continue
            section_id = str(result.get("section_id") or "")
            section_json = result.get("section_json")
            if isinstance(section_json, dict) and section_json.get("status") == "not_found":
                raw_chunk_names = result.get("source_chunk_file_names") or []
                chunk_names = raw_chunk_names if isinstance(raw_chunk_names, list) else [str(raw_chunk_names)]
                fallback = self.build_table_section_json(section_id, chunk_names)
                if fallback is None:
                    continue
                result["section_json"] = fallback
                section_json = fallback
            html = self.render_section_payload(section_id, section_json, result_index)
            if html:
                pieces.append(html)
                rendered_count += 1
        if rendered_count == 0:
            pieces.append('<div class="empty-document">No resolved document sections are ready to display.</div>')
        pieces.append("</div>")
        return "".join(pieces)


def default_steps() -> list[dict[str, Any]]:
    return [
        {"id": "upload", "label": "Upload document", "status": "pending", "detail": "Waiting for a DOCX file"},
        {"id": "prepare", "label": "Read document", "status": "pending", "detail": "Extract text, tables, and embedded files"},
        {"id": "classify", "label": "Find document parts", "status": "pending", "detail": "Group each extracted part"},
        {"id": "review", "label": "Human review", "status": "pending", "detail": "Confirm each extracted part"},
        {"id": "inspect", "label": "Check tables", "status": "pending", "detail": "Resolve table headers and layout"},
        {"id": "extract", "label": "Create structured document", "status": "pending", "detail": "Build the final JSON"},
        {"id": "final", "label": "Display final document", "status": "pending", "detail": "Format the generated JSON"},
    ]


def default_milvus_steps() -> list[dict[str, Any]]:
    return [
        {"id": "select", "label": "Select documents", "status": "pending", "detail": "Choose one or more RAG TXT document sets"},
        {"id": "validate", "label": "Validate chunk sizes", "status": "pending", "detail": "Stop if any TXT would exceed Milvus limits"},
        {"id": "model", "label": "Load models", "status": "pending", "detail": "Load the embedding and reranker models on CUDA"},
        {"id": "collection", "label": "Recreate collection", "status": "pending", "detail": "Drop and rebuild the target Milvus collection"},
        {"id": "ingest", "label": "Embed and insert", "status": "pending", "detail": "Write vectors and metadata to Milvus"},
        {"id": "ready", "label": "Ready to query", "status": "pending", "detail": "Ask a question and inspect the results"},
    ]

@dataclass
class UiJob:
    id: str
    display_name: str
    document_path: Path
    source_docx_path: Path
    config: dict[str, Any]
    status: str = "queued"
    phase: str = "queued"
    phase_label: str = "Queued"
    progress: dict[str, Any] = field(default_factory=lambda: {"current": 0, "total": 1, "percent": 0, "detail": "Queued"})
    steps: list[dict[str, Any]] = field(default_factory=default_steps)
    messages: list[dict[str, str]] = field(default_factory=list)
    artifact_dir: Path | None = None
    classification_output_path: Path | None = None
    inspection_output_path: Path | None = None
    section_json_output_path: Path | None = None
    classification_payload: dict[str, Any] | None = None
    inspection_payload: dict[str, Any] | None = None
    section_json_payload: dict[str, Any] | None = None
    review_items: list[dict[str, Any]] = field(default_factory=list)
    final_html: str | None = None
    error: str | None = None
    version: int = 0
    review_event: threading.Event = field(default_factory=threading.Event)
    review_updates: list[dict[str, Any]] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def touch(self) -> None:
        self.version += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "display_name": self.display_name,
                "status": self.status,
                "phase": self.phase,
                "phase_label": self.phase_label,
                "progress": dict(self.progress),
                "steps": [dict(step) for step in self.steps],
                "messages": list(self.messages[-80:]),
                "review_items": self.review_items if self.status == "awaiting_review" else [],
                "section_options": section_options() if self.status == "awaiting_review" else [],
                "final_ready": self.final_html is not None,
                "error": self.error,
                "version": self.version,
            }

    def set_status(self, status: str, phase: str, label: str, detail: str | None = None) -> None:
        with self.lock:
            self.status = status
            self.phase = phase
            self.phase_label = label
            if detail is not None:
                self.progress["detail"] = detail
            self.touch()

    def set_progress(self, *, current: int, total: int, detail: str) -> None:
        with self.lock:
            safe_total = max(total, 0)
            percent = 0 if safe_total <= 0 else max(0, min(100, round((current / safe_total) * 100)))
            self.progress = {"current": max(0, current), "total": safe_total, "percent": percent, "detail": detail}
            self.touch()

    def update_step(self, step_id: str, status: str, detail: str | None = None) -> None:
        with self.lock:
            for step in self.steps:
                if step["id"] == step_id:
                    step["status"] = status
                    if detail is not None:
                        step["detail"] = detail
                    break
            self.touch()

    def add_message(self, message: str, kind: str = "info") -> None:
        with self.lock:
            self.messages.append({"time": utc_timestamp(), "kind": kind, "message": repair_text(message)})
            self.touch()


@dataclass
class MilvusJob:
    id: str
    collection_name: str
    config: dict[str, Any]
    documents: list[dict[str, Any]]
    status: str = "queued"
    phase: str = "queued"
    phase_label: str = "Queued"
    progress: dict[str, Any] = field(default_factory=lambda: {"current": 0, "total": 1, "percent": 0, "detail": "Queued"})
    steps: list[dict[str, Any]] = field(default_factory=default_milvus_steps)
    messages: list[dict[str, str]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    version: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)

    def touch(self) -> None:
        self.version += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "collection_name": self.collection_name,
                "status": self.status,
                "phase": self.phase,
                "phase_label": self.phase_label,
                "progress": dict(self.progress),
                "steps": [dict(step) for step in self.steps],
                "messages": list(self.messages[-80:]),
                "documents": [
                    {"id": document.get("id"), "name": document.get("name")}
                    for document in self.documents
                ],
                "result": dict(self.result) if isinstance(self.result, dict) else self.result,
                "error": self.error,
                "version": self.version,
            }

    def set_status(self, status: str, phase: str, label: str, detail: str | None = None) -> None:
        with self.lock:
            self.status = status
            self.phase = phase
            self.phase_label = label
            if detail is not None:
                self.progress["detail"] = detail
            self.touch()

    def set_progress(self, *, current: int, total: int, detail: str) -> None:
        with self.lock:
            safe_total = max(total, 0)
            percent = 0 if safe_total <= 0 else max(0, min(100, round((current / safe_total) * 100)))
            self.progress = {"current": max(0, current), "total": safe_total, "percent": percent, "detail": repair_text(detail)}
            self.touch()

    def update_step(self, step_id: str, status: str, detail: str | None = None) -> None:
        with self.lock:
            for step in self.steps:
                if step["id"] == step_id:
                    step["status"] = status
                    if detail is not None:
                        step["detail"] = repair_text(detail)
                    break
            self.touch()

    def add_message(self, message: str, kind: str = "info") -> None:
        with self.lock:
            self.messages.append({"time": utc_timestamp(), "kind": kind, "message": repair_text(message)})
            self.touch()


class PipelineError(RuntimeError):
    pass


def build_targets(document_name: str, artifact_dir: Path) -> list[classification.ChunkTarget]:
    chunk_paths = sorted(
        path for path in artifact_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt" and "_nested_" not in path.name
    )
    if not chunk_paths:
        raise FileNotFoundError(f"No extracted document parts were found in {artifact_dir}")
    return [
        classification.ChunkTarget(
            document_name=document_name,
            txt_file_name=chunk_path.name,
            relative_path=classification.project_relative_path(chunk_path),
            file_path=chunk_path,
            raw_text=chunk_path.read_text(encoding="utf-8"),
        )
        for chunk_path in chunk_paths
    ]


def load_artifact_maps(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    table_map = load_json_file(artifact_dir / chunking.DEFAULT_TABLE_MAP_NAME)
    cell_map = load_json_file(artifact_dir / chunking.DEFAULT_CELL_MAP_NAME)
    asset_path = artifact_dir / chunking.DEFAULT_ASSET_MAP_NAME
    asset_map = load_json_file(asset_path) if asset_path.exists() else {}
    return table_map, cell_map, asset_map


def build_renderer(job: UiJob) -> DocumentRenderer:
    if job.artifact_dir is None:
        raise PipelineError("Document artifacts are not ready yet.")
    table_map, cell_map, asset_map = load_artifact_maps(job.artifact_dir)
    return DocumentRenderer(
        job_id=job.id,
        document_path=job.document_path,
        artifact_dir=job.artifact_dir,
        table_map=table_map,
        cell_map=cell_map,
        asset_map=asset_map,
        inspection_payload=job.inspection_payload,
    )


def apply_hierarchy_shift_to_payload(
    section_json_payload: dict[str, Any],
    *,
    result_index: int,
    entry_index: int,
    direction: str,
) -> tuple[dict[str, Any], bool]:
    results = section_json_payload.get("results")
    if not isinstance(results, list):
        raise PipelineError("Final document results are not available.")
    if result_index < 0 or result_index >= len(results):
        raise PipelineError("The selected section could not be found.")

    target_result = results[result_index]
    if not isinstance(target_result, dict) or target_result.get("status") != "resolved":
        raise PipelineError("The selected section is not editable.")

    section_id = str(target_result.get("section_id") or "")
    section_json = target_result.get("section_json")
    if not isinstance(section_json, dict) or section_json.get("status") == "not_found":
        raise PipelineError("This section does not contain editable hierarchy entries.")

    entries = section_json.get("entries")
    if not isinstance(entries, list):
        raise PipelineError("This section does not contain editable hierarchy entries.")

    updated_entries, changed = shift_hierarchy_entries(section_id, entries, entry_index, direction)
    if not changed:
        return section_json_payload, False

    updated_section_json = dict(section_json)
    updated_section_json["entries"] = updated_entries

    updated_result = dict(target_result)
    updated_result["section_json"] = updated_section_json

    updated_results = list(results)
    updated_results[result_index] = updated_result

    updated_payload = dict(section_json_payload)
    updated_payload["results"] = updated_results
    summary = dict(updated_payload.get("summary") or {})
    summary["manual_hierarchy_edit_timestamp_utc"] = utc_timestamp()
    summary["manual_hierarchy_edit_count"] = int(summary.get("manual_hierarchy_edit_count") or 0) + 1
    updated_payload["summary"] = summary
    return updated_payload, True


def parse_hierarchy_shift_request(payload: dict[str, Any]) -> tuple[int, int, str]:
    try:
        result_index = int(payload.get("result_index"))
        entry_index = int(payload.get("entry_index"))
    except (TypeError, ValueError) as exc:
        raise PipelineError("Hierarchy edit payload is invalid.") from exc
    direction = str(payload.get("direction") or "").strip().lower()
    if direction not in {"left", "right"}:
        raise PipelineError("Hierarchy direction must be left or right.")
    return result_index, entry_index, direction


def persist_job_section_json(job: UiJob, section_json_payload: dict[str, Any]) -> None:
    renderer = build_renderer(job)
    final_html = renderer.render_final_document(section_json_payload, job.display_name)
    if job.section_json_output_path is not None:
        classification.write_json(job.section_json_output_path, section_json_payload)
    with job.lock:
        job.section_json_payload = section_json_payload
        job.final_html = final_html
        job.touch()


def sync_jobs_for_document(document_path: Path, section_json_payload: dict[str, Any]) -> None:
    resolved_document_path = document_path.resolve()
    with _jobs_lock:
        matching_jobs = [job for job in _jobs.values() if job.document_path.resolve() == resolved_document_path]
    for job in matching_jobs:
        persist_job_section_json(job, section_json_payload)


def build_history_renderer(doc_path: Path, doc_id: str) -> DocumentRenderer:
    artifact_dir = history_artifact_dir(doc_path)
    if artifact_dir is None:
        raise FileNotFoundError("Chunk artifacts not found.")
    table_map = load_json_file(artifact_dir / chunking.DEFAULT_TABLE_MAP_NAME)
    cell_map = load_json_file(artifact_dir / chunking.DEFAULT_CELL_MAP_NAME)
    asset_map_path = artifact_dir / chunking.DEFAULT_ASSET_MAP_NAME
    asset_map = load_json_file(asset_map_path) if asset_map_path.exists() else {}

    inspection_payload: dict[str, Any] = {}
    inspection_path = doc_path / DEFAULT_INSPECTION_OUTPUT_NAME
    if inspection_path.exists():
        try:
            inspection_payload = json.loads(inspection_path.read_text(encoding="utf-8"))
        except Exception:
            inspection_payload = {}

    return DocumentRenderer(
        job_id=doc_id,
        document_path=doc_path,
        artifact_dir=artifact_dir,
        table_map=table_map,
        cell_map=cell_map,
        asset_map=asset_map,
        inspection_payload=inspection_payload,
        asset_url_prefix=f"/api/history/{doc_id}/asset",
    )


def render_history_final_html(doc_path: Path, doc_id: str, section_json_payload: dict[str, Any]) -> str:
    renderer = build_history_renderer(doc_path, doc_id)
    return renderer.render_final_document(section_json_payload, doc_path.name)


def export_job_rag_txt(job: UiJob) -> dict[str, Any]:
    if job.section_json_payload is None:
        raise PipelineError("The final document is not ready yet.")
    table_map: dict[str, Any] = {}
    cell_map: dict[str, Any] = {}
    asset_map: dict[str, Any] = {}
    if job.artifact_dir is not None:
        try:
            table_map, cell_map, asset_map = load_artifact_maps(job.artifact_dir)
        except FileNotFoundError:
            # Still export useful RAG files from the structured JSON if the
            # artifact maps are unavailable; unresolved CL/TB/EM tokens are
            # recorded in chunk metadata.
            table_map, cell_map, asset_map = {}, {}, {}
    return rag_txt_export.export_rag_txt_files(
        section_json_payload=job.section_json_payload,
        output_dir=job.document_path / "rag_txt",
        table_map=table_map,
        cell_map=cell_map,
        asset_map=asset_map,
    )


def list_milvus_documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for entry in list_history_documents():
        if not entry.get("has_rag_txt"):
            continue
        doc_id = str(entry["id"])
        doc_path = doc_id_to_path(doc_id)
        if doc_path is None:
            continue
        try:
            manifest = milvus_rag.load_rag_manifest(doc_path)
        except Exception:
            continue
        total_characters = 0
        for chunk in manifest.get("chunks") or []:
            if isinstance(chunk, dict):
                total_characters += int(chunk.get("content_character_count") or 0)
        documents.append({
            "id": doc_id,
            "name": entry["name"],
            "created_at": entry["created_at"],
            "rag_chunk_count": int(manifest.get("chunk_count") or 0),
            "total_characters": total_characters,
            "has_section_json": bool(entry.get("has_section_json")),
        })
    return documents


def resolve_milvus_documents(doc_ids: list[Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_doc_id in doc_ids:
        if not isinstance(raw_doc_id, str):
            continue
        doc_id = raw_doc_id.strip()
        if not doc_id or doc_id in seen:
            continue
        doc_path = doc_id_to_path(doc_id)
        if doc_path is None:
            raise PipelineError("One or more selected documents could not be found.")
        try:
            milvus_rag.load_rag_manifest(doc_path)
        except Exception as exc:
            raise PipelineError(f"{doc_path.name} is missing a valid rag_txt manifest: {repair_text(exc)}") from exc
        selected.append({
            "id": doc_id,
            "name": doc_path.name,
            "path": str(doc_path),
        })
        seen.add(doc_id)
    if not selected:
        raise PipelineError("Select at least one document with rag_txt output.")
    return selected


def milvus_progress(job: MilvusJob, payload: dict[str, Any]) -> None:
    stage = str(payload.get("stage") or "")
    current = int(payload.get("current") or 0)
    total = int(payload.get("total") or 0)
    detail = repair_text(str(payload.get("detail") or "Working"))

    if stage == "validate":
        job.update_step("select", "done", f"{len(job.documents)} documents selected")
        job.update_step("validate", "running", detail)
        job.set_status("running", "validate", "Validating RAG TXT files", detail)
    elif stage == "model":
        job.update_step("validate", "done", "All selected TXT files fit the Milvus limits")
        job.update_step("model", "running", detail)
        job.set_status("running", "model", "Loading embedding model", detail)
    elif stage == "connect":
        job.update_step("model", "done", "Models ready")
        job.update_step("collection", "running", "Connecting to Milvus and recreating the collection")
        job.set_status("running", "collection", "Recreating collection", detail)
    elif stage in {"embed", "insert"}:
        if job.steps[3]["status"] != "done":
            job.update_step("collection", "done", f"Collection {job.collection_name} is ready")
        job.update_step("ingest", "running", detail)
        job.set_status("running", "ingest", "Ingesting chunks", detail)

    job.set_progress(current=current, total=total, detail=detail)


def run_milvus_ingestion(job: MilvusJob) -> None:
    try:
        job.update_step("select", "running", f"{len(job.documents)} documents queued")
        job.set_status("running", "validate", "Preparing ingestion", "Checking the selected RAG TXT files")
        job.set_progress(current=0, total=max(1, len(job.documents)), detail="Checking the selected RAG TXT files")
        result = milvus_rag.ingest_documents(
            collection_name=job.collection_name,
            milvus_uri=str(job.config["milvus_uri"]),
            milvus_token=str(job.config.get("milvus_token") or ""),
            milvus_db_name=str(job.config["milvus_db_name"]),
            documents=job.documents,
            batch_size=int(job.config["batch_size"]),
            progress_callback=lambda payload: milvus_progress(job, payload),
        )
        if result.get("recreated"):
            job.add_message(f"Existing collection {job.collection_name} was dropped and recreated.")
        job.add_message(
            f"Ingested {result.get('chunk_count', 0)} chunks from {result.get('document_count', 0)} documents into {job.collection_name}."
        )
        with job.lock:
            job.result = result
            job.touch()
        job.update_step("model", "done", "Embedding model ready")
        job.update_step("collection", "done", f"Collection {job.collection_name} is ready")
        job.update_step("ingest", "done", f"{result.get('chunk_count', 0)} chunks inserted")
        job.update_step("ready", "done", "Collection is ready for retrieval")
        job.set_progress(current=1, total=1, detail="Collection is ready for retrieval")
        job.set_status("completed", "ready", "Milvus collection ready", "Collection is ready for retrieval")
    except Exception as exc:
        with job.lock:
            job.status = "failed"
            job.phase = "failed"
            job.phase_label = "Stopped"
            job.error = repair_text(str(exc)) or exc.__class__.__name__
            for step in job.steps:
                if step["status"] in {"running", "paused"}:
                    step["status"] = "error"
                    step["detail"] = job.error
            job.touch()
        job.add_message(traceback.format_exc(), kind="error")


def build_review_items(job: UiJob, classification_payload: dict[str, Any]) -> list[dict[str, Any]]:
    renderer = build_renderer(job)
    results = classification_payload.get("results") or []
    items: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        if result.get("skipped"):
            continue
        sections = [str(section_id) for section_id in result.get("predicted_sections") or []]
        item_html = renderer.render_chunk(str(result.get("txt_file_name") or ""), str(result.get("relative_path") or ""))
        items.append({
            "index": index,
            "title": f"Part {index + 1}",
            "sections": sections,
            "section_labels": [section_display(section_id)["label"] for section_id in sections],
            "html": item_html,
            "approved": False,
        })
    return items


def run_classification_step(job: UiJob, llm_logger: LlmLogger | None = None) -> dict[str, Any]:
    if job.artifact_dir is None:
        raise PipelineError("Document artifacts are not ready yet.")
    targets = build_targets(job.display_name, job.artifact_dir)
    job.set_status("running", "classify", "Finding document parts", "Starting document analysis")
    job.update_step("classify", "running", f"0 of {len(targets)} parts complete")
    job.set_progress(current=0, total=len(targets), detail="Starting document analysis")
    client, resolved_base_url = classification.initialize_client(
        base_url=job.config["base_url"], api_key=job.config["api_key"], model=job.config["model"]
    )
    if resolved_base_url.rstrip("/") != job.config["base_url"].rstrip("/"):
        job.add_message(f"Resolved AI server address to {resolved_base_url}")
    system_prompt = classification.load_system_prompt()
    results: list[dict[str, Any]] = []
    invalid_json_retry_count = 0
    document_prediction_context: list[tuple[str, list[str]]] = []
    for index, target in enumerate(targets, start=1):
        job.set_progress(current=index - 1, total=len(targets), detail=f"Reading part {index} of {len(targets)}")
        if not target.raw_text.strip() or target.raw_text.strip() == chunking.EMPTY_TABLE_SENTINEL or not any(c.isalnum() for c in target.raw_text):
            results.append({
                "document_name": target.document_name,
                "txt_file_name": target.txt_file_name,
                "relative_path": target.relative_path,
                "predicted_sections": [],
                "json_retry_count": 0,
                "invalid_attempts": [],
                "preview": "",
                "skipped": "empty",
            })
            job.update_step("classify", "running", f"{index} of {len(targets)} parts complete")
            job.set_progress(current=index, total=len(targets), detail=f"Part {index} skipped (empty)")
            continue

        predicted_sections, json_retry_count, invalid_attempts = classification.request_prediction(
            client=client,
            system_prompt=system_prompt,
            target=target,
            model=job.config["model"],
            max_json_retries=int(job.config["max_classification_json_retries"]),
            previous_predictions=document_prediction_context,
            log_callback=llm_logger.log if llm_logger else None,
        )
        invalid_json_retry_count += json_retry_count
        document_prediction_context.append((target.txt_file_name, predicted_sections))
        results.append({
            "document_name": target.document_name,
            "txt_file_name": target.txt_file_name,
            "relative_path": target.relative_path,
            "predicted_sections": predicted_sections,
            "json_retry_count": json_retry_count,
            "invalid_attempts": invalid_attempts,
            "preview": classification.preview_text(target.raw_text),
        })
        labels = ", ".join(section_display(section_id)["short_label"] for section_id in predicted_sections)
        job.update_step("classify", "running", f"{index} of {len(targets)} parts complete")
        job.set_progress(current=index, total=len(targets), detail=f"Part {index} grouped as {labels}")
    payload = {
        "summary": {
            "run_timestamp_utc": utc_timestamp(),
            "document_name": job.display_name,
            "document_path": classification.project_relative_path(job.document_path),
            "chunk_artifact_dir": classification.project_relative_path(job.artifact_dir),
            "prompt_path": classification.project_relative_path(classification.PROMPT_PATH),
            "model": job.config["model"],
            "resolved_base_url": resolved_base_url,
            "total_chunks": len(results),
            "invalid_json_retry_count": invalid_json_retry_count,
        },
        "results": results,
    }
    output_path = job.document_path / DEFAULT_CLASSIFICATION_OUTPUT_NAME
    classification.write_json(output_path, payload)
    job.classification_output_path = output_path
    job.classification_payload = payload
    job.update_step("classify", "done", f"{len(results)} parts grouped")
    return payload

def apply_review_updates(job: UiJob) -> dict[str, Any]:
    if job.classification_payload is None or job.classification_output_path is None:
        raise PipelineError("Classification results are not ready for review.")
    updates = job.review_updates or []
    results = list(job.classification_payload.get("results") or [])
    reviewable_count = sum(1 for r in results if isinstance(r, dict) and not r.get("skipped"))
    if len(updates) != reviewable_count:
        raise PipelineError("Every extracted part must be reviewed before continuing.")
    modified_count = 0
    updates_by_index = {int(item["index"]): item for item in updates}
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        if result.get("skipped"):
            result["review_action"] = "skipped"
            continue
        update = updates_by_index.get(index)
        if update is None or not update.get("approved"):
            raise PipelineError("Every extracted part must be approved before continuing.")
        selected_sections = [str(section_id) for section_id in update.get("sections") or []]
        if not selected_sections:
            raise PipelineError("Each extracted part needs at least one document area.")
        invalid = [section_id for section_id in selected_sections if section_id not in classification.VALID_SECTION_IDS]
        if invalid:
            raise PipelineError(f"Unknown document area selected: {', '.join(invalid)}")
        selected_sections = list(dict.fromkeys(selected_sections))
        original = list(result.get("predicted_sections") or [])
        if "llm_predicted_sections" not in result:
            result["llm_predicted_sections"] = original
        result["predicted_sections"] = selected_sections
        if selected_sections != original:
            modified_count += 1
            result["review_action"] = "modified"
        else:
            result["review_action"] = "confirmed"
    reviewed = dict(job.classification_payload)
    reviewed["results"] = results
    summary = dict(reviewed.get("summary") or {})
    summary["human_reviewed"] = True
    summary["review_timestamp_utc"] = utc_timestamp()
    summary["review_modifications_count"] = modified_count
    reviewed["summary"] = summary
    classification.write_json(job.classification_output_path, reviewed)
    job.classification_payload = reviewed
    return reviewed


def inspection_progress(job: UiJob, payload: dict[str, Any]) -> None:
    total = int(payload.get("total") or 0)
    current = int(payload.get("current") or 0)
    section_id = str(payload.get("section_id") or "")
    label = section_display(section_id)["short_label"] if section_id else "tables"
    detail = f"Checking {label}" if total else "No table checks needed"
    if total:
        detail = f"{detail} ({current} of {total})"
    job.set_progress(current=current, total=total, detail=detail)
    job.update_step("inspect", "running", detail)


def extraction_progress(job: UiJob, payload: dict[str, Any]) -> None:
    total = int(payload.get("total") or 0)
    current = int(payload.get("current") or 0)
    section_id = str(payload.get("section_id") or "")
    label = section_display(section_id)["short_label"] if section_id else "sections"
    detail = f"Creating {label}" if total else "Preparing final document"
    if total:
        detail = f"{detail} ({current} of {total})"
    job.set_progress(current=current, total=total, detail=detail)
    job.update_step("extract", "running", detail)


def run_pipeline(job: UiJob) -> None:
    llm_logger = LlmLogger(job.document_path / "llm_calls.jsonl")
    try:
        job.set_status("running", "prepare", "Reading document", "Extracting document content")
        job.update_step("upload", "done", "Document received")
        job.update_step("prepare", "running", "Extracting text, tables, colors, and embedded files")
        job.set_progress(current=0, total=1, detail="Extracting text, tables, colors, and embedded files")
        document = chunking.build_document_paths(job.document_path)
        export_result = chunking.export_document(
            document,
            reporter=lambda _message: None,
            unwrap_outer_cell=bool(job.config.get("unwrap_outer_cell")),
        )
        job.artifact_dir = Path(export_result["table_map_path"]).parent.resolve()
        chunk_count = int(export_result.get("chunk_count") or 0)
        job.update_step("prepare", "done", f"{chunk_count} parts extracted")
        job.set_progress(current=1, total=1, detail=f"{chunk_count} parts extracted")

        classification_payload = run_classification_step(job, llm_logger=llm_logger)
        job.review_items = build_review_items(job, classification_payload)
        job.set_status("awaiting_review", "review", "Waiting for human review", "Confirm each extracted part to continue")
        job.update_step("review", "paused", f"0 of {len(job.review_items)} parts approved")
        job.set_progress(current=0, total=len(job.review_items), detail="Waiting for approvals")
        job.review_event.wait()

        if job.review_updates is None:
            raise PipelineError("Review was not submitted.")
        reviewed_payload = apply_review_updates(job)
        approved_count = len(reviewed_payload.get("results") or [])
        job.update_step("review", "done", f"{approved_count} parts approved")
        if job.classification_output_path is None:
            raise PipelineError("Classification output path is missing.")

        job.set_status("running", "inspect", "Checking tables", "Resolving table layouts")
        job.update_step("inspect", "running", "Resolving table layouts")
        job.inspection_output_path = job.document_path / DEFAULT_INSPECTION_OUTPUT_NAME
        inspection_payload = header_inspection.inspect_classified_document(
            document_path=job.document_path,
            classification_output_path=job.classification_output_path,
            output_path=job.inspection_output_path,
            model=job.config["model"],
            base_url=job.config["base_url"],
            api_key=job.config["api_key"],
            max_llm_retries=int(job.config["max_inspection_retries"]),
            quiet=True,
            progress_callback=lambda payload: inspection_progress(job, payload),
            log_callback=llm_logger.log,
        )
        job.inspection_payload = inspection_payload
        inspection_summary = inspection_payload.get("summary") or {}
        job.update_step("inspect", "done", f"{inspection_summary.get('resolved_result_count', 0)} table checks complete")

        job.set_status("running", "extract", "Creating structured document", "Building the final JSON")
        job.update_step("extract", "running", "Building the final JSON")
        job.section_json_output_path = job.document_path / DEFAULT_SECTION_JSON_OUTPUT_NAME
        section_json_payload = section_json_extraction.extract_document_sections(
            document_path=job.document_path,
            classification_output_path=job.classification_output_path,
            column_inspection_output_path=job.inspection_output_path,
            output_path=job.section_json_output_path,
            model=job.config["model"],
            base_url=job.config["base_url"],
            api_key=job.config["api_key"],
            max_llm_retries=int(job.config["max_section_json_retries"]),
            quiet=True,
            progress_callback=lambda payload: extraction_progress(job, payload),
            log_callback=llm_logger.log,
        )
        job.section_json_payload = section_json_payload
        section_summary = section_json_payload.get("summary") or {}
        job.update_step(
            "extract",
            "done" if section_summary.get("failed_count", 0) == 0 else "warning",
            f"{section_summary.get('resolved_count', 0)} sections created; {section_summary.get('failed_count', 0)} need attention",
        )
        job.set_status("running", "final", "Formatting final document", "Preparing display")
        job.update_step("final", "running", "Preparing display")
        renderer = build_renderer(job)
        job.final_html = renderer.render_final_document(section_json_payload, job.display_name)
        if job.section_json_output_path is not None:
            classification.write_json(job.section_json_output_path, section_json_payload)
        job.update_step("final", "done", "Final document is ready")
        job.set_progress(current=1, total=1, detail="Final document is ready")
        job.set_status("completed", "final", "Final document ready", "Final document is ready")
    except Exception as exc:
        with job.lock:
            job.status = "failed"
            job.error = repair_text(str(exc)) or exc.__class__.__name__
            job.phase_label = "Stopped"
            for step in job.steps:
                if step["status"] in {"running", "paused"}:
                    step["status"] = "error"
                    step["detail"] = job.error
            job.touch()
        job.add_message(traceback.format_exc(), kind="error")

@app.get("/")
def index() -> str:
    return render_template(
        "ui_index.html",
        default_model=classification.DEFAULT_MODEL,
        default_base_url=os.environ.get("OPENAI_BASE_URL", classification.DEFAULT_BASE_URL),
        has_api_key=bool(os.environ.get("OPENAI_API_KEY")),
    )


@app.get("/milvus")
def milvus_page() -> str:
    return render_template(
        "milvus.html",
        default_milvus_uri=milvus_rag.DEFAULT_MILVUS_URI,
        default_milvus_db_name=milvus_rag.DEFAULT_MILVUS_DB_NAME,
        default_embedding_model=milvus_rag.EMBEDDING_MODEL_NAME,
        default_reranker_model=milvus_rag.RERANKER_MODEL_NAME,
        default_top_k=milvus_rag.DEFAULT_TOP_K,
        default_top_n=milvus_rag.DEFAULT_TOP_N,
        dependency_error=milvus_rag.dependency_error_message(),
    )


@app.get("/api/department-catalog")
def api_department_catalog() -> Response:
    return jsonify(department_catalog())


@app.post("/api/upload")
def api_upload() -> Response:
    global _active_job_id
    upload = request.files.get("document")
    if upload is None or not upload.filename:
        return jsonify({"error": "Upload a DOCX document first."}), 400
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only DOCX files are supported."}), 400
    departments = parse_requested_departments(request.form.get("departments_json"))
    if not departments:
        return jsonify({"error": "Select at least one department before starting."}), 400
    with _jobs_lock:
        if _active_job_id and _jobs.get(_active_job_id) and _jobs[_active_job_id].status in {"queued", "running", "awaiting_review"}:
            return jsonify({"error": "A document is already being processed. Finish it before uploading another."}), 409
        stem = safe_document_stem(upload.filename)
        document_path = unique_document_dir(stem)
        document_path.mkdir(parents=True, exist_ok=False)
        source_path = document_path / chunking.DEFAULT_SOURCE_DOCX_NAME
        upload.save(source_path)
        save_document_departments(document_path, departments)
        job_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        config = {
            "model": request.form.get("model") or classification.DEFAULT_MODEL,
            "base_url": request.form.get("base_url") or os.environ.get("OPENAI_BASE_URL", classification.DEFAULT_BASE_URL),
            "api_key": request.form.get("api_key") or os.environ.get("OPENAI_API_KEY", ""),
            "max_classification_json_retries": int(request.form.get("max_classification_json_retries") or 0),
            "max_inspection_retries": int(request.form.get("max_inspection_retries") or 6),
            "max_section_json_retries": int(request.form.get("max_section_json_retries") or 6),
            "unwrap_outer_cell": request.form.get("unwrap_outer_cell") == "1",
        }
        job = UiJob(
            id=job_id,
            display_name=Path(upload.filename).stem,
            document_path=document_path.resolve(),
            source_docx_path=source_path.resolve(),
            config=config,
        )
        _jobs[job_id] = job
        _active_job_id = job_id
    thread = threading.Thread(target=run_pipeline, args=(job,), name=f"ui-pipeline-{job_id}", daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.get("/api/state")
def api_state() -> Response:
    with _jobs_lock:
        job = _jobs.get(_active_job_id or "")
    if job is None:
        return jsonify({"status": "idle", "steps": default_steps(), "messages": [], "version": 0})
    return jsonify(job.snapshot())


@app.get("/api/events")
def api_events() -> Response:
    def stream() -> Any:
        last_version: int | None = None
        idle_sent = False
        while True:
            with _jobs_lock:
                job = _jobs.get(_active_job_id or "")
            if job is None:
                if not idle_sent:
                    yield "data: " + json.dumps({"status": "idle", "version": 0}) + "\n\n"
                    idle_sent = True
                time.sleep(0.8)
                continue
            snapshot = job.snapshot()
            if snapshot["version"] != last_version:
                yield "data: " + json.dumps(snapshot, ensure_ascii=False) + "\n\n"
                last_version = snapshot["version"]
            time.sleep(0.5)
    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/review")
def api_review() -> Response:
    with _jobs_lock:
        job = _jobs.get(_active_job_id or "")
    if job is None or job.status != "awaiting_review":
        return jsonify({"error": "No document is waiting for review."}), 409
    payload = request.get_json(silent=True) or {}
    items = payload.get("items")
    if not isinstance(items, list):
        return jsonify({"error": "Review payload is invalid."}), 400
    with job.lock:
        expected = len(job.review_items)
    if len(items) != expected:
        return jsonify({"error": "Every extracted part must be submitted."}), 400
    try:
        normalized: list[dict[str, Any]] = []
        for item in items:
            index = int(item.get("index"))
            sections = [str(section_id) for section_id in item.get("sections") or []]
            if not item.get("approved"):
                return jsonify({"error": "Every extracted part must be approved."}), 400
            if not sections:
                return jsonify({"error": "Each extracted part needs at least one document area."}), 400
            invalid = [section_id for section_id in sections if section_id not in classification.VALID_SECTION_IDS]
            if invalid:
                return jsonify({"error": "One or more selected areas are invalid."}), 400
            normalized.append({"index": index, "sections": sections, "approved": True})
    except (TypeError, ValueError):
        return jsonify({"error": "Review payload is invalid."}), 400
    with job.lock:
        job.review_updates = normalized
        job.status = "running"
        job.phase = "inspect"
        job.phase_label = "Continuing"
        job.review_items = []
        job.touch()
    job.review_event.set()
    return jsonify({"ok": True})


@app.get("/api/final")
def api_final() -> Response:
    with _jobs_lock:
        job = _jobs.get(_active_job_id or "")
    if job is None or job.final_html is None:
        return jsonify({"error": "The final document is not ready yet."}), 404
    return jsonify({"html": job.final_html})


@app.post("/api/final/hierarchy")
def api_final_hierarchy() -> Response:
    with _jobs_lock:
        job = _jobs.get(_active_job_id or "")
    if job is None or job.section_json_payload is None:
        return jsonify({"error": "The final document is not ready yet."}), 404

    request_payload = request.get_json(silent=True) or {}
    try:
        result_index, entry_index, direction = parse_hierarchy_shift_request(request_payload)
        section_json_payload, changed = apply_hierarchy_shift_to_payload(
            job.section_json_payload,
            result_index=result_index,
            entry_index=entry_index,
            direction=direction,
        )
        if changed:
            persist_job_section_json(job, section_json_payload)
    except PipelineError as exc:
        return jsonify({"error": repair_text(exc)}), 400
    except (IndexError, ValueError) as exc:
        return jsonify({"error": repair_text(exc)}), 400

    return jsonify({"ok": True, "changed": changed, "html": job.final_html or ""})


@app.get("/api/download")
def api_download() -> Response:
    with _jobs_lock:
        job = _jobs.get(_active_job_id or "")
    if job is None or job.section_json_payload is None:
        return jsonify({"error": "The final document is not ready yet."}), 404
    raw_name = re.sub(r'\s+', ' ', job.display_name).strip() or "document"
    filename = f"{raw_name}.json"
    # ASCII-safe fallback for clients that don't support RFC 5987
    ascii_filename = re.sub(r'[^\x20-\x7e]', '_', filename)
    # RFC 5987 percent-encoded UTF-8 filename for modern clients
    encoded_filename = _url_quote(filename, safe=" -._~()'!*")
    content_disposition = (
        f'attachment; filename="{ascii_filename}"; '
        f"filename*=UTF-8''{encoded_filename}"
    )
    data = json.dumps(job.section_json_payload, ensure_ascii=False, indent=2)
    return Response(
        data,
        mimetype="application/json",
        headers={"Content-Disposition": content_disposition},
    )


@app.get("/api/download-rag-txt")
def api_download_rag_txt() -> Response:
    with _jobs_lock:
        job = _jobs.get(_active_job_id or "")
    if job is None or job.section_json_payload is None:
        return jsonify({"error": "The final document is not ready yet."}), 404
    try:
        manifest = export_job_rag_txt(job)
    except FileNotFoundError as exc:
        return jsonify({"error": f"TXT export artifacts are missing: {repair_text(exc)}"}), 404
    if int(manifest.get("chunk_count") or 0) == 0:
        return jsonify({"error": "No resolved sections were available for TXT export."}), 404

    output_dir = Path(str(manifest["output_dir"]))
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for chunk in manifest.get("chunks") or []:
            file_path = Path(str(chunk.get("file_path") or ""))
            if file_path.exists() and file_path.is_file():
                archive.write(file_path, arcname=file_path.name)
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            archive.write(manifest_path, arcname=manifest_path.name)
    zip_buffer.seek(0)

    raw_name = re.sub(r'\s+', ' ', job.display_name).strip() or "document"
    filename = f"{raw_name}_rag_txt.zip"
    ascii_filename = re.sub(r'[^\x20-\x7e]', '_', filename)
    encoded_filename = _url_quote(filename, safe=" -._~()'!*")
    content_disposition = (
        f'attachment; filename="{ascii_filename}"; '
        f"filename*=UTF-8''{encoded_filename}"
    )
    return Response(
        zip_buffer.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": content_disposition},
    )


@app.get("/api/assets/<job_id>/<asset_id>")
def api_asset(job_id: str, asset_id: str) -> Response:
    job = _jobs.get(job_id)
    if job is None or job.artifact_dir is None:
        return jsonify({"error": "Asset not found."}), 404
    asset_map_path = job.artifact_dir / chunking.DEFAULT_ASSET_MAP_NAME
    if not asset_map_path.exists():
        return jsonify({"error": "Asset not found."}), 404
    asset_map = load_json_file(asset_map_path)
    asset = asset_map.get(asset_id)
    if not isinstance(asset, dict):
        return jsonify({"error": "Asset not found."}), 404
    try:
        return send_asset_response(job.artifact_dir, asset, asset_id)
    except ValueError:
        return jsonify({"error": "Asset path is invalid."}), 400
    except FileNotFoundError:
        return jsonify({"error": "Asset not found."}), 404


@app.post("/api/reset")
def api_reset() -> Response:
    global _active_job_id
    with _jobs_lock:
        active = _jobs.get(_active_job_id or "")
        if active and active.status in {"queued", "running", "awaiting_review"}:
            return jsonify({"error": "Cannot reset while a document is being processed."}), 409
        _active_job_id = None
    return jsonify({"ok": True})


@app.get("/api/milvus/documents")
def api_milvus_documents() -> Response:
    return jsonify(list_milvus_documents())


@app.get("/api/milvus/state")
def api_milvus_state() -> Response:
    with _milvus_jobs_lock:
        job = _milvus_jobs.get(_active_milvus_job_id or "")
    if job is None:
        return jsonify({"status": "idle", "steps": default_milvus_steps(), "messages": [], "version": 0})
    return jsonify(job.snapshot())


@app.get("/api/milvus/events")
def api_milvus_events() -> Response:
    def stream() -> Any:
        last_version: int | None = None
        idle_sent = False
        while True:
            with _milvus_jobs_lock:
                job = _milvus_jobs.get(_active_milvus_job_id or "")
            if job is None:
                if not idle_sent:
                    yield "data: " + json.dumps({"status": "idle", "version": 0}) + "\n\n"
                    idle_sent = True
                time.sleep(0.8)
                continue
            snapshot = job.snapshot()
            if snapshot["version"] != last_version:
                yield "data: " + json.dumps(snapshot, ensure_ascii=False) + "\n\n"
                last_version = snapshot["version"]
            time.sleep(0.5)

    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/milvus/reset")
def api_milvus_reset() -> Response:
    global _active_milvus_job_id
    with _milvus_jobs_lock:
        active = _milvus_jobs.get(_active_milvus_job_id or "")
        if active and active.status in {"queued", "running"}:
            return jsonify({"error": "Cannot reset while an ingestion job is running."}), 409
        _active_milvus_job_id = None
    return jsonify({"ok": True})


@app.post("/api/milvus/ingest")
def api_milvus_ingest() -> Response:
    global _active_milvus_job_id
    body = request.get_json(force=True, silent=True) or {}
    try:
        collection_name = milvus_rag.validate_collection_name(str(body.get("collection_name") or ""))
        documents = resolve_milvus_documents(body.get("doc_ids") if isinstance(body.get("doc_ids"), list) else [])
        config = {
            "milvus_uri": milvus_rag.normalize_milvus_uri(str(body.get("milvus_uri") or milvus_rag.DEFAULT_MILVUS_URI)),
            "milvus_token": str(body.get("milvus_token") or ""),
            "milvus_db_name": str(body.get("milvus_db_name") or milvus_rag.DEFAULT_MILVUS_DB_NAME),
            "batch_size": max(1, int(body.get("batch_size") or milvus_rag.DEFAULT_BATCH_SIZE)),
        }
    except PipelineError as exc:
        return jsonify({"error": repair_text(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": repair_text(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": repair_text(exc)}), 400

    with _milvus_jobs_lock:
        if _active_milvus_job_id:
            active = _milvus_jobs.get(_active_milvus_job_id)
            if active and active.status in {"queued", "running"}:
                return jsonify({"error": "A Milvus ingestion job is already running."}), 409
        job_id = datetime.now().strftime("milvus-%Y%m%d%H%M%S%f")
        job = MilvusJob(
            id=job_id,
            collection_name=collection_name,
            config=config,
            documents=documents,
        )
        _milvus_jobs[job_id] = job
        _active_milvus_job_id = job_id

    thread = threading.Thread(target=run_milvus_ingestion, args=(job,), name=f"milvus-ingest-{job_id}", daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


_METADATA_END_MARKER = "--- END RAG METADATA ---"
_ORDER_PREFIX_RE = re.compile(r"^(\d{4})")


def _parse_rag_metadata_key(text: str) -> tuple[str, str, str]:
    """Return (document_line, section_line, hierarchy_line) from a RAG txt body."""
    doc = section = hierarchy = ""
    in_meta = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "--- RAG METADATA ---":
            in_meta = True
            continue
        if stripped == "--- END RAG METADATA ---":
            break
        if in_meta:
            if stripped.startswith("Document:"):
                doc = stripped[len("Document:"):].strip()
            elif stripped.startswith("Section:"):
                section = stripped[len("Section:"):].strip()
            elif stripped.startswith("Hierarchy:"):
                hierarchy = stripped[len("Hierarchy:"):].strip()
    return doc, section, hierarchy


def stitch_reranked_documents(reranked_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group reranked hits by identical RAG metadata within the same document.

    For each unique (document_id, Document, Section, Hierarchy) group found in
    the reranker results, all txt files in that document's rag_txt/ directory
    that carry the exact same metadata are collected, sorted by their order
    prefix, and stitched into one entry with a single metadata header.
    Output is sorted by best reranker score descending.
    """
    if not reranked_documents:
        return []

    # Step 1 — parse each reranked hit's metadata and group by it
    # key = (document_id, doc_line, section_line, hierarchy_line)
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for doc in reranked_documents:
        document_id = str(doc.get("document_id") or "")
        meta = _parse_rag_metadata_key(str(doc.get("text") or ""))
        key = (document_id, *meta)
        score = float(doc.get("reranker_score") or float("-inf"))

        fname = str(doc.get("file_name") or "")
        if key not in groups:
            groups[key] = {
                "document_id": document_id,
                "document_name": doc.get("document_name") or "",
                "section_id": doc.get("section_id") or "",
                "chunk_type": doc.get("chunk_type") or "",
                "hierarchy_path": doc.get("hierarchy_path") or "",
                "best_reranker_score": doc.get("reranker_score"),
                "best_milvus_score": doc.get("milvus_score"),
                "_sort_score": score,
                "retrieved_file_names": [fname] if fname else [],
            }
        else:
            if score > groups[key]["_sort_score"]:
                groups[key]["best_reranker_score"] = doc.get("reranker_score")
                groups[key]["_sort_score"] = score
            ms = doc.get("milvus_score")
            if ms is not None and (
                groups[key]["best_milvus_score"] is None
                or ms > groups[key]["best_milvus_score"]
            ):
                groups[key]["best_milvus_score"] = ms
            if fname and fname not in groups[key]["retrieved_file_names"]:
                groups[key]["retrieved_file_names"].append(fname)

    # Step 2 — for each unique document_id scan its rag_txt/ dir once and
    # build a metadata-key → [(order, path, raw_text)] index
    doc_meta_index: dict[str, dict[tuple[str, str, str], list[tuple[str, Path, str]]]] = {}

    for document_id in {key[0] for key in groups}:
        doc_path = doc_id_to_path(document_id)
        if doc_path is None:
            continue
        rag_dir = doc_path / "rag_txt"
        if not rag_dir.is_dir():
            continue

        meta_files: dict[tuple[str, str, str], list[tuple[str, Path, str]]] = {}
        for f in rag_dir.iterdir():
            if f.suffix.lower() != ".txt":
                continue
            try:
                raw = f.read_text(encoding="utf-8")
            except Exception:
                continue
            fkey = _parse_rag_metadata_key(raw)
            m = _ORDER_PREFIX_RE.match(f.stem)
            order = m.group(1) if m else f.stem
            meta_files.setdefault(fkey, []).append((order, f, raw))

        for fkey in meta_files:
            meta_files[fkey].sort(key=lambda x: x[0])

        doc_meta_index[document_id] = meta_files

    # Step 3 — stitch each group
    stitched: list[dict[str, Any]] = []

    for (document_id, doc_line, section_line, hierarchy_line), group in groups.items():
        fkey = (doc_line, section_line, hierarchy_line)
        file_list = doc_meta_index.get(document_id, {}).get(fkey, [])
        if not file_list:
            continue

        text_parts: list[str] = []
        for idx, (_, f, raw) in enumerate(file_list):
            raw = raw.strip()
            if idx == 0:
                text_parts.append(raw)
            else:
                pos = raw.find(_METADATA_END_MARKER)
                content = raw[pos + len(_METADATA_END_MARKER):].lstrip("\n") if pos != -1 else raw
                if content:
                    text_parts.append(content)

        stitched.append({
            "document_id": document_id,
            "document_name": group["document_name"],
            "section_id": group["section_id"],
            "chunk_type": group["chunk_type"],
            "hierarchy_path": group["hierarchy_path"],
            "retrieved_file_names": group["retrieved_file_names"],
            "part_count": len(file_list),
            "text": "\n\n".join(text_parts),
            "reranker_score": group["best_reranker_score"],
            "milvus_score": group["best_milvus_score"],
        })

    stitched.sort(key=lambda x: float(x.get("reranker_score") or float("-inf")), reverse=True)
    return stitched


_HEADING_WS_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"</?[^>]+>")
_WORKFLOW_HEADER_TYPES = frozenset({"group_header", "subgroup_header", "sub_subgroup_header", "subsubgroup_header"})
_WORKFLOW_HEADER_LEVEL: dict[str, int] = {
    "group_header": 1, "subgroup_header": 2, "sub_subgroup_header": 3, "subsubgroup_header": 3,
}
_HIERARCHY_SHIFT_BTN_RE = re.compile(r"<button\b[^>]*\bdata-hierarchy-shift\b[^>]*>.*?</button>", re.DOTALL | re.IGNORECASE)


def _normalize_heading(value: Any) -> str:
    """Strip HTML tags, color tokens and invisible chars, then collapse whitespace.
    Mirrors what plain_label() does in rag_txt_export so headings from section_json
    can be compared with the plain-text versions stored in hierarchy_path.
    """
    text = str(value or "")
    text = COLOR_TOKEN_RE.sub("", text)       # [#RRGGBB] color markers
    text = FORMATTING_TAG_RE.sub("", text)    # <strong>, <em>, <HL:…> etc.
    text = _HTML_TAG_RE.sub("", text)          # remaining HTML tags
    text = INVISIBLE_CHARS_RE.sub("", text)
    return _HEADING_WS_RE.sub(" ", text).strip()


def _header_semantic_level(entry: dict[str, Any]) -> int | None:
    """Return the semantic nesting level of a header entry, or None if not a header."""
    etype = str(entry.get("type") or "")
    if etype in _WORKFLOW_HEADER_TYPES:
        return _WORKFLOW_HEADER_LEVEL.get(etype, 1)
    if etype == "subsection":
        try:
            return max(1, int(entry.get("semantic_level") or entry.get("level") or 1))
        except (TypeError, ValueError):
            return 1
    return None


def _filter_section_entries_for_hierarchy(
    entries: list[dict[str, Any]],
    target_header_titles: list[str],
) -> list[dict[str, Any]]:
    """Return only the entries (plus their required ancestor headers) that sit
    under the given header path.

    target_header_titles is the ordered list of normalized header headings
    derived from the hierarchy_path after stripping the section name prefix.
    An empty list means the RAG chunk is at the top level of the section
    (not inside any group/subgroup header), so only top-level entries are
    returned — entries nested under a group header are excluded.

    Two structural patterns are handled:
    • Workflow sections (SEC12/13): group_header/subgroup_header are pure
      structural containers; matching content lives in subsequent 'step' siblings.
    • Hierarchy sections (SEC11/18): 'subsection' entries are both the header
      AND the content container (their value/values fields hold the content).
      When the subsection heading matches the final target level, the entry
      itself is included directly so the renderer shows its content.
    """

    normalized_target = [_normalize_heading(h) for h in target_header_titles]
    result: list[dict[str, Any]] = []
    # Stack: (semantic_level, normalized_title, entry, already_added_to_result)
    header_stack: list[tuple[int, str, dict[str, Any], bool]] = []

    def _flush_ancestors() -> None:
        nonlocal header_stack
        flushed: list[tuple[int, str, dict[str, Any], bool]] = []
        for l, t, e, added in header_stack:
            if not added:
                result.append(e)
            flushed.append((l, t, e, True))
        header_stack = flushed

    for entry in entries:
        etype = str(entry.get("type") or "")
        sem_level = _header_semantic_level(entry)

        if sem_level is not None:
            # Pop headers at the same or deeper semantic level
            header_stack = [(l, t, e, a) for l, t, e, a in header_stack if l < sem_level]
            title = _normalize_heading(entry.get("heading") or "")
            current_path = [t for _, t, _, _ in header_stack] + [title]

            if etype == "subsection" and current_path == normalized_target:
                # SEC11/18 style: the subsection IS the content container.
                # Flush ancestors and include this entry directly — the renderer
                # will render its value/values fields as the content body.
                _flush_ancestors()
                result.append(entry)
                header_stack.append((sem_level, title, entry, True))
            else:
                # Pure structural header (group_header etc.) — just track it.
                header_stack.append((sem_level, title, entry, False))
            continue

        # Leaf entry (step / content / anything else)
        current_titles = [t for _, t, _, _ in header_stack]
        if current_titles != normalized_target:
            continue

        _flush_ancestors()
        result.append(entry)

    return result


def enrich_stitched_with_html(stitched_documents: list[dict[str, Any]]) -> None:
    """Add rendered_html to each stitched document in-place.

    For each unique document_id, loads section_json_output.json once and
    builds the DocumentRenderer (same path as the Final Output / history page).

    For sections that have hierarchical entries (SEC12, SEC13, SEC11, SEC18),
    the entries are filtered to only those that match the stitched group's
    hierarchy_path — so only the relevant procedure/instruction steps are shown,
    together with their ancestor group/subgroup headers for context.

    Hierarchy-shift editing buttons are stripped from the output because the
    stitched view is read-only.
    """
    unique_doc_ids = {doc["document_id"] for doc in stitched_documents}

    for document_id in unique_doc_ids:
        doc_path = doc_id_to_path(document_id)
        if doc_path is None:
            continue

        section_json_path = doc_path / DEFAULT_SECTION_JSON_OUTPUT_NAME
        if not section_json_path.exists():
            continue

        try:
            section_json_payload = json.loads(section_json_path.read_text(encoding="utf-8"))
            renderer = build_history_renderer(doc_path, document_id)
        except Exception:
            continue

        # Build section_id → (section_json_dict, result_index) — first resolved match wins
        section_lookup: dict[str, tuple[Any, int]] = {}
        for result_index, result in enumerate(section_json_payload.get("results") or []):
            if not isinstance(result, dict) or result.get("status") != "resolved":
                continue
            sid = str(result.get("section_id") or "")
            if sid and sid not in section_lookup:
                sj = result.get("section_json")
                if isinstance(sj, dict) and sj.get("status") != "not_found":
                    section_lookup[sid] = (sj, result_index)

        for doc in stitched_documents:
            if doc["document_id"] != document_id:
                continue
            section_id = doc.get("section_id") or ""
            lookup_entry = section_lookup.get(section_id)
            if lookup_entry is None:
                doc["rendered_html"] = None
                continue

            section_json_dict, result_index = lookup_entry

            # --- Derive the target header titles from hierarchy_path ---
            section_json_for_render = section_json_dict
            if isinstance(section_json_dict, dict) and isinstance(section_json_dict.get("entries"), list):
                hierarchy_path = doc.get("hierarchy_path") or ""
                section_heading = _normalize_heading(section_json_dict.get("section_heading") or "")
                path_parts = [p.strip() for p in hierarchy_path.split(" > ")]
                # Strip the leading section-name segment so we have only header titles
                if path_parts and _normalize_heading(path_parts[0]) == section_heading:
                    target_headers = path_parts[1:]
                else:
                    target_headers = path_parts[1:] if len(path_parts) > 1 else []

                filtered_entries = _filter_section_entries_for_hierarchy(
                    section_json_dict["entries"], target_headers
                )
                if not filtered_entries:
                    doc["rendered_html"] = None
                    continue
                section_json_for_render = {**section_json_dict, "entries": filtered_entries}

            try:
                html = renderer.render_section_payload(section_id, section_json_for_render, result_index)
                # Strip hierarchy-shift buttons — this is a read-only view
                html = _HIERARCHY_SHIFT_BTN_RE.sub("", html)
                doc["rendered_html"] = html
            except Exception:
                doc["rendered_html"] = None


@app.post("/api/milvus/query")
def api_milvus_query() -> Response:
    body = request.get_json(force=True, silent=True) or {}
    try:
        raw_doc_ids = body.get("doc_ids") if isinstance(body.get("doc_ids"), list) else []
        selected_documents = resolve_milvus_documents(raw_doc_ids) if raw_doc_ids else []
        payload = milvus_rag.query_collection(
            question=str(body.get("question") or ""),
            collection_name=str(body.get("collection_name") or ""),
            milvus_uri=str(body.get("milvus_uri") or milvus_rag.DEFAULT_MILVUS_URI),
            milvus_token=str(body.get("milvus_token") or ""),
            milvus_db_name=str(body.get("milvus_db_name") or milvus_rag.DEFAULT_MILVUS_DB_NAME),
            top_k=int(body.get("top_k") or milvus_rag.DEFAULT_TOP_K),
            top_n=int(body.get("top_n") or milvus_rag.DEFAULT_TOP_N),
            document_ids=[str(document["id"]) for document in selected_documents],
        )
    except ValueError as exc:
        return jsonify({"error": repair_text(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": repair_text(exc)}), 400
    except Exception as exc:
        return jsonify({"error": repair_text(exc)}), 500
    payload["stitched_documents"] = stitch_reranked_documents(payload.get("reranked_documents") or [])
    enrich_stitched_with_html(payload["stitched_documents"])
    return jsonify(payload)


# ── History routes ────────────────────────────────────────────────────────────

@app.get("/history")
def history_page() -> str:
    return render_template("history.html")


@app.get("/api/history")
def api_history_list() -> Response:
    return jsonify(list_history_documents())


@app.delete("/api/history/<doc_id>")
def api_history_delete(doc_id: str) -> Response:
    global _active_job_id
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404

    documents_root = DOCUMENTS_ROOT.resolve()
    resolved_doc_path = doc_path.resolve()
    try:
        resolved_doc_path.relative_to(documents_root)
    except ValueError:
        return jsonify({"error": "Document path is invalid."}), 400
    if resolved_doc_path == documents_root:
        return jsonify({"error": "Document path is invalid."}), 400

    with _jobs_lock:
        for job in _jobs.values():
            if job.document_path.resolve() == resolved_doc_path and job.status in {"queued", "running", "awaiting_review"}:
                return jsonify({"error": "Cannot delete a document while it is being processed."}), 409

    try:
        shutil.rmtree(resolved_doc_path)
    except FileNotFoundError:
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": f"Failed to delete document: {exc}"}), 500

    with _jobs_lock:
        deleted_job_ids = [
            job_id for job_id, job in _jobs.items()
            if job.document_path.resolve() == resolved_doc_path
        ]
        for job_id in deleted_job_ids:
            _jobs.pop(job_id, None)
        if _active_job_id in deleted_job_ids:
            _active_job_id = None

    return jsonify({"ok": True})


@app.get("/api/history/<doc_id>")
def api_history_detail(doc_id: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404

    artifact_dir = history_artifact_dir(doc_path)
    source_docx = history_source_docx(doc_path)
    departments = get_document_departments(doc_path)
    chunk_count = 0
    if artifact_dir is not None:
        chunk_count = sum(
            1 for p in artifact_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt" and "_nested_" not in p.name
        )
    info: dict[str, Any] = {
        "id": doc_id,
        "name": doc_path.name,
        "created_at": datetime.fromtimestamp(history_entry_mtime(doc_path), tz=timezone.utc).isoformat(),
        "departments": departments,
        "has_docx": source_docx is not None,
        "chunk_count": chunk_count,
        "assets": history_asset_list(artifact_dir),
    }
    for filename, key in [
        (DEFAULT_CLASSIFICATION_OUTPUT_NAME, "classification"),
        (DEFAULT_INSPECTION_OUTPUT_NAME, "inspection"),
        (DEFAULT_SECTION_JSON_OUTPUT_NAME, "section_json"),
    ]:
        p = doc_path / filename
        if p.exists():
            try:
                info[key] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                info[key] = None
        else:
            info[key] = None
    return jsonify(info)


@app.post("/api/history/<doc_id>/departments")
def api_history_departments(doc_id: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404

    body = request.get_json(silent=True) or {}
    departments = parse_requested_departments(body.get("departments"))
    if not departments:
        return jsonify({"error": "Select at least one department."}), 400

    try:
        departments = save_document_departments(doc_path, departments)
    except ValueError as exc:
        return jsonify({"error": repair_text(exc)}), 400

    return jsonify({
        "ok": True,
        "departments": departments,
        "options": department_catalog().get("options") or [],
    })


@app.get("/api/history/<doc_id>/chunks")
def api_history_chunks(doc_id: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404

    artifact_dir = history_artifact_dir(doc_path)
    if artifact_dir is None:
        return jsonify([])

    try:
        table_map = load_json_file(artifact_dir / chunking.DEFAULT_TABLE_MAP_NAME)
        cell_map = load_json_file(artifact_dir / chunking.DEFAULT_CELL_MAP_NAME)
        asset_map_path = artifact_dir / chunking.DEFAULT_ASSET_MAP_NAME
        asset_map = load_json_file(asset_map_path) if asset_map_path.exists() else {}
    except FileNotFoundError:
        return jsonify([])

    renderer = DocumentRenderer(
        job_id=doc_id,
        document_path=doc_path,
        artifact_dir=artifact_dir,
        table_map=table_map,
        cell_map=cell_map,
        asset_map=asset_map,
        asset_url_prefix=f"/api/history/{doc_id}/asset",
    )

    classification_payload: dict[str, Any] = {}
    classification_path = doc_path / DEFAULT_CLASSIFICATION_OUTPUT_NAME
    if classification_path.exists():
        try:
            classification_payload = json.loads(classification_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    results_by_name: dict[str, dict[str, Any]] = {}
    for result in classification_payload.get("results") or []:
        if isinstance(result, dict):
            results_by_name[str(result.get("txt_file_name") or "")] = result

    chunk_paths = sorted(
        p for p in artifact_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt" and "_nested_" not in p.name
    )
    chunks = []
    for chunk_path in chunk_paths:
        result = results_by_name.get(chunk_path.name, {})
        if result.get("skipped"):
            continue
        sections = [str(s) for s in result.get("predicted_sections") or []]
        html = renderer.render_chunk(
            chunk_path.name,
            classification.project_relative_path(chunk_path),
        )
        chunks.append({
            "name": chunk_path.name,
            "sections": sections,
            "section_labels": [section_display(s)["label"] for s in sections],
            "review_action": result.get("review_action"),
            "llm_predicted_sections": result.get("llm_predicted_sections"),
            "html": html,
        })
    return jsonify(chunks)


@app.get("/api/history/<doc_id>/llm-logs")
def api_history_llm_logs(doc_id: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404
    log_path = doc_path / "llm_calls.jsonl"
    if not log_path.exists():
        return jsonify([])
    entries: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return jsonify(entries)


@app.get("/api/history/<doc_id>/final")
def api_history_final(doc_id: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404

    section_json_path = doc_path / DEFAULT_SECTION_JSON_OUTPUT_NAME
    if not section_json_path.exists():
        return jsonify({"error": "Final document JSON not found."}), 404

    try:
        section_json_payload = json.loads(section_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return jsonify({"error": f"Failed to load artifacts: {exc}"}), 500

    try:
        html = render_history_final_html(doc_path, doc_id, section_json_payload)
    except FileNotFoundError:
        return jsonify({"error": "Chunk artifacts not found."}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to load artifacts: {exc}"}), 500
    return jsonify({"html": html})


@app.post("/api/history/<doc_id>/final/hierarchy")
def api_history_final_hierarchy(doc_id: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404

    section_json_path = doc_path / DEFAULT_SECTION_JSON_OUTPUT_NAME
    if not section_json_path.exists():
        return jsonify({"error": "Final document JSON not found."}), 404

    request_payload = request.get_json(silent=True) or {}
    try:
        section_json_payload = json.loads(section_json_path.read_text(encoding="utf-8"))
        result_index, entry_index, direction = parse_hierarchy_shift_request(request_payload)
        section_json_payload, changed = apply_hierarchy_shift_to_payload(
            section_json_payload,
            result_index=result_index,
            entry_index=entry_index,
            direction=direction,
        )
        if changed:
            classification.write_json(section_json_path, section_json_payload)
            sync_jobs_for_document(doc_path, section_json_payload)
        html = render_history_final_html(doc_path, doc_id, section_json_payload)
    except PipelineError as exc:
        return jsonify({"error": repair_text(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Chunk artifacts not found."}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to update hierarchy: {repair_text(exc)}"}), 500

    return jsonify({"ok": True, "changed": changed, "html": html})


@app.get("/api/history/<doc_id>/asset/<asset_id>")
def api_history_asset(doc_id: str, asset_id: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Asset not found."}), 404
    artifact_dir = history_artifact_dir(doc_path)
    if artifact_dir is None:
        return jsonify({"error": "Asset not found."}), 404
    asset_map_path = artifact_dir / chunking.DEFAULT_ASSET_MAP_NAME
    if not asset_map_path.exists():
        return jsonify({"error": "Asset not found."}), 404
    asset_map = load_json_file(asset_map_path)
    asset = asset_map.get(asset_id)
    if not isinstance(asset, dict):
        return jsonify({"error": "Asset not found."}), 404
    try:
        return send_asset_response(artifact_dir, asset, asset_id)
    except ValueError:
        return jsonify({"error": "Asset path is invalid."}), 400
    except FileNotFoundError:
        return jsonify({"error": "Asset not found."}), 404


@app.get("/api/history/<doc_id>/download/<which>")
def api_history_download(doc_id: str, which: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404

    raw_name = re.sub(r"\s+", " ", doc_path.name).strip() or "document"

    if which == "docx":
        docx_path = history_source_docx(doc_path)
        if docx_path is None or not docx_path.exists():
            return jsonify({"error": "Source DOCX not found."}), 404
        filename = f"{raw_name}.docx"
        ascii_filename = re.sub(r"[^\x20-\x7e]", "_", filename)
        encoded_filename = _url_quote(filename, safe=" -._~()'!*")
        return send_file(
            docx_path,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True,
            download_name=ascii_filename,
        )

    json_files = {
        "classification": (DEFAULT_CLASSIFICATION_OUTPUT_NAME, "classification"),
        "inspection": (DEFAULT_INSPECTION_OUTPUT_NAME, "inspection"),
        "section_json": (DEFAULT_SECTION_JSON_OUTPUT_NAME, "section_json"),
    }
    if which in json_files:
        file_name, suffix = json_files[which]
        json_path = doc_path / file_name
        if not json_path.exists():
            return jsonify({"error": "File not found."}), 404
        filename = f"{raw_name}_{suffix}.json"
        ascii_filename = re.sub(r"[^\x20-\x7e]", "_", filename)
        encoded_filename = _url_quote(filename, safe=" -._~()'!*")
        content_disposition = (
            f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{encoded_filename}'
        )
        return Response(
            json_path.read_bytes(),
            mimetype="application/json",
            headers={"Content-Disposition": content_disposition},
        )

    if which == "rag-txt":
        section_json_path = doc_path / DEFAULT_SECTION_JSON_OUTPUT_NAME
        if not section_json_path.exists():
            return jsonify({"error": "Final document JSON not found."}), 404
        artifact_dir = history_artifact_dir(doc_path)
        try:
            section_json_payload = json.loads(section_json_path.read_text(encoding="utf-8"))
            table_map, cell_map, asset_map = load_artifact_maps(artifact_dir) if artifact_dir is not None else ({}, {}, {})
        except Exception:
            table_map, cell_map, asset_map = {}, {}, {}
        manifest = rag_txt_export.export_rag_txt_files(
            section_json_payload=section_json_payload,
            output_dir=doc_path / "rag_txt",
            table_map=table_map,
            cell_map=cell_map,
            asset_map=asset_map,
        )
        if int(manifest.get("chunk_count") or 0) == 0:
            return jsonify({"error": "No resolved sections available for export."}), 404
        output_dir = Path(str(manifest["output_dir"]))
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for chunk in manifest.get("chunks") or []:
                file_path = Path(str(chunk.get("file_path") or ""))
                if file_path.exists() and file_path.is_file():
                    archive.write(file_path, arcname=file_path.name)
            manifest_path = output_dir / "manifest.json"
            if manifest_path.exists():
                archive.write(manifest_path, arcname=manifest_path.name)
        zip_buffer.seek(0)
        filename = f"{raw_name}_rag_txt.zip"
        ascii_filename = re.sub(r"[^\x20-\x7e]", "_", filename)
        encoded_filename = _url_quote(filename, safe=" -._~()'!*")
        content_disposition = (
            f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{encoded_filename}'
        )
        return Response(
            zip_buffer.getvalue(),
            mimetype="application/zip",
            headers={"Content-Disposition": content_disposition},
        )

    return jsonify({"error": "Unknown download type."}), 400


@app.post("/api/history/download-rag-txts")
def api_history_download_rag_txts() -> Response:
    body = request.get_json(force=True, silent=True) or {}
    doc_ids = body.get("doc_ids")
    if not isinstance(doc_ids, list) or not doc_ids:
        return jsonify({"error": "doc_ids must be a non-empty list."}), 400

    zip_buffer = io.BytesIO()
    included = 0
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for doc_id in doc_ids:
            if not isinstance(doc_id, str):
                continue
            doc_path = doc_id_to_path(doc_id)
            if doc_path is None:
                continue
            section_json_path = doc_path / DEFAULT_SECTION_JSON_OUTPUT_NAME
            if not section_json_path.exists():
                continue
            artifact_dir = history_artifact_dir(doc_path)
            try:
                section_json_payload = json.loads(section_json_path.read_text(encoding="utf-8"))
                table_map, cell_map, asset_map = load_artifact_maps(artifact_dir) if artifact_dir is not None else ({}, {}, {})
            except Exception:
                table_map, cell_map, asset_map = {}, {}, {}
            manifest = rag_txt_export.export_rag_txt_files(
                section_json_payload=section_json_payload,
                output_dir=doc_path / "rag_txt",
                table_map=table_map,
                cell_map=cell_map,
                asset_map=asset_map,
            )
            folder = re.sub(r'[\\/:*?"<>|]+', "_", doc_path.name).strip() or f"doc_{doc_id[:8]}"
            for chunk in manifest.get("chunks") or []:
                file_path = Path(str(chunk.get("file_path") or ""))
                if file_path.exists() and file_path.is_file():
                    archive.write(file_path, arcname=f"{folder}/{file_path.name}")
            manifest_path = doc_path / "rag_txt" / "manifest.json"
            if manifest_path.exists():
                archive.write(manifest_path, arcname=f"{folder}/manifest.json")
            included += 1

    if included == 0:
        return jsonify({"error": "None of the selected documents have a final output available."}), 404

    zip_buffer.seek(0)
    return Response(
        zip_buffer.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": 'attachment; filename="rag_txts.zip"'},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8012, debug=False, threaded=True)
