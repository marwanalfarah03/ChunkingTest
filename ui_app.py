from __future__ import annotations

import base64
import json
import io
import os
import re
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

from flask import Flask, Response, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chunking
import classification
import header_inspection
import rag_txt_export
import section_json_extraction

DOCUMENTS_ROOT = PROJECT_ROOT / "documents"
ALLOWED_EXTENSIONS = {".docx"}
DEFAULT_CLASSIFICATION_OUTPUT_NAME = "classification_output.json"
DEFAULT_INSPECTION_OUTPUT_NAME = "column_header_inspection.json"
DEFAULT_SECTION_JSON_OUTPUT_NAME = "section_json_output.json"

CL_TOKEN_RE = re.compile(r"CL\d{6}")
TB_TOKEN_RE = re.compile(r"<TB\d{6}>")
EM_TOKEN_RE = re.compile(r"<EM\d{6}>")
REFERENCE_TOKEN_RE = re.compile(r"(CL\d{6}|<TB\d{6}>|<EM\d{6}>)")
COLOR_TOKEN_RE = re.compile(r"\s*\[#([0-9A-Fa-f]{6})\]\s*")
RTL_CHAR_RE = re.compile(r"[\u0590-\u08ff\ufb1d-\ufdfd\ufe70-\ufefc]")
LTR_CHAR_RE = re.compile(r"[A-Za-z]")
FORMATTING_TAG_RE = re.compile(r"</?(?:strong|em|u)>", re.IGNORECASE)
INVISIBLE_CHARS_RE = re.compile(r"[​‌‍‎‏‪-‮⁠﻿]")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024

_jobs: dict[str, "UiJob"] = {}
_active_job_id: str | None = None
_jobs_lock = threading.RLock()


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


def list_history_documents() -> list[dict[str, Any]]:
    """Return metadata for every document directory under DOCUMENTS_ROOT."""
    if not DOCUMENTS_ROOT.exists():
        return []
    docs: list[dict[str, Any]] = []
    for entry in sorted(DOCUMENTS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not entry.is_dir():
            continue
        source_docx = entry / chunking.DEFAULT_SOURCE_DOCX_NAME
        if not source_docx.exists():
            continue
        artifact_dir = entry / chunking.DEFAULT_OUTPUT_DIRNAME
        chunk_count = 0
        if artifact_dir.is_dir():
            chunk_count = sum(
                1 for p in artifact_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".txt" and "_nested_" not in p.name
            )
        has_llm_logs = (entry / "llm_calls.jsonl").exists()
        docs.append({
            "id": doc_dir_to_id(entry.name),
            "name": entry.name,
            "created_at": datetime.fromtimestamp(source_docx.stat().st_mtime, tz=timezone.utc).isoformat(),
            "has_classification": (entry / DEFAULT_CLASSIFICATION_OUTPUT_NAME).exists(),
            "has_inspection": (entry / DEFAULT_INSPECTION_OUTPUT_NAME).exists(),
            "has_section_json": (entry / DEFAULT_SECTION_JSON_OUTPUT_NAME).exists(),
            "has_llm_logs": has_llm_logs,
            "has_rag_txt": (entry / "rag_txt").is_dir(),
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
    return re.sub(r"\s+", " ", label).strip(), color


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
                    parts.append(repair_text(asset.get("original_name") or asset.get("stored_name") or part))
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
        label = repair_text(asset.get("original_name") or asset.get("stored_name") or asset_id)
        content_type = str(asset.get("content_type") or "")
        url = f"{self.asset_url_prefix}/{escape(asset_id)}"
        if content_type.startswith("image/"):
            return '<figure class="embedded-asset">' f'<img src="{url}" alt="{escape(label)}">' f'<figcaption>{escape(label)}</figcaption></figure>'
        return f'<a class="embedded-file" href="{url}" target="_blank" rel="noopener"><span>Embedded file</span><strong>{escape(label)}</strong></a>'

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
            path = PROJECT_ROOT / relative_path
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

    def render_hierarchy_entries(self, entries: list[Any]) -> str:
        html: list[str] = ['<div class="hierarchy-list hierarchy-collapsible">']
        subsection_open = False

        def close_subsection() -> None:
            nonlocal subsection_open
            if subsection_open:
                html.append("</div></details>")
                subsection_open = False

        for entry in entries:
            if not isinstance(entry, dict):
                html.append(self.render_content_card(entry, nested=subsection_open))
                continue
            entry_type = entry.get("type")
            if entry_type == "subsection":
                close_subsection()
                heading = entry.get("heading") or "Subsection"
                heading_label, heading_color = extract_display_color(heading)
                style = (
                    f' style="--section-color: {heading_color}; --section-text: {color_text(heading_color)}"'
                    if heading_color
                    else ""
                )
                heading_direction = dominant_direction(heading_label)
                html.append(
                    f'<details class="collapsible-node hierarchy-node hierarchy-subsection-node" open{style}>'
                    f'<summary class="collapsible-summary hierarchy-subsection-summary text-{heading_direction}" '
                    f'dir="{heading_direction}"><span class="collapse-marker" aria-hidden="true"></span>'
                    f'<span class="summary-label">{self.render_inline_text(heading_label)}</span></summary>'
                    '<div class="collapsible-body hierarchy-node-body">'
                )
                if "value" in entry:
                    html.append(self.render_content_card(entry.get("value"), nested=True))
                for value in entry.get("values") or []:
                    html.append(self.render_content_card(value, nested=True))
                subsection_open = True
            elif entry_type == "content":
                html.append(self.render_content_card(entry.get("value"), nested=subsection_open))
            else:
                label = repair_text(entry_type or "Item")
                html.append(self.render_content_card(entry.get("value") or entry, label=label, nested=subsection_open))
        close_subsection()
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

    def render_workflow_entries(self, data: dict[str, Any]) -> str:
        entries = data.get("entries") or []
        columns = self.workflow_columns(data, entries)
        html: list[str] = ['<div class="workflow-view workflow-collapsible">']
        table_open = False
        group_open = False
        subgroup_open = False
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

        def flush_steps() -> None:
            if not pending_steps:
                return
            grid = compute_rowspans(pending_steps)
            for row_idx, step in enumerate(pending_steps):
                fallback_values = [v for k, v in step.items() if k != "type"]
                html.append("<tr>")
                for col_idx, col in enumerate(columns):
                    rs = grid[row_idx][col_idx]
                    if rs == 0:
                        continue
                    value = step.get(col)
                    if value is None and col_idx < len(fallback_values):
                        value = fallback_values[col_idx]
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

        def open_node(kind: str, heading: Any) -> None:
            heading_label, heading_color = extract_display_color(heading or "")
            style = (
                f' style="--group-color: {heading_color}; --group-text: {color_text(heading_color)}"'
                if heading_color
                else ""
            )
            heading_direction = dominant_direction(heading_label)
            node_class = "workflow-group-node" if kind == "group" else "workflow-subgroup-node"
            summary_class = "workflow-group-summary" if kind == "group" else "workflow-subgroup-summary"
            html.append(
                f'<details class="collapsible-node workflow-node {node_class}" open{style}>'
                f'<summary class="collapsible-summary {summary_class} text-{heading_direction}" '
                f'dir="{heading_direction}"><span class="collapse-marker" aria-hidden="true"></span>'
                f'<span class="summary-label">{self.render_inline_text(heading_label)}</span></summary>'
                '<div class="collapsible-body workflow-node-body">'
            )

        def close_subgroup() -> None:
            nonlocal subgroup_open
            close_table()
            if subgroup_open:
                html.append("</div></details>")
                subgroup_open = False

        def close_group() -> None:
            nonlocal group_open
            close_subgroup()
            if group_open:
                html.append("</div></details>")
                group_open = False

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

        for entry in entries:
            if not isinstance(entry, dict):
                close_table()
                html.append(self.render_content_card(entry))
                continue
            entry_type = entry.get("type")
            if entry_type == "group_header":
                close_group()
                open_node("group", entry.get("heading"))
                group_open = True
            elif entry_type == "subgroup_header":
                close_subgroup()
                open_node("subgroup", entry.get("heading"))
                subgroup_open = True
            elif entry_type == "content":
                close_table()
                html.append(self.render_content_card(entry.get("value")))
            elif entry_type == "step":
                open_table()
                pending_steps.append(entry)
            else:
                close_table()
                html.append(self.render_content_card(entry))
        if group_open:
            close_group()
        else:
            close_subgroup()
        close_table()
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

    def render_section_payload(self, section_id: str, payload: Any) -> str:
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
        has_hierarchy_entries = isinstance(entries, list) and section_id in {"SEC11", "SEC12", "SEC13"}
        if isinstance(entries, list) and any(isinstance(item, dict) and item.get("type") in {"group_header", "subgroup_header", "step"} for item in entries):
            body = self.render_workflow_entries(payload)
        elif isinstance(entries, list):
            body = self.render_hierarchy_entries(entries)
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
        for result in results:
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
            html = self.render_section_payload(section_id, section_json)
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


def build_review_items(job: UiJob, classification_payload: dict[str, Any]) -> list[dict[str, Any]]:
    renderer = build_renderer(job)
    results = classification_payload.get("results") or []
    items: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
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
    if len(updates) != len(results):
        raise PipelineError("Every extracted part must be reviewed before continuing.")
    modified_count = 0
    updates_by_index = {int(item["index"]): item for item in updates}
    for index, result in enumerate(results):
        update = updates_by_index.get(index)
        if update is None or not update.get("approved"):
            raise PipelineError("Every extracted part must be approved before continuing.")
        selected_sections = [str(section_id) for section_id in update.get("sections") or []]
        if not selected_sections:
            raise PipelineError("Each extracted part needs at least one document area.")
        invalid = [section_id for section_id in selected_sections if section_id not in classification.VALID_SECTION_IDS]
        if invalid:
            raise PipelineError(f"Unknown document area selected: {', '.join(invalid)}")
        selected_sections = sorted(dict.fromkeys(selected_sections), key=classification.section_sort_key)
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
        export_result = chunking.export_document(document, reporter=lambda _message: None)
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


@app.post("/api/upload")
def api_upload() -> Response:
    global _active_job_id
    upload = request.files.get("document")
    if upload is None or not upload.filename:
        return jsonify({"error": "Upload a DOCX document first."}), 400
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only DOCX files are supported."}), 400
    with _jobs_lock:
        if _active_job_id and _jobs.get(_active_job_id) and _jobs[_active_job_id].status in {"queued", "running", "awaiting_review"}:
            return jsonify({"error": "A document is already being processed. Finish it before uploading another."}), 409
        stem = safe_document_stem(upload.filename)
        document_path = unique_document_dir(stem)
        document_path.mkdir(parents=True, exist_ok=False)
        source_path = document_path / chunking.DEFAULT_SOURCE_DOCX_NAME
        upload.save(source_path)
        job_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        config = {
            "model": request.form.get("model") or classification.DEFAULT_MODEL,
            "base_url": request.form.get("base_url") or os.environ.get("OPENAI_BASE_URL", classification.DEFAULT_BASE_URL),
            "api_key": request.form.get("api_key") or os.environ.get("OPENAI_API_KEY", ""),
            "max_classification_json_retries": int(request.form.get("max_classification_json_retries") or 0),
            "max_inspection_retries": int(request.form.get("max_inspection_retries") or 6),
            "max_section_json_retries": int(request.form.get("max_section_json_retries") or 6),
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
    relative_path = Path(str(asset.get("relative_path") or ""))
    asset_path = (job.artifact_dir / relative_path).resolve()
    try:
        asset_path.relative_to(job.artifact_dir.resolve())
    except ValueError:
        return jsonify({"error": "Asset path is invalid."}), 400
    if not asset_path.exists():
        return jsonify({"error": "Asset not found."}), 404
    return send_file(asset_path)


@app.post("/api/reset")
def api_reset() -> Response:
    global _active_job_id
    with _jobs_lock:
        active = _jobs.get(_active_job_id or "")
        if active and active.status in {"queued", "running", "awaiting_review"}:
            return jsonify({"error": "Cannot reset while a document is being processed."}), 409
        _active_job_id = None
    return jsonify({"ok": True})


# ── History routes ────────────────────────────────────────────────────────────

@app.get("/history")
def history_page() -> str:
    return render_template("history.html")


@app.get("/api/history")
def api_history_list() -> Response:
    return jsonify(list_history_documents())


@app.get("/api/history/<doc_id>")
def api_history_detail(doc_id: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404

    info: dict[str, Any] = {"id": doc_id, "name": doc_path.name}
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


@app.get("/api/history/<doc_id>/chunks")
def api_history_chunks(doc_id: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404

    artifact_dir = doc_path / chunking.DEFAULT_OUTPUT_DIRNAME
    if not artifact_dir.is_dir():
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

    artifact_dir = doc_path / chunking.DEFAULT_OUTPUT_DIRNAME
    if not artifact_dir.is_dir():
        return jsonify({"error": "Chunk artifacts not found."}), 404

    try:
        section_json_payload = json.loads(section_json_path.read_text(encoding="utf-8"))
        table_map = load_json_file(artifact_dir / chunking.DEFAULT_TABLE_MAP_NAME)
        cell_map = load_json_file(artifact_dir / chunking.DEFAULT_CELL_MAP_NAME)
        asset_map_path = artifact_dir / chunking.DEFAULT_ASSET_MAP_NAME
        asset_map = load_json_file(asset_map_path) if asset_map_path.exists() else {}
    except Exception as exc:
        return jsonify({"error": f"Failed to load artifacts: {exc}"}), 500

    inspection_payload: dict[str, Any] = {}
    inspection_path = doc_path / DEFAULT_INSPECTION_OUTPUT_NAME
    if inspection_path.exists():
        try:
            inspection_payload = json.loads(inspection_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    renderer = DocumentRenderer(
        job_id=doc_id,
        document_path=doc_path,
        artifact_dir=artifact_dir,
        table_map=table_map,
        cell_map=cell_map,
        asset_map=asset_map,
        inspection_payload=inspection_payload,
        asset_url_prefix=f"/api/history/{doc_id}/asset",
    )
    html = renderer.render_final_document(section_json_payload, doc_path.name)
    return jsonify({"html": html})


@app.get("/api/history/<doc_id>/asset/<asset_id>")
def api_history_asset(doc_id: str, asset_id: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Asset not found."}), 404
    artifact_dir = doc_path / chunking.DEFAULT_OUTPUT_DIRNAME
    asset_map_path = artifact_dir / chunking.DEFAULT_ASSET_MAP_NAME
    if not asset_map_path.exists():
        return jsonify({"error": "Asset not found."}), 404
    asset_map = load_json_file(asset_map_path)
    asset = asset_map.get(asset_id)
    if not isinstance(asset, dict):
        return jsonify({"error": "Asset not found."}), 404
    relative_path = Path(str(asset.get("relative_path") or ""))
    asset_path = (artifact_dir / relative_path).resolve()
    try:
        asset_path.relative_to(artifact_dir.resolve())
    except ValueError:
        return jsonify({"error": "Asset path is invalid."}), 400
    if not asset_path.exists():
        return jsonify({"error": "Asset not found."}), 404
    return send_file(asset_path)


@app.get("/api/history/<doc_id>/download/<which>")
def api_history_download(doc_id: str, which: str) -> Response:
    doc_path = doc_id_to_path(doc_id)
    if doc_path is None:
        return jsonify({"error": "Document not found."}), 404

    raw_name = re.sub(r"\s+", " ", doc_path.name).strip() or "document"

    if which == "docx":
        docx_path = doc_path / chunking.DEFAULT_SOURCE_DOCX_NAME
        if not docx_path.exists():
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
        artifact_dir = doc_path / chunking.DEFAULT_OUTPUT_DIRNAME
        try:
            section_json_payload = json.loads(section_json_path.read_text(encoding="utf-8"))
            table_map, cell_map, asset_map = load_artifact_maps(artifact_dir) if artifact_dir.is_dir() else ({}, {}, {})
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8012, debug=False, threaded=True)
