from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chunking
import classification
import header_inspection
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
    ) -> None:
        self.job_id = job_id
        self.document_path = document_path
        self.artifact_dir = artifact_dir
        self.table_map = table_map
        self.cell_map = cell_map
        self.asset_map = asset_map

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
        url = f"/api/assets/{escape(self.job_id)}/{escape(asset_id)}"
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
        html: list[str] = ['<div class="hierarchy-list">']
        inside_subsection = False
        for entry in entries:
            if not isinstance(entry, dict):
                html.append(self.render_content_card(entry))
                continue
            entry_type = entry.get("type")
            if entry_type == "subsection":
                inside_subsection = True
                heading = entry.get("heading") or "Subsection"
                heading_label, heading_color = extract_display_color(heading)
                style = f' style="--section-color: {heading_color}; --section-text: {color_text(heading_color)}"' if heading_color else ""
                html.append(f'<section class="subsection-block"{style}>')
                heading_direction = dominant_direction(heading_label)
                html.append(
                    f'<h3 class="text-{heading_direction}" dir="{heading_direction}">'
                    f'{self.render_inline_text(heading_label)}</h3>'
                )
                if "value" in entry:
                    html.append(self.render_content_card(entry.get("value"), nested=True))
                for value in entry.get("values") or []:
                    html.append(self.render_content_card(value, nested=True))
                html.append("</section>")
            elif entry_type == "content":
                html.append(self.render_content_card(entry.get("value"), nested=inside_subsection))
            else:
                label = repair_text(entry_type or "Item")
                html.append(self.render_content_card(entry.get("value") or entry, label=label))
        html.append("</div>")
        return "".join(html)

    def workflow_columns(self, data: dict[str, Any], entries: list[Any]) -> list[str]:
        columns = data.get("columns") or []
        if len(columns) >= 2:
            return list(columns[:2])
        for entry in entries:
            if isinstance(entry, dict) and entry.get("type") == "step":
                keys = [key for key in entry.keys() if key != "type"]
                if len(keys) >= 2:
                    return keys[:2]
        return ["Step", "Owner"]

    def render_workflow_entries(self, data: dict[str, Any]) -> str:
        entries = data.get("entries") or []
        columns = self.workflow_columns(data, entries)
        html: list[str] = ['<div class="workflow-view">']
        table_open = False

        def close_table() -> None:
            nonlocal table_open
            if table_open:
                html.append("</tbody></table></div>")
                table_open = False

        def open_table() -> None:
            nonlocal table_open
            if table_open:
                return
            table_direction = dominant_direction(" ".join(repair_text(column) for column in columns))
            html.append(
                f'<div class="table-scroll"><table class="workflow-table table-{table_direction}" '
                f'dir="ltr" data-content-dir="{table_direction}"><thead><tr>'
            )
            first_direction = dominant_direction(columns[0])
            second_direction = dominant_direction(columns[1])
            html.append(f'<th class="text-{first_direction}" dir="{first_direction}">{self.render_inline_text(columns[0])}</th>')
            html.append(f'<th class="text-{second_direction}" dir="{second_direction}">{self.render_inline_text(columns[1])}</th>')
            html.append("</tr></thead><tbody>")
            table_open = True

        for entry in entries:
            if not isinstance(entry, dict):
                close_table()
                html.append(self.render_content_card(entry))
                continue
            entry_type = entry.get("type")
            if entry_type == "group_header":
                close_table()
                heading, color = extract_display_color(entry.get("heading") or "")
                style = f' style="--group-color: {color}; --group-text: {color_text(color)}"' if color else ""
                heading_direction = dominant_direction(heading)
                html.append(
                    f'<div class="workflow-group text-{heading_direction}" dir="{heading_direction}"{style}>'
                    f'{self.render_inline_text(heading)}</div>'
                )
            elif entry_type == "subgroup_header":
                close_table()
                heading, color = extract_display_color(entry.get("heading") or "")
                style = f' style="--group-color: {color}; --group-text: {color_text(color)}"' if color else ""
                heading_direction = dominant_direction(heading)
                html.append(
                    f'<div class="workflow-subgroup text-{heading_direction}" dir="{heading_direction}"{style}>'
                    f'{self.render_inline_text(heading)}</div>'
                )
            elif entry_type == "content":
                close_table()
                html.append(self.render_content_card(entry.get("value")))
            elif entry_type == "step":
                open_table()
                first = entry.get(columns[0])
                second = entry.get(columns[1])
                if first is None or second is None:
                    values = [value for key, value in entry.items() if key != "type"]
                    first = first if first is not None else (values[0] if values else None)
                    second = second if second is not None else (values[1] if len(values) > 1 else None)
                html.append("<tr>")
                first_direction = dominant_direction(self.direction_source(first))
                second_direction = dominant_direction(self.direction_source(second))
                html.append(
                    f'<td class="workflow-step text-{first_direction}" dir="{first_direction}">'
                    f'{self.render_rich_text(first)}</td>'
                )
                html.append(
                    f'<td class="workflow-owner text-{second_direction}" dir="{second_direction}">'
                    f'{self.render_rich_text(second)}</td>'
                )
                html.append("</tr>")
            else:
                close_table()
                html.append(self.render_content_card(entry))
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
            return '<article class="final-section">' f'{self.render_heading(fallback_heading, fallback_heading)}' f'{body}</article>'
        if not isinstance(payload, dict):
            return '<article class="final-section">' f'{self.render_heading(fallback_heading, fallback_heading)}' f'{self.render_content_card(payload)}</article>'
        if payload.get("status") == "not_found":
            return ""
        heading = payload.get("section_heading") or fallback_heading
        entries = payload.get("entries")
        rows = payload.get("rows")
        columns = payload.get("columns")
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
        return '<article class="final-section">' f'{self.render_heading(heading, fallback_heading)}' f'{body}</article>'

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
            html = self.render_section_payload(str(result.get("section_id") or ""), result.get("section_json"))
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


def run_classification_step(job: UiJob) -> dict[str, Any]:
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

        classification_payload = run_classification_step(job)
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8012, debug=False, threaded=True)
