from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from openai import NotFoundError

from classification import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SECTION_INDEX,
    initialize_client,
    project_relative_path,
    resolve_document_path,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIRNAME = "chunks"
DEFAULT_TABLE_MAP_NAME = "schema_table_map.json"
DEFAULT_CELL_MAP_NAME = "schema_cell_map.json"
DEFAULT_CLASSIFICATION_OUTPUT_NAME = "classification_output.json"
DEFAULT_INSPECTION_OUTPUT_NAME = "column_header_inspection.json"
DEFAULT_INSPECTION_INPUT_DIRNAME = "column_header_inspection_inputs"
ALLOWED_HEADER_STATES = {
    "visible_valid",
    "hidden_valid",
    "visible_invalid",
    "hidden_invalid",
    "missing",
}
CELL_ID_PATTERN = re.compile(r"^CL\d+$")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_WORD_PATTERN = re.compile(r"[^\w\s]+", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
PROMPT_ROW_LIMIT = 8
INPUT_EXCERPT_LINE_LIMIT = 28

INSPECTION_SYSTEM_PROMPT = """
You resolve column-header order for extracted SOP tables.

Rules:
- A full-width section title row is not a column-header row.
- Hidden header cells can exist even when the rendered chunk only shows CL ids.
- If the actual header row is present but corrupt, duplicated, blank, or partially missing, still identify that row's cell ids and resolve the final valid canonical column order.
- If this extracted chunk has no actual header row, return no header cell ids and use the canonical section order.
- valid_column_order must exactly match one of the allowed canonical orders.
- Return JSON only.
""".strip()


@dataclass(frozen=True)
class SectionHeaderSchema:
    section_id: str
    canonical_orders: tuple[tuple[str, ...], ...]
    notes: str

    def orders_for_column_count(self, column_count: int) -> list[tuple[str, ...]]:
        return [order for order in self.canonical_orders if len(order) == column_count]


HEADER_SECTION_SCHEMAS: dict[str, SectionHeaderSchema] = {
    "SEC03": SectionHeaderSchema(
        section_id="SEC03",
        canonical_orders=(("اسم الوثيقة", "نوعها", "الدائرة / الوحدة"),),
        notes="Document control metadata table: document name, type, and owning department or unit.",
    ),
    "SEC04": SectionHeaderSchema(
        section_id="SEC04",
        canonical_orders=(("الاعداد", "المراجعة"), ("الاعداد", "المراجعة", "الموافقة")),
        notes="Document approval table. Most documents have preparation and review columns; some variants add approval.",
    ),
    "SEC05": SectionHeaderSchema(
        section_id="SEC05",
        canonical_orders=(("رقم الاصدار", "تاريخه", "تم الاعداد / التعديل من قبل", "اسباب التعديل"),),
        notes="Version-control history table with edition number, date, preparer or modifier, and reason for change.",
    ),
    "SEC12": SectionHeaderSchema(
        section_id="SEC12",
        canonical_orders=(("الخطوات", "المكلف بالتنفيذ"),),
        notes="Main procedure workflow table: procedural steps on the left and responsible party on the right.",
    ),
    "SEC13": SectionHeaderSchema(
        section_id="SEC13",
        canonical_orders=(("الخطوات", "المكلف بالتنفيذ"),),
        notes="Control-procedure table: control steps or checkpoints on the left and responsible party on the right.",
    ),
    "SEC15": SectionHeaderSchema(
        section_id="SEC15",
        canonical_orders=(("الفئة", "اسم الملف"),),
        notes="Files and retention table. The stable visible headers in the corpus are category and file name.",
    ),
}


def has_chunk_artifacts(directory: Path) -> bool:
    return directory.is_dir() and (directory / DEFAULT_TABLE_MAP_NAME).exists() and (directory / DEFAULT_CELL_MAP_NAME).exists()


def find_chunk_artifact_dir(document_path: Path) -> Path | None:
    candidates = [document_path, document_path / DEFAULT_OUTPUT_DIRNAME]
    for candidate in candidates:
        if has_chunk_artifacts(candidate):
            return candidate.resolve()
    return None


def normalize_whitespace(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def strip_markup(value: str) -> str:
    return normalize_whitespace(HTML_TAG_PATTERN.sub(" ", value or ""))


def normalize_label(value: str) -> str:
    plain_text = strip_markup(value).lower().replace("/", " ")
    plain_text = NON_WORD_PATTERN.sub(" ", plain_text)
    return WHITESPACE_PATTERN.sub("", plain_text)


def label_similarity(left: str, right: str) -> float:
    normalized_left = normalize_label(left)
    normalized_right = normalize_label(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return 0.9
    return SequenceMatcher(a=normalized_left, b=normalized_right).ratio()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def cell_is_visible(cell: dict[str, Any]) -> bool:
    display_text = str(cell.get("display_text") or "").strip()
    return bool(display_text) and display_text != str(cell.get("cell_id"))


def build_table_rows(
    table_id: str,
    table_entry: dict[str, Any],
    cell_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped_rows: dict[int, list[dict[str, Any]]] = {}
    cell_lookup: dict[str, dict[str, Any]] = {}

    for cell_id, payload in cell_map.items():
        if payload.get("table_id") != table_id:
            continue

        cell = {
            "cell_id": cell_id,
            "row": int(payload["row"]),
            "col": int(payload["col"]),
            "rowspan": int(payload["rowspan"]),
            "colspan": int(payload["colspan"]),
            "text": str(payload.get("text") or ""),
            "plain_text": strip_markup(str(payload.get("text") or "")),
            "color": payload.get("color"),
            "display_text": str(payload.get("display_text") or ""),
            "displayed_in_chunk": cell_is_visible({"cell_id": cell_id, "display_text": payload.get("display_text")}),
            "nested_table_ids": list(payload.get("nested_table_ids") or []),
        }
        cell_lookup[cell_id] = cell
        grouped_rows.setdefault(cell["row"], []).append(cell)

    column_count = int(table_entry["column_count"])
    ordered_rows: list[dict[str, Any]] = []
    for row_index in sorted(grouped_rows):
        cells = sorted(grouped_rows[row_index], key=lambda value: (value["col"], value["cell_id"]))
        ordered_rows.append(
            {
                "row_index": row_index,
                "cells": cells,
                "is_full_width_title": len(cells) == 1 and int(cells[0]["colspan"]) >= column_count,
            }
        )

    return ordered_rows, cell_lookup


def logical_column_count(rows: list[dict[str, Any]]) -> int:
    non_title_counts = [len(row["cells"]) for row in rows if not row["is_full_width_title"]]
    if non_title_counts:
        return max(non_title_counts)
    return 1 if rows else 0


def resolve_chunk_path(relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    candidate = PROJECT_ROOT / relative_path
    if candidate.exists():
        return candidate.resolve()
    return None


def load_chunk_content(relative_path: str | None, fallback: str = "") -> str:
    chunk_path = resolve_chunk_path(relative_path)
    if chunk_path is None:
        return fallback
    return chunk_path.read_text(encoding="utf-8")


def build_chunk_records(
    classification_results: list[dict[str, Any]],
    chunk_table_lookup: dict[str, tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, classification_result in enumerate(classification_results):
        if not isinstance(classification_result, dict):
            continue

        txt_file_name = str(classification_result.get("txt_file_name") or "")
        relative_path = str(classification_result.get("relative_path") or "")
        predicted_sections = classification_result.get("predicted_sections", [])
        if not isinstance(predicted_sections, list):
            predicted_sections = []

        table_lookup_entry = chunk_table_lookup.get(txt_file_name)
        table_id = None
        table_entry = None
        chunk_type = "unknown"
        if table_lookup_entry is not None:
            table_id, table_entry = table_lookup_entry
            chunk_type = "table"
        elif txt_file_name.endswith("_text.txt"):
            chunk_type = "text"

        fallback_content = ""
        if table_entry is not None:
            fallback_content = str(table_entry.get("rendered_table") or "")

        records.append(
            {
                "index": index,
                "classification_result": classification_result,
                "txt_file_name": txt_file_name,
                "relative_path": relative_path,
                "predicted_sections": list(predicted_sections),
                "chunk_type": chunk_type,
                "table_id": table_id,
                "table_entry": table_entry,
                "content": load_chunk_content(relative_path, fallback=fallback_content),
            }
        )
    return records


def build_section_spans(chunk_records: list[dict[str, Any]], section_id: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []

    if section_id == "SEC99":
        span_index = 0
        for record in chunk_records:
            if section_id not in record["predicted_sections"]:
                continue
            span_index += 1
            spans.append({"section_id": section_id, "span_index": span_index, "records": (record,)})
        return spans

    active_records: list[dict[str, Any]] = []
    span_index = 0

    for record in chunk_records:
        if section_id in record["predicted_sections"]:
            active_records.append(record)
            continue

        if not active_records:
            continue

        span_index += 1
        spans.append({"section_id": section_id, "span_index": span_index, "records": tuple(active_records)})
        active_records = []

    if active_records:
        span_index += 1
        spans.append({"section_id": section_id, "span_index": span_index, "records": tuple(active_records)})

    return spans


def cell_has_meaningful_text(value: str) -> bool:
    return bool(strip_markup(value).strip())


def append_display_suffix(value: str, suffix: str) -> str:
    if not value:
        return suffix

    lines = value.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip():
            lines[index] = f"{lines[index]} {suffix}"
            return "\n".join(lines)

    if lines:
        lines[-1] = suffix
        return "\n".join(lines)
    return suffix


def build_virtual_cell_display_text(cell: dict[str, Any], show_cell_ids: bool) -> str:
    text = str(cell.get("text") or "")
    if not cell_has_meaningful_text(text):
        return ""

    if show_cell_ids and not cell.get("color"):
        return str(cell["cell_id"])

    preview = text.strip("\n")
    if cell.get("color"):
        color_token = f"[{cell['color']}]"
        return append_display_suffix(preview, color_token) if preview else color_token
    return preview


def split_display_lines(value: str) -> list[str]:
    if not value:
        return [""]
    lines = value.splitlines()
    return lines if lines else [value]


def distribute_width_increase(widths: list[int], start_index: int, span: int, deficit: int) -> None:
    if deficit <= 0 or span <= 0:
        return
    base_increase = deficit // span
    remaining = deficit % span
    for offset in range(span):
        widths[start_index + offset] += base_increase
        if offset < remaining:
            widths[start_index + offset] += 1


def compute_virtual_column_widths(
    rows: list[dict[str, Any]],
    column_count: int,
    show_cell_ids: bool,
) -> list[int]:
    widths = [1] * column_count
    for row in rows:
        for cell in row["cells"]:
            text = build_virtual_cell_display_text(cell, show_cell_ids)
            if not text:
                continue

            needed_width = max(len(line) for line in split_display_lines(text))
            start_index = int(cell["col"]) - 1
            colspan = int(cell.get("colspan") or 1)
            current_width = sum(widths[start_index:start_index + colspan]) + (3 * (colspan - 1))
            deficit = needed_width - current_width
            distribute_width_increase(widths, start_index, colspan, deficit)
    return widths


def render_virtual_boundary(widths: list[int]) -> str:
    return "+" + "+".join("-" * (width + 2) for width in widths) + "+"


def render_virtual_row(
    row: dict[str, Any],
    widths: list[int],
    show_cell_ids: bool,
) -> list[str]:
    segments: list[tuple[int, list[str]]] = []
    for cell in row["cells"]:
        start_index = int(cell["col"]) - 1
        colspan = int(cell.get("colspan") or 1)
        inner_width = sum(widths[start_index:start_index + colspan]) + (3 * (colspan - 1))
        lines = split_display_lines(build_virtual_cell_display_text(cell, show_cell_ids))
        segments.append((inner_width, lines))

    row_height = max((len(lines) for _, lines in segments), default=1)
    rendered_lines: list[str] = []
    for line_index in range(row_height):
        parts = ["|"]
        for inner_width, lines in segments:
            line = lines[line_index] if line_index < len(lines) else ""
            parts.append(f" {line.ljust(inner_width)} ")
            parts.append("|")
        rendered_lines.append("".join(parts))
    return rendered_lines


def render_virtual_table(rows: list[dict[str, Any]], column_count: int) -> str:
    if not rows or column_count <= 0:
        return "(empty table)"

    show_cell_ids = any(cell.get("color") is not None for row in rows for cell in row["cells"])
    widths = compute_virtual_column_widths(rows, column_count, show_cell_ids)
    boundary = render_virtual_boundary(widths)
    rendered_lines = [boundary]
    for row in rows:
        rendered_lines.extend(render_virtual_row(row, widths, show_cell_ids))
        rendered_lines.append(boundary)
    return "\n".join(rendered_lines)


def build_single_table_group_component(
    record: dict[str, Any],
    cell_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows, cell_lookup = build_table_rows(str(record["table_id"]), dict(record["table_entry"]), cell_map)
    return {
        "kind": "table_group",
        "merged_as_single_table": False,
        "source_chunk_file_names": [record["txt_file_name"]],
        "source_table_ids": [record["table_id"]],
        "rendered_text": record["content"].rstrip() or str(record["table_entry"].get("rendered_table") or "").rstrip(),
        "rows": rows,
        "cell_lookup": cell_lookup,
        "logical_column_count": logical_column_count(rows),
        "row_count": len(rows),
    }


def build_merged_table_group_component(
    table_run: list[dict[str, Any]],
    cell_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_chunk_file_names = [record["txt_file_name"] for record in table_run]
    source_table_ids = [record["table_id"] for record in table_run]

    table_rows_by_record: list[list[dict[str, Any]]] = []
    group_column_count = 0
    show_cell_ids = False
    for record in table_run:
        rows, cell_lookup = build_table_rows(str(record["table_id"]), dict(record["table_entry"]), cell_map)
        table_rows_by_record.append(rows)
        group_column_count = max(group_column_count, logical_column_count(rows))
        if any(cell.get("color") is not None for cell in cell_lookup.values()):
            show_cell_ids = True

    merged_rows: list[dict[str, Any]] = []
    merged_cell_lookup: dict[str, dict[str, Any]] = {}
    next_row_index = 1

    for rows in table_rows_by_record:
        for row in rows:
            ordered_cells = sorted(row["cells"], key=lambda value: (value["col"], value["cell_id"]))
            merged_cells: list[dict[str, Any]] = []
            if row["is_full_width_title"] and ordered_cells:
                source_cell = ordered_cells[0]
                display_text = build_virtual_cell_display_text(source_cell, show_cell_ids)
                merged_cell = {
                    **source_cell,
                    "row": next_row_index,
                    "col": 1,
                    "rowspan": 1,
                    "colspan": max(group_column_count, 1),
                    "display_text": display_text,
                    "displayed_in_chunk": cell_is_visible(
                        {"cell_id": source_cell["cell_id"], "display_text": display_text}
                    ),
                }
                merged_cells.append(merged_cell)
                merged_cell_lookup[merged_cell["cell_id"]] = merged_cell
            else:
                for logical_col, source_cell in enumerate(ordered_cells, start=1):
                    display_text = build_virtual_cell_display_text(source_cell, show_cell_ids)
                    merged_cell = {
                        **source_cell,
                        "row": next_row_index,
                        "col": logical_col,
                        "rowspan": 1,
                        "colspan": 1,
                        "display_text": display_text,
                        "displayed_in_chunk": cell_is_visible(
                            {"cell_id": source_cell["cell_id"], "display_text": display_text}
                        ),
                    }
                    merged_cells.append(merged_cell)
                    merged_cell_lookup[merged_cell["cell_id"]] = merged_cell

            merged_rows.append(
                {
                    "row_index": next_row_index,
                    "cells": merged_cells,
                    "is_full_width_title": row["is_full_width_title"],
                }
            )
            next_row_index += 1

    return {
        "kind": "table_group",
        "merged_as_single_table": True,
        "source_chunk_file_names": source_chunk_file_names,
        "source_table_ids": source_table_ids,
        "rendered_text": render_virtual_table(merged_rows, max(group_column_count, 1)),
        "rows": merged_rows,
        "cell_lookup": merged_cell_lookup,
        "logical_column_count": max(group_column_count, 1),
        "row_count": len(merged_rows),
    }


def build_table_group_component(
    table_run: list[dict[str, Any]],
    cell_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(table_run) == 1:
        return build_single_table_group_component(table_run[0], cell_map)
    return build_merged_table_group_component(table_run, cell_map)


def build_span_components(
    span_records: tuple[dict[str, Any], ...],
    cell_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    table_run: list[dict[str, Any]] = []

    for record in span_records:
        if record["chunk_type"] == "table" and record.get("table_id") and record.get("table_entry"):
            table_run.append(record)
            continue

        if table_run:
            components.append(build_table_group_component(table_run, cell_map))
            table_run = []

        components.append(
            {
                "kind": "text",
                "source_chunk_file_names": [record["txt_file_name"]],
                "content": record["content"].rstrip(),
            }
        )

    if table_run:
        components.append(build_table_group_component(table_run, cell_map))

    return components


def build_component_summaries(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for component in components:
        if component["kind"] == "table_group":
            summaries.append(
                {
                    "kind": "table_group",
                    "merged_as_single_table": component["merged_as_single_table"],
                    "source_chunk_file_names": list(component["source_chunk_file_names"]),
                    "source_table_ids": list(component["source_table_ids"]),
                    "logical_column_count": int(component["logical_column_count"]),
                    "row_count": int(component["row_count"]),
                }
            )
            continue

        summaries.append(
            {
                "kind": "text",
                "source_chunk_file_names": list(component["source_chunk_file_names"]),
                "character_count": len(component["content"]),
            }
        )
    return summaries


def build_inspection_input_content(components: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for component in components:
        if component["kind"] == "table_group":
            content = component["rendered_text"].rstrip()
        else:
            content = component["content"].rstrip()
        if content:
            parts.append(content)
    return "\n\n".join(parts).rstrip() + ("\n" if parts else "")


def build_input_excerpt(content: str, line_limit: int = INPUT_EXCERPT_LINE_LIMIT) -> str:
    lines = content.splitlines()
    if len(lines) <= line_limit:
        return content
    return "\n".join(lines[:line_limit]) + "\n..."


def build_span_file_name(section_id: str, span_index: int, span_records: tuple[dict[str, Any], ...]) -> str:
    first_stem = Path(span_records[0]["txt_file_name"]).stem if span_records else "span"
    last_stem = Path(span_records[-1]["txt_file_name"]).stem if span_records else "span"
    middle = first_stem if first_stem == last_stem else f"{first_stem}__{last_stem}"
    return f"{section_id}_span_{span_index:02d}_{middle}.txt"


def write_inspection_input_file(
    document_path: Path,
    file_name: str,
    content: str,
) -> Path:
    input_dir = document_path / DEFAULT_INSPECTION_INPUT_DIRNAME
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / file_name
    input_path.write_text(content, encoding="utf-8")
    return input_path


def select_primary_table_group(
    components: list[dict[str, Any]],
    section_schema: SectionHeaderSchema,
) -> tuple[dict[str, Any] | None, list[tuple[str, ...]]]:
    candidates: list[tuple[float, int, int, dict[str, Any], list[tuple[str, ...]]]] = []
    table_group_position = 0
    for component in components:
        if component["kind"] != "table_group":
            continue
        allowed_orders = section_schema.orders_for_column_count(int(component["logical_column_count"]))
        if not allowed_orders:
            table_group_position += 1
            continue
        _, _, best_score = find_best_header_match(component["rows"], allowed_orders)
        candidates.append(
            (
                best_score,
                int(component["row_count"]),
                -table_group_position,
                component,
                allowed_orders,
            )
        )
        table_group_position += 1

    if not candidates:
        return None, []

    candidates.sort(key=lambda value: (value[0], value[1], value[2]), reverse=True)
    _, _, _, component, allowed_orders = candidates[0]
    return component, allowed_orders


def build_row_excerpt(rows: list[dict[str, Any]], limit: int = PROMPT_ROW_LIMIT) -> list[dict[str, Any]]:
    excerpt: list[dict[str, Any]] = []
    for row in rows[:limit]:
        excerpt.append(
            {
                "row_index": row["row_index"],
                "is_full_width_title": row["is_full_width_title"],
                "cells": [
                    {
                        "cell_id": cell["cell_id"],
                        "col": cell["col"],
                        "rowspan": cell["rowspan"],
                        "colspan": cell["colspan"],
                        "color": cell["color"],
                        "displayed_in_chunk": cell["displayed_in_chunk"],
                        "plain_text": cell["plain_text"],
                        "nested_table_ids": cell["nested_table_ids"],
                    }
                    for cell in row["cells"]
                ],
            }
        )
    return excerpt


def build_rendered_excerpt(rendered_table: str, line_limit: int = 18) -> str:
    lines = rendered_table.splitlines()
    if len(lines) <= line_limit:
        return rendered_table
    return "\n".join(lines[:line_limit]) + "\n..."


def format_column_order(columns: list[str] | tuple[str, ...]) -> str:
    return " -> ".join(columns)


def find_best_header_match(
    rows: list[dict[str, Any]],
    allowed_orders: list[tuple[str, ...]],
) -> tuple[dict[str, Any] | None, tuple[str, ...] | None, float]:
    best_row: dict[str, Any] | None = None
    best_order: tuple[str, ...] | None = None
    best_score = 0.0

    for row in rows[:PROMPT_ROW_LIMIT]:
        if row["is_full_width_title"]:
            continue
        for order in allowed_orders:
            if len(row["cells"]) != len(order):
                continue
            scores = [label_similarity(cell["text"], label) for cell, label in zip(row["cells"], order)]
            if not scores:
                continue
            row_score = min(scores) + (sum(scores) / len(scores))
            if row_score > best_score:
                best_row = row
                best_order = order
                best_score = row_score

    return best_row, best_order, best_score / 2 if best_score else 0.0


def build_heuristic_resolution(
    section_id: str,
    allowed_orders: list[tuple[str, ...]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    default_order = list(allowed_orders[0])
    matched_row, matched_order, matched_score = find_best_header_match(rows, allowed_orders)
    if matched_row is not None and matched_order is not None and matched_score >= 0.9:
        header_cell_ids = [cell["cell_id"] for cell in matched_row["cells"]]
        visible = all(cell["displayed_in_chunk"] for cell in matched_row["cells"])
        header_state = "visible_valid" if visible else "hidden_valid"
        return {
            "status": "resolved",
            "inspection_required": not visible,
            "header_state": header_state,
            "valid_column_order": list(matched_order),
            "actual_header_row_exists": True,
            "actual_header_row_index": matched_row["row_index"],
            "actual_header_cell_ids": header_cell_ids,
            "brief_description": (
                f"Valid column order is {format_column_order(matched_order)}. "
                f"Actual header cells are {', '.join(header_cell_ids)} on row {matched_row['row_index']} "
                f"and they are {'visible' if visible else 'hidden'} in the extracted chunk."
            ),
        }

    return {
        "status": "resolved",
        "inspection_required": True,
        "header_state": "missing",
        "valid_column_order": default_order,
        "actual_header_row_exists": False,
        "actual_header_row_index": None,
        "actual_header_cell_ids": [],
        "brief_description": (
            f"No actual header row is present in this {section_id} chunk. "
            f"Use the canonical column order {format_column_order(default_order)}."
        ),
    }


def build_user_prompt(
    document_name: str,
    chunk_file_name: str,
    section_id: str,
    table_entry: dict[str, Any],
    section_schema: SectionHeaderSchema,
    allowed_orders: list[tuple[str, ...]],
    rows: list[dict[str, Any]],
    heuristic_resolution: dict[str, Any],
) -> str:
    section_meta = SECTION_INDEX[section_id]
    row_excerpt = build_row_excerpt(rows)
    payload = {
        "document_name": document_name,
        "chunk_file_name": chunk_file_name,
        "section_id": section_id,
        "section_arabic": section_meta["arabic"],
        "section_english": section_meta["english"],
        "section_notes": section_schema.notes,
        "table_row_count": int(table_entry["row_count"]),
        "table_column_count": int(table_entry["column_count"]),
        "allowed_canonical_orders": [list(order) for order in allowed_orders],
        "source_chunk_file_names": table_entry.get("source_chunk_file_names", []),
        "primary_table_source_chunk_file_names": table_entry.get("primary_group_source_chunk_file_names", []),
        "input_components": table_entry.get("input_components", []),
        "inspection_input_excerpt": table_entry.get("inspection_input_excerpt", ""),
        "rendered_table_excerpt": build_rendered_excerpt(str(table_entry["rendered_table"])),
        "row_excerpt": row_excerpt,
        "heuristic_hint": heuristic_resolution,
    }
    return (
        "Inspect this classified SOP section span and resolve the final valid column order.\n"
        "Possible situations:\n"
        "1. No actual header row exists in this input.\n"
        "2. The actual header row exists and is valid, but hidden in the inspection input.\n"
        "3. The actual header row exists and is visible, but invalid or corrupt.\n"
        "4. The actual header row exists, is hidden, and is invalid or corrupt.\n"
        "5. The actual header row exists, is visible, and is already valid.\n\n"
        "Return JSON with this exact shape:\n"
        '{"status":"resolved","inspection_required":true,"header_state":"visible_valid","valid_column_order":["..."],"actual_header_row_exists":true,"actual_header_row_index":2,"actual_header_cell_ids":["CL000001"],"brief_description":"..."}\n\n'
        "Rules for the JSON:\n"
        "- status must always be \"resolved\".\n"
        "- header_state must be one of visible_valid, hidden_valid, visible_invalid, hidden_invalid, missing.\n"
        "- valid_column_order must exactly match one allowed canonical order.\n"
        "- If actual_header_row_exists is false, actual_header_row_index must be null and actual_header_cell_ids must be empty.\n"
        "- If actual_header_row_exists is true, actual_header_cell_ids must list the real header cells from left to right.\n"
        "- brief_description must briefly describe the final valid order and whether any cell ids are actual header cells.\n\n"
        "Table context:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def parse_inspection_payload(
    response_text: str,
    allowed_orders: list[tuple[str, ...]],
    cell_lookup: dict[str, dict[str, Any]],
    heuristic_resolution: dict[str, Any],
) -> dict[str, Any]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Response JSON must be an object.")

    required_keys = {
        "status",
        "inspection_required",
        "header_state",
        "valid_column_order",
        "actual_header_row_exists",
        "actual_header_row_index",
        "actual_header_cell_ids",
        "brief_description",
    }
    extra_keys = set(payload) - required_keys
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(f"Missing keys in response JSON: {sorted(missing_keys)}")
    if extra_keys:
        raise ValueError(f"Unexpected keys in response JSON: {sorted(extra_keys)}")

    if payload["status"] != "resolved":
        raise ValueError("The 'status' field must be 'resolved'.")
    if not isinstance(payload["inspection_required"], bool):
        raise ValueError("The 'inspection_required' field must be a boolean.")

    header_state = payload["header_state"]
    if header_state not in ALLOWED_HEADER_STATES:
        raise ValueError(f"Unknown header_state returned: {header_state}")

    valid_column_order = payload["valid_column_order"]
    if not isinstance(valid_column_order, list) or not all(isinstance(value, str) for value in valid_column_order):
        raise ValueError("The 'valid_column_order' field must be a list of strings.")
    if tuple(valid_column_order) not in set(allowed_orders):
        raise ValueError(
            "The 'valid_column_order' field must exactly match one of the allowed canonical orders."
        )

    if not isinstance(payload["actual_header_row_exists"], bool):
        raise ValueError("The 'actual_header_row_exists' field must be a boolean.")

    actual_header_cell_ids = payload["actual_header_cell_ids"]
    if not isinstance(actual_header_cell_ids, list) or not all(isinstance(value, str) for value in actual_header_cell_ids):
        raise ValueError("The 'actual_header_cell_ids' field must be a list of strings.")

    brief_description = payload["brief_description"]
    if not isinstance(brief_description, str) or not brief_description.strip():
        raise ValueError("The 'brief_description' field must be a non-empty string.")

    header_row_index = payload["actual_header_row_index"]
    if payload["actual_header_row_exists"]:
        if header_row_index is None or not isinstance(header_row_index, int):
            raise ValueError("A resolved header row must include an integer 'actual_header_row_index'.")
        if not actual_header_cell_ids:
            raise ValueError("A resolved header row must include 'actual_header_cell_ids'.")
        if len(actual_header_cell_ids) != len(valid_column_order):
            raise ValueError("Header cell count must match the resolved column count.")
        if len(set(actual_header_cell_ids)) != len(actual_header_cell_ids):
            raise ValueError("Header cell ids must be unique.")
        for cell_id in actual_header_cell_ids:
            if cell_id not in cell_lookup:
                raise ValueError(f"Unknown header cell id returned: {cell_id}")

        header_rows = {cell_lookup[cell_id]["row"] for cell_id in actual_header_cell_ids}
        if len(header_rows) != 1:
            raise ValueError("All header cell ids must belong to the same row.")
        actual_row_index = next(iter(header_rows))
        if actual_row_index != header_row_index:
            raise ValueError("The returned header row index does not match the supplied cell ids.")

        ordered_row_cells = sorted(
            [cell for cell in cell_lookup.values() if cell["row"] == actual_row_index],
            key=lambda value: (value["col"], value["cell_id"]),
        )
        ordered_row_ids = [cell["cell_id"] for cell in ordered_row_cells]
        if ordered_row_ids != actual_header_cell_ids:
            raise ValueError("Header cell ids must be listed from left to right for the actual header row.")

        all_visible = all(cell_is_visible(cell_lookup[cell_id]) for cell_id in actual_header_cell_ids)
        if header_state.startswith("visible") and not all_visible:
            raise ValueError("A visible header row must use header cells that are visible in the rendered chunk.")
        if header_state.startswith("hidden") and all_visible:
            raise ValueError("A hidden header row must include at least one header cell hidden in the rendered chunk.")
        if header_state == "missing":
            raise ValueError("The 'missing' header state cannot include actual header cells.")
    else:
        if header_state != "missing":
            raise ValueError("Only the 'missing' state can omit actual header cells.")
        if header_row_index is not None:
            raise ValueError("A missing header row must set 'actual_header_row_index' to null.")
        if actual_header_cell_ids:
            raise ValueError("A missing header row must not include 'actual_header_cell_ids'.")

    if payload["inspection_required"] != (header_state != "visible_valid"):
        raise ValueError("inspection_required must be false only for 'visible_valid' and true otherwise.")

    heuristic_header_ids = heuristic_resolution["actual_header_cell_ids"]
    if heuristic_header_ids and heuristic_resolution["header_state"] in {"visible_valid", "hidden_valid"}:
        if payload["actual_header_cell_ids"] != heuristic_header_ids:
            raise ValueError(
                "The exact header row is already identifiable from the cell map; reuse that row's cell ids."
            )
        if payload["header_state"] != heuristic_resolution["header_state"]:
            raise ValueError(
                "The exact header row is already identifiable from the cell map; keep its visible or hidden state."
            )
        if payload["valid_column_order"] != heuristic_resolution["valid_column_order"]:
            raise ValueError(
                "The exact header row is already identifiable from the cell map; keep the canonical column order."
            )

    return payload


def request_inspection_resolution(
    client: Any,
    model: str,
    document_name: str,
    chunk_file_name: str,
    section_id: str,
    table_entry: dict[str, Any],
    section_schema: SectionHeaderSchema,
    allowed_orders: list[tuple[str, ...]],
    rows: list[dict[str, Any]],
    cell_lookup: dict[str, dict[str, Any]],
    heuristic_resolution: dict[str, Any],
    max_llm_retries: int,
) -> tuple[dict[str, Any] | None, int, list[dict[str, str]]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": INSPECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                document_name=document_name,
                chunk_file_name=chunk_file_name,
                section_id=section_id,
                table_entry=table_entry,
                section_schema=section_schema,
                allowed_orders=allowed_orders,
                rows=rows,
                heuristic_resolution=heuristic_resolution,
            ),
        },
    ]
    invalid_attempts: list[dict[str, str]] = []
    attempt_count = 0

    while True:
        attempt_count += 1
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=messages,
            )
        except NotFoundError as exc:
            raise RuntimeError(
                "Received 404 Not Found from the chat completions endpoint while running column-header inspection. "
                "Check that the base URL points to an OpenAI-compatible chat route and that the model exists."
            ) from exc

        response_text = (response.choices[0].message.content or "").strip()
        try:
            parsed_payload = parse_inspection_payload(
                response_text=response_text,
                allowed_orders=allowed_orders,
                cell_lookup=cell_lookup,
                heuristic_resolution=heuristic_resolution,
            )
            return parsed_payload, attempt_count, invalid_attempts
        except ValueError as exc:
            invalid_attempts.append(
                {
                    "attempt": str(attempt_count),
                    "error": str(exc),
                    "response": response_text,
                }
            )
            if max_llm_retries > 0 and attempt_count >= max_llm_retries:
                return None, attempt_count, invalid_attempts
            messages.extend(
                [
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid. "
                            f"Validation error: {exc}. "
                            "Return only valid JSON with the exact shape requested earlier."
                        ),
                    },
                ]
            )


def build_skipped_result(
    document_name: str,
    section_id: str,
    span_records: tuple[dict[str, Any], ...],
    inspection_input_path: Path,
    input_components: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    section_meta = SECTION_INDEX[section_id]
    return {
        "document_name": document_name,
        "txt_file_name": span_records[0]["txt_file_name"] if span_records else None,
        "relative_path": span_records[0]["relative_path"] if span_records else None,
        "predicted_sections": list(span_records[0]["predicted_sections"]) if span_records else [],
        "source_chunk_file_names": [record["txt_file_name"] for record in span_records],
        "source_relative_paths": [record["relative_path"] for record in span_records],
        "inspection_input_file_name": inspection_input_path.name,
        "inspection_input_path": project_relative_path(inspection_input_path),
        "input_components": input_components,
        "inspected_section_id": section_id,
        "inspected_section_arabic": section_meta["arabic"],
        "inspected_section_english": section_meta["english"],
        "status": "skipped",
        "skip_reason": reason,
        "table_id": None,
        "table_ids": [],
        "table_row_count": None,
        "table_column_count": None,
    }


def inspect_classified_document(
    document_path: Path,
    classification_output_path: Path,
    output_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "",
    max_llm_retries: int = 6,
    quiet: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    artifact_dir = find_chunk_artifact_dir(document_path)
    if artifact_dir is None:
        raise FileNotFoundError(
            f"Could not find chunk artifacts under {document_path} or {document_path / DEFAULT_OUTPUT_DIRNAME}"
        )

    classification_payload = load_json(classification_output_path)
    table_map_path = artifact_dir / DEFAULT_TABLE_MAP_NAME
    cell_map_path = artifact_dir / DEFAULT_CELL_MAP_NAME
    table_map = load_json(table_map_path)
    cell_map = load_json(cell_map_path)

    chunk_table_lookup: dict[str, tuple[str, dict[str, Any]]] = {}
    for table_id, table_entry in table_map.items():
        if not isinstance(table_entry, dict):
            continue
        chunk_file_name = table_entry.get("chunk_file_name")
        if isinstance(chunk_file_name, str) and chunk_file_name:
            chunk_table_lookup[chunk_file_name] = (table_id, table_entry)

    classification_results = classification_payload.get("results")
    if not isinstance(classification_results, list):
        raise ValueError("Classification output must contain a 'results' list.")

    chunk_records = build_chunk_records(classification_results, chunk_table_lookup)

    inspection_spans: list[dict[str, Any]] = []
    for section_id in HEADER_SECTION_SCHEMAS:
        inspection_spans.extend(build_section_spans(chunk_records, section_id))

    inspection_spans.sort(
        key=lambda span: (
            span["records"][0]["index"] if span["records"] else 999999,
            span["section_id"],
            span["span_index"],
        )
    )

    inspectable_targets: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    input_dir_path = document_path / DEFAULT_INSPECTION_INPUT_DIRNAME

    for span in inspection_spans:
        section_id = span["section_id"]
        span_records = span["records"]
        components = build_span_components(span_records, cell_map)
        input_content = build_inspection_input_content(components)
        input_file_name = build_span_file_name(section_id, span["span_index"], span_records)
        inspection_input_path = write_inspection_input_file(document_path, input_file_name, input_content)
        input_components = build_component_summaries(components)

        primary_table_group, allowed_orders = select_primary_table_group(
            components,
            HEADER_SECTION_SCHEMAS[section_id],
        )
        if primary_table_group is None:
            expected_counts = sorted(
                {len(order) for order in HEADER_SECTION_SCHEMAS[section_id].canonical_orders}
            )
            results.append(
                build_skipped_result(
                    document_name=document_path.name,
                    section_id=section_id,
                    span_records=span_records,
                    inspection_input_path=inspection_input_path,
                    input_components=input_components,
                    reason=(
                        f"No table component in this section span matched the expected logical column counts {expected_counts}."
                    ),
                )
            )
            continue

        inspectable_targets.append(
            {
                "section_id": section_id,
                "span_records": span_records,
                "inspection_input_path": inspection_input_path,
                "inspection_input_excerpt": build_input_excerpt(input_content),
                "input_components": input_components,
                "primary_table_group": primary_table_group,
                "allowed_orders": allowed_orders,
            }
        )

    client = None
    resolved_base_url = None
    if inspectable_targets:
        client, resolved_base_url = initialize_client(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        if not quiet and resolved_base_url.rstrip("/") != base_url.rstrip("/"):
            print(
                f"Resolved base URL from '{base_url}' to '{resolved_base_url}' after endpoint probing.",
                file=sys.stderr,
                flush=True,
            )

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "inspection",
                "current": 0,
                "total": len(inspectable_targets),
                "message": "Preparing table checks",
            }
        )

    fallback_count = 0
    resolved_count = 0
    for index, target in enumerate(inspectable_targets, start=1):
        section_id = target["section_id"]
        span_records = target["span_records"]
        primary_table_group = target["primary_table_group"]
        rows = primary_table_group["rows"]
        cell_lookup = primary_table_group["cell_lookup"]
        allowed_orders = target["allowed_orders"]

        inspection_table_entry = {
            "row_count": int(primary_table_group["row_count"]),
            "column_count": int(primary_table_group["logical_column_count"]),
            "rendered_table": primary_table_group["rendered_text"],
            "source_chunk_file_names": [record["txt_file_name"] for record in span_records],
            "primary_group_source_chunk_file_names": list(primary_table_group["source_chunk_file_names"]),
            "input_components": target["input_components"],
            "inspection_input_excerpt": target["inspection_input_excerpt"],
        }
        heuristic_resolution = build_heuristic_resolution(section_id, allowed_orders, rows)

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "inspection",
                    "current": index - 1,
                    "total": len(inspectable_targets),
                    "section_id": section_id,
                    "message": f"Checking table {index} of {len(inspectable_targets)}",
                }
            )

        llm_resolution, attempt_count, invalid_attempts = request_inspection_resolution(
            client=client,
            model=model,
            document_name=document_path.name,
            chunk_file_name=target["inspection_input_path"].name,
            section_id=section_id,
            table_entry=inspection_table_entry,
            section_schema=HEADER_SECTION_SCHEMAS[section_id],
            allowed_orders=allowed_orders,
            rows=rows,
            cell_lookup=cell_lookup,
            heuristic_resolution=heuristic_resolution,
            max_llm_retries=max_llm_retries,
        )

        resolved_by = "llm"
        resolution = llm_resolution
        if resolution is None:
            resolution = heuristic_resolution
            resolved_by = "heuristic_fallback"
            fallback_count += 1

        section_meta = SECTION_INDEX[section_id]
        results.append(
            {
                "document_name": document_path.name,
                "txt_file_name": span_records[0]["txt_file_name"],
                "relative_path": span_records[0]["relative_path"],
                "predicted_sections": list(span_records[0]["predicted_sections"]),
                "source_chunk_file_names": [record["txt_file_name"] for record in span_records],
                "source_relative_paths": [record["relative_path"] for record in span_records],
                "inspection_input_file_name": target["inspection_input_path"].name,
                "inspection_input_path": project_relative_path(target["inspection_input_path"]),
                "input_components": target["input_components"],
                "inspected_section_id": section_id,
                "inspected_section_arabic": section_meta["arabic"],
                "inspected_section_english": section_meta["english"],
                "table_id": primary_table_group["source_table_ids"][0],
                "table_ids": list(primary_table_group["source_table_ids"]),
                "table_group_source_chunk_file_names": list(primary_table_group["source_chunk_file_names"]),
                "table_group_was_merged": bool(primary_table_group["merged_as_single_table"]),
                "table_row_count": int(primary_table_group["row_count"]),
                "table_column_count": int(primary_table_group["logical_column_count"]),
                "allowed_canonical_orders": [list(order) for order in allowed_orders],
                "resolved_by": resolved_by,
                "llm_attempt_count": attempt_count,
                "invalid_attempts": invalid_attempts,
                "resolution": resolution,
            }
        )
        resolved_count += 1

        if not quiet:
            print(
                f"[{index}/{len(inspectable_targets)}] {target['inspection_input_path'].name} | "
                f"section={section_id} | state={resolution['header_state']} | resolved_by={resolved_by}",
                file=sys.stderr,
                flush=True,
            )

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "inspection",
                    "current": index,
                    "total": len(inspectable_targets),
                    "section_id": section_id,
                    "message": f"Checked table {index} of {len(inspectable_targets)}",
                }
            )

    payload = {
        "summary": {
            "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "document_name": document_path.name,
            "document_path": project_relative_path(document_path),
            "artifact_dir": project_relative_path(artifact_dir),
            "classification_output_path": project_relative_path(classification_output_path),
            "table_map_path": project_relative_path(table_map_path),
            "cell_map_path": project_relative_path(cell_map_path),
            "inspection_input_dir": project_relative_path(input_dir_path),
            "model": model if inspectable_targets else None,
            "resolved_base_url": resolved_base_url,
            "section_span_count": len(inspection_spans),
            "inspectable_result_count": len(inspectable_targets),
            "resolved_result_count": resolved_count,
            "fallback_result_count": fallback_count,
            "skipped_result_count": sum(1 for result in results if result.get("status") == "skipped"),
        },
        "results": results,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, payload)
        if not quiet:
            print(f"Wrote column-header inspection output to {output_path}", file=sys.stderr, flush=True)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect classified SOP tables and resolve their column-header order."
    )
    parser.add_argument(
        "document",
        help=(
            "Document directory name under documents/, an explicit extracted document directory path, "
            "or a source .docx path whose extracted directory shares the same stem under documents/."
        ),
    )
    parser.add_argument("--documents-root", default="documents", help="Directory holding extracted document folders.")
    parser.add_argument(
        "--classification-output",
        default=None,
        help="Path to the saved classification JSON. Defaults to <document>/classification_output.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the saved inspection JSON. Defaults to <document>/column_header_inspection.json.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key. Defaults to OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--max-llm-retries",
        type=int,
        default=6,
        help="Maximum number of LLM retries before falling back to deterministic resolution.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging on stderr.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    documents_root = Path(args.documents_root)
    if not documents_root.is_absolute():
        documents_root = PROJECT_ROOT / documents_root

    document_path = resolve_document_path(args.document, documents_root)
    classification_output_path = Path(args.classification_output) if args.classification_output else document_path / DEFAULT_CLASSIFICATION_OUTPUT_NAME
    if not classification_output_path.is_absolute():
        classification_output_path = PROJECT_ROOT / classification_output_path

    output_path = Path(args.output) if args.output else document_path / DEFAULT_INSPECTION_OUTPUT_NAME
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    inspect_classified_document(
        document_path=document_path,
        classification_output_path=classification_output_path,
        output_path=output_path,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_llm_retries=args.max_llm_retries,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
