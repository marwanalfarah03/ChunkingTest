from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SECTION_FILE_PREFIX = {
    "SEC02": "document_header",
    "SEC03": "document_control",
    "SEC04": "document_approval",
    "SEC05": "version_control",
    "SEC07": "purpose",
    "SEC08": "scope",
    "SEC09": "responsibilities",
    "SEC10": "key_references",
    "SEC11": "general_instruction",
    "SEC12": "procedure",
    "SEC13": "control_procedure",
    "SEC14": "reports_statements",
    "SEC15": "files_retention_periods",
    "SEC16": "appendices",
    "SEC17": "forms_used",
    "SEC18": "definitions_terms_abbreviations",
    "SEC19": "forms_templates",
    "SEC99": "other",
}

SECTION_ENGLISH = {
    "SEC02": "Document Header",
    "SEC03": "Document Control",
    "SEC04": "Document Approval",
    "SEC05": "Version Control",
    "SEC07": "Purpose",
    "SEC08": "Scope",
    "SEC09": "Responsibilities",
    "SEC10": "Key References",
    "SEC11": "General Instructions",
    "SEC12": "Procedures / Workflow Steps",
    "SEC13": "Control Procedures",
    "SEC14": "Reports & Statements",
    "SEC15": "Files & Retention Periods",
    "SEC16": "Appendices",
    "SEC17": "Forms Used",
    "SEC18": "Definitions / Terms / Abbreviations",
    "SEC19": "Forms / Templates",
    "SEC99": "Other / Unclassified",
}

CL_TOKEN_RE = re.compile(r"CL\d{6}")
TB_TOKEN_RE = re.compile(r"<TB\d{6}>")
EM_TOKEN_RE = re.compile(r"<EM\d{6}>")
REFERENCE_TOKEN_RE = re.compile(r"(CL\d{6}|<TB\d{6}>|<EM\d{6}>)")
COLOR_TOKEN_RE = re.compile(r"\s*\[#([0-9A-Fa-f]{6})\]\s*")
HTML_TAG_RE = re.compile(r"</?[^>]+>")
INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\ufeff]")
UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def text_quality(value: str) -> int:
    arabic = sum(1 for ch in value if "\u0600" <= ch <= "\u06ff")
    mojibake = sum(value.count(marker) for marker in ("\u00d8", "\u00d9", "\u00c3", "\u00c2", "\u00e2"))
    replacement = value.count("\ufffd")
    return arabic * 4 - mojibake * 3 - replacement * 8


def repair_text(value: Any) -> str:
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
    return best.replace("\uf0b7", "-").replace("\uf0a7", "-").replace("\u00a0", " ")


def plain_label(value: Any) -> str:
    text = repair_text(value)
    text = COLOR_TOKEN_RE.sub(" ", text)
    text = HTML_TAG_RE.sub("", text)
    text = html.unescape(text)
    text = INVISIBLE_CHARS_RE.sub("", text)
    return normalize_text(text)


def normalize_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    compacted: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and compacted:
                compacted.append("")
            previous_blank = True
            continue
        compacted.append(line)
        previous_blank = False
    return "\n".join(compacted).strip()


def safe_filename_part(value: str) -> str:
    cleaned = UNSAFE_FILENAME_RE.sub("_", value)
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" ._")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "chunk"


def slugify(value: str) -> str:
    cleaned = repair_text(value).replace("&", " and ")
    cleaned = re.sub(r"[/\\]+", " ", cleaned)
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned).strip("_").lower()
    return re.sub(r"_+", "_", cleaned) or "section"


def section_file_prefix(section_id: str, section_english: str) -> str:
    return SECTION_FILE_PREFIX.get(section_id) or slugify(section_english or section_id)


def strip_document_run_suffix(document_name: str) -> str:
    return re.sub(r"(?:[_\s-]+)?\d{8}_\d{6}(?:_\d+)?$", "", repair_text(document_name)).strip(" _-")


def find_sop_code(document_name: str) -> str | None:
    match = re.search(r"\bSOP\W*([0-9]{2,6})\b", repair_text(document_name), flags=re.IGNORECASE)
    return match.group(1) if match else None


def list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


@dataclass
class ResolvedValue:
    text: str = ""
    cell_ids: set[str] = field(default_factory=set)
    table_ids: set[str] = field(default_factory=set)
    asset_ids: set[str] = field(default_factory=set)
    unresolved_refs: set[str] = field(default_factory=set)

    def merge(self, other: "ResolvedValue") -> None:
        if other.text:
            if self.text:
                self.text = f"{self.text}\n{other.text}"
            else:
                self.text = other.text
        self.cell_ids.update(other.cell_ids)
        self.table_ids.update(other.table_ids)
        self.asset_ids.update(other.asset_ids)
        self.unresolved_refs.update(other.unresolved_refs)


class PlainTextResolver:
    def __init__(
        self,
        *,
        cell_map: dict[str, Any] | None = None,
        table_map: dict[str, Any] | None = None,
        asset_map: dict[str, Any] | None = None,
    ) -> None:
        self.cell_map = cell_map or {}
        self.table_map = table_map or {}
        self.asset_map = asset_map or {}

    def resolve(self, value: Any, *, seen_cells: set[str] | None = None, seen_tables: set[str] | None = None) -> ResolvedValue:
        if value is None:
            return ResolvedValue()
        if isinstance(value, list):
            resolved = ResolvedValue()
            for item in value:
                resolved.merge(self.resolve(item, seen_cells=seen_cells, seen_tables=seen_tables))
            return resolved
        if isinstance(value, dict):
            resolved = ResolvedValue()
            for key, item in value.items():
                if key in {"type", "section_id", "header_cell_ids"}:
                    continue
                item_resolved = self.resolve(item, seen_cells=seen_cells, seen_tables=seen_tables)
                if item_resolved.text:
                    label = plain_label(key)
                    item_resolved.text = f"{label}: {item_resolved.text}" if label else item_resolved.text
                resolved.merge(item_resolved)
            return resolved

        raw = str(value)
        if not raw:
            return ResolvedValue()

        seen_cells = seen_cells or set()
        seen_tables = seen_tables or set()
        result = ResolvedValue()
        text_parts: list[str] = []
        for part in REFERENCE_TOKEN_RE.split(raw):
            if not part:
                continue
            if CL_TOKEN_RE.fullmatch(part):
                resolved = self.resolve_cell(part, seen_cells=seen_cells, seen_tables=seen_tables)
                result.merge(resolved)
                if resolved.text:
                    text_parts.append(resolved.text)
            elif TB_TOKEN_RE.fullmatch(part):
                table_id = part[1:-1]
                resolved = self.resolve_table(table_id, seen_cells=seen_cells, seen_tables=seen_tables)
                result.merge(resolved)
                if resolved.text:
                    text_parts.append(resolved.text)
            elif EM_TOKEN_RE.fullmatch(part):
                asset_id = part[1:-1]
                resolved = self.resolve_asset(asset_id)
                result.merge(resolved)
                if resolved.text:
                    text_parts.append(resolved.text)
            else:
                cleaned = plain_label(part)
                if cleaned:
                    text_parts.append(cleaned)
        result.text = normalize_text("\n".join(text_parts))
        return result

    def resolve_cell(self, cell_id: str, *, seen_cells: set[str], seen_tables: set[str]) -> ResolvedValue:
        result = ResolvedValue(cell_ids={cell_id})
        if cell_id in seen_cells:
            return result
        cell = self.cell_map.get(cell_id)
        if not isinstance(cell, dict):
            result.text = cell_id
            result.unresolved_refs.add(cell_id)
            return result

        next_seen_cells = set(seen_cells)
        next_seen_cells.add(cell_id)
        text_value = cell.get("text")
        if not text_value and cell.get("display_text"):
            text_value = cell.get("display_text")
        resolved = self.resolve(text_value or "", seen_cells=next_seen_cells, seen_tables=seen_tables)
        result.merge(resolved)

        raw_text = str(text_value or "")
        for table_id in list_value(cell.get("nested_table_ids")):
            if f"<{table_id}>" in raw_text:
                continue
            nested = self.resolve_table(table_id, seen_cells=next_seen_cells, seen_tables=seen_tables)
            result.merge(nested)
        return result

    def resolve_asset(self, asset_id: str) -> ResolvedValue:
        result = ResolvedValue(asset_ids={asset_id})
        asset = self.asset_map.get(asset_id)
        if not isinstance(asset, dict):
            result.text = f"Embedded asset: {asset_id}"
            result.unresolved_refs.add(asset_id)
            return result
        label = repair_text(asset.get("original_name") or asset.get("stored_name") or asset_id)
        content_type = repair_text(asset.get("content_type") or "")
        result.text = normalize_text(f"Embedded asset: {label}" + (f" ({content_type})" if content_type else ""))
        return result

    def cells_for_table(self, table_id: str) -> list[dict[str, Any]]:
        cells: list[dict[str, Any]] = []
        for cell_id, payload in self.cell_map.items():
            if isinstance(payload, dict) and payload.get("table_id") == table_id:
                cells.append({"cell_id": cell_id, **payload})
        return sorted(cells, key=lambda item: (int(item.get("row") or 0), int(item.get("col") or 0), item["cell_id"]))

    def resolve_table(self, table_id: str, *, seen_cells: set[str], seen_tables: set[str]) -> ResolvedValue:
        result = ResolvedValue(table_ids={table_id})
        if table_id in seen_tables:
            return result
        table = self.table_map.get(table_id)
        cells = self.cells_for_table(table_id)
        if not isinstance(table, dict) and not cells:
            result.text = table_id
            result.unresolved_refs.add(table_id)
            return result

        next_seen_tables = set(seen_tables)
        next_seen_tables.add(table_id)
        rows: dict[int, list[dict[str, Any]]] = {}
        for cell in cells:
            rows.setdefault(int(cell.get("row") or 0), []).append(cell)

        row_lines: list[str] = []
        for row_index in sorted(rows):
            row_values: list[str] = []
            for cell in sorted(rows[row_index], key=lambda item: (int(item.get("col") or 0), item["cell_id"])):
                cell_result = self.resolve_cell(str(cell["cell_id"]), seen_cells=seen_cells, seen_tables=next_seen_tables)
                result.merge(cell_result)
                if cell_result.text:
                    row_values.append(cell_result.text.replace("\n", " "))
            if row_values:
                row_lines.append(" | ".join(row_values))
        result.text = normalize_text("\n".join(row_lines))
        return result


def workflow_columns(section_json: dict[str, Any], entries: list[Any]) -> list[str]:
    columns = [plain_label(column) for column in list_value(section_json.get("columns"))]
    columns = [column for column in columns if column]
    if len(columns) >= 2:
        return columns
    for entry in entries:
        if isinstance(entry, dict) and entry.get("type") == "step":
            keys = [plain_label(key) for key in entry if key != "type"]
            keys = [key for key in keys if key]
            if len(keys) >= 2:
                return keys
    return ["Step", "Owner"]


def get_entry_value_by_label(entry: dict[str, Any], label: str, ordinal: int) -> Any:
    for key, value in entry.items():
        if key == "type":
            continue
        if plain_label(key) == label:
            return value
    values = [value for key, value in entry.items() if key != "type"]
    return values[ordinal] if ordinal < len(values) else None


def build_hierarchy_path(hierarchy: list[dict[str, Any]]) -> str:
    labels = [str(item.get("title") or "").strip() for item in hierarchy if str(item.get("title") or "").strip()]
    return " > ".join(labels)


def format_number_path(indices: list[int]) -> str:
    return "_".join(f"{max(0, index):02d}" for index in indices)


def next_order_prefix(order_state: dict[str, int]) -> str:
    order_state["value"] = int(order_state.get("value") or 0) + 1
    return f"{order_state['value']:04d}"


def build_chunk_text(metadata: dict[str, Any], content_blocks: list[tuple[str, str]]) -> str:
    section_parts = [
        str(metadata.get("section_arabic") or "").strip(),
        str(metadata.get("section_english") or "").strip(),
    ]
    section_label = " - ".join(part for part in section_parts if part)
    content_lines = [
        f"Document: {metadata.get('document_name', '')}",
        f"Section: {section_label}",
        f"Hierarchy: {metadata.get('hierarchy_path', '')}",
        f"Chunk type: {metadata.get('chunk_type', '')}",
        "",
    ]

    for label, value in content_blocks:
        clean_label = plain_label(label)
        clean_value = normalize_text(value)
        if not clean_value:
            continue
        if clean_label:
            content_lines.append(f"{clean_label}:")
        content_lines.append(clean_value)
        content_lines.append("")

    content = normalize_text("\n".join(content_lines))
    metadata["content_character_count"] = len(content)
    return content + "\n"


def base_metadata(
    *,
    payload: dict[str, Any],
    result: dict[str, Any],
    section_json: dict[str, Any],
    section_id: str,
    chunk_type: str,
    hierarchy: list[dict[str, Any]],
    indices: dict[str, int],
    output_file_name: str,
    resolved_values: list[ResolvedValue],
) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    original_document_name = repair_text(summary.get("document_name") or result.get("document_name") or "")
    document_name = strip_document_run_suffix(original_document_name)
    section_heading = plain_label(section_json.get("section_heading") or "")
    if not section_heading:
        section_heading = SECTION_ENGLISH.get(section_id, section_id)
    section_arabic = repair_text(result.get("section_arabic") or "")
    section_english = repair_text(result.get("section_english") or SECTION_ENGLISH.get(section_id, section_id))

    merged = ResolvedValue()
    for item in resolved_values:
        merged.merge(item)

    return {
        "chunk_id": Path(output_file_name).stem,
        "chunk_type": chunk_type,
        "document_name": document_name,
        "original_document_name": original_document_name,
        "document_path": repair_text(summary.get("document_path") or ""),
        "document_sop_code": find_sop_code(document_name),
        "run_timestamp_utc": repair_text(summary.get("run_timestamp_utc") or ""),
        "model": repair_text(summary.get("model") or ""),
        "section_id": section_id,
        "section_arabic": section_arabic,
        "section_english": section_english,
        "section_heading": section_heading,
        "span_index": result.get("span_index"),
        "hierarchy": hierarchy,
        "hierarchy_path": build_hierarchy_path(hierarchy),
        "hierarchy_indices": indices,
        "source_chunk_file_names": list_value(result.get("source_chunk_file_names")),
        "source_relative_paths": list_value(result.get("source_relative_paths")),
        "source_cell_ids": sorted(merged.cell_ids),
        "source_table_ids": sorted(merged.table_ids),
        "source_asset_ids": sorted(merged.asset_ids),
        "unresolved_references": sorted(merged.unresolved_refs),
        "output_file_name": output_file_name,
    }


def write_chunk(
    *,
    output_dir: Path,
    output_file_name: str,
    metadata: dict[str, Any],
    content_blocks: list[tuple[str, str]],
) -> dict[str, Any]:
    output_path = output_dir / output_file_name
    output_path.write_text(build_chunk_text(metadata, content_blocks), encoding="utf-8")
    return {
        "chunk_id": metadata["chunk_id"],
        "file_name": output_file_name,
        "file_path": str(output_path),
        "section_id": metadata["section_id"],
        "chunk_type": metadata["chunk_type"],
        "hierarchy_path": metadata["hierarchy_path"],
        "content_character_count": metadata["content_character_count"],
    }


def unique_file_name(candidate: str, used_names: set[str]) -> str:
    path = Path(candidate)
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    counter = 2
    while True:
        next_candidate = f"{path.stem}_{counter:02d}{path.suffix}"
        if next_candidate not in used_names:
            used_names.add(next_candidate)
            return next_candidate
        counter += 1


def ordered_file_name(base_name: str, *, order_state: dict[str, int], used_names: set[str]) -> str:
    return unique_file_name(f"{next_order_prefix(order_state)}_{base_name}", used_names)


def section_hierarchy(document_name: str, section_id: str, section_heading: str) -> list[dict[str, Any]]:
    return [
        {"level": 0, "type": "document", "title": document_name},
        {
            "level": 1,
            "type": "section",
            "section_id": section_id,
            "title": section_heading or SECTION_ENGLISH.get(section_id, section_id),
        },
    ]


def merge_sources(items: list[ResolvedValue], *, text: str = "") -> ResolvedValue:
    merged = ResolvedValue(text=normalize_text(text))
    for item in items:
        merged.cell_ids.update(item.cell_ids)
        merged.table_ids.update(item.table_ids)
        merged.asset_ids.update(item.asset_ids)
        merged.unresolved_refs.update(item.unresolved_refs)
    return merged


def resolve_rows_text(rows: list[Any], columns: list[str], resolver: PlainTextResolver) -> ResolvedValue:
    lines: list[str] = []
    resolved_items: list[ResolvedValue] = []
    for row_index, row in enumerate(rows, start=1):
        row_lines: list[str] = [f"Row {row_index}:"]
        if isinstance(row, dict):
            row_columns = columns or [plain_label(key) for key in row if key != "type"]
            if isinstance(row.get("cells"), list):
                for cell_index, cell in enumerate(row["cells"], start=1):
                    label = row_columns[cell_index - 1] if cell_index <= len(row_columns) else f"Column {cell_index}"
                    resolved = resolver.resolve(cell)
                    resolved_items.append(resolved)
                    if resolved.text:
                        row_lines.append(f"{label}: {resolved.text}")
            else:
                for key in [key for key in row if key != "type"]:
                    label = plain_label(key)
                    resolved = resolver.resolve(row.get(key))
                    resolved_items.append(resolved)
                    if resolved.text:
                        row_lines.append(f"{label}: {resolved.text}")
        elif isinstance(row, list):
            for cell_index, cell in enumerate(row, start=1):
                label = columns[cell_index - 1] if cell_index <= len(columns) else f"Column {cell_index}"
                resolved = resolver.resolve(cell)
                resolved_items.append(resolved)
                if resolved.text:
                    row_lines.append(f"{label}: {resolved.text}")
        else:
            resolved = resolver.resolve(row)
            resolved_items.append(resolved)
            if resolved.text:
                row_lines.append(resolved.text)
        if len(row_lines) > 1:
            lines.extend(row_lines)
            lines.append("")
    return merge_sources(resolved_items, text="\n".join(lines))


def resolve_section_content(section_payload: Any, resolver: PlainTextResolver) -> tuple[list[tuple[str, str]], list[ResolvedValue]]:
    if isinstance(section_payload, list):
        resolved = resolver.resolve(section_payload)
        return [("Content", resolved.text)] if resolved.text else [], [resolved]
    if not isinstance(section_payload, dict):
        resolved = resolver.resolve(section_payload)
        return [("Content", resolved.text)] if resolved.text else [], [resolved]

    blocks: list[tuple[str, str]] = []
    resolved_values: list[ResolvedValue] = []
    if "content" in section_payload:
        resolved = resolver.resolve(section_payload.get("content"))
        resolved_values.append(resolved)
        if resolved.text:
            blocks.append(("Content", resolved.text))

    columns = [plain_label(column) for column in list_value(section_payload.get("columns"))]
    rows = section_payload.get("rows")
    if isinstance(rows, list):
        resolved = resolve_rows_text(rows, columns, resolver)
        resolved_values.append(resolved)
        if resolved.text:
            blocks.append(("Rows", resolved.text))

    skipped = {"section_id", "section_heading", "columns", "header_cell_ids", "format", "status", "content", "rows"}
    for key, value in section_payload.items():
        if key in skipped:
            continue
        resolved = resolver.resolve(value)
        resolved_values.append(resolved)
        if resolved.text:
            blocks.append((plain_label(key), resolved.text))
    return blocks, resolved_values


def export_workflow_entries(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    result: dict[str, Any],
    section_json: dict[str, Any],
    resolver: PlainTextResolver,
    used_names: set[str],
    order_state: dict[str, int],
) -> list[dict[str, Any]]:
    section_id = str(result.get("section_id") or section_json.get("section_id") or "")
    section_english = repair_text(result.get("section_english") or SECTION_ENGLISH.get(section_id, section_id))
    prefix = section_file_prefix(section_id, section_english)
    entries = section_json.get("entries") if isinstance(section_json.get("entries"), list) else []
    columns = workflow_columns(section_json, entries)
    chunks: list[dict[str, Any]] = []
    document_name = strip_document_run_suffix((payload.get("summary") or {}).get("document_name") or result.get("document_name") or "")
    section_heading = plain_label(section_json.get("section_heading") or SECTION_ENGLISH.get(section_id, section_id))

    header_levels = {
        "group_header": 1,
        "subgroup_header": 2,
        "sub_subgroup_header": 3,
        "subsubgroup_header": 3,
    }
    level_counters: list[int] = []
    active_indices: list[int] = []
    active_headers: list[dict[str, Any]] = []
    item_index = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_type = str(entry.get("type") or "")
        if entry_type in header_levels:
            level = min(header_levels[entry_type], len(level_counters) + 1)
            level_counters = level_counters[:level]
            while len(level_counters) < level:
                level_counters.append(0)
            level_counters[level - 1] += 1
            active_indices = level_counters[:]
            active_headers = active_headers[: level - 1]
            active_headers.append(
                {
                    "level": level + 1,
                    "type": entry_type,
                    "index": active_indices[-1],
                    "index_path": active_indices[:],
                    "title": plain_label(entry.get("heading")),
                }
            )
            item_index = 0
            continue
        if entry_type != "step":
            continue

        field_values: list[tuple[str, ResolvedValue]] = []
        for index, column in enumerate(columns):
            resolved = resolver.resolve(get_entry_value_by_label(entry, column, index))
            field_values.append((column, resolved))
        if not any(resolved.text for _, resolved in field_values):
            continue

        item_index += 1
        hierarchy_numbers = active_indices + [item_index]
        base_name = f"{prefix}_{format_number_path(hierarchy_numbers)}.txt"
        file_name = ordered_file_name(base_name, order_state=order_state, used_names=used_names)
        hierarchy = section_hierarchy(document_name, section_id, section_heading)
        hierarchy.extend(active_headers)
        indices = {"path": hierarchy_numbers}
        metadata = base_metadata(
            payload=payload,
            result=result,
            section_json=section_json,
            section_id=section_id,
            chunk_type="workflow_step",
            hierarchy=hierarchy,
            indices=indices,
            output_file_name=file_name,
            resolved_values=[resolved for _, resolved in field_values],
        )
        chunks.append(
            write_chunk(
                output_dir=output_dir,
                output_file_name=file_name,
                metadata=metadata,
                content_blocks=[(label, resolved.text) for label, resolved in field_values],
            )
        )
    return chunks


def export_general_instruction_entries(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    result: dict[str, Any],
    section_json: dict[str, Any],
    resolver: PlainTextResolver,
    used_names: set[str],
    order_state: dict[str, int],
) -> list[dict[str, Any]]:
    section_id = str(result.get("section_id") or section_json.get("section_id") or "")
    section_english = repair_text(result.get("section_english") or SECTION_ENGLISH.get(section_id, section_id))
    prefix = section_file_prefix(section_id, section_english)
    entries = section_json.get("entries") if isinstance(section_json.get("entries"), list) else []
    chunks: list[dict[str, Any]] = []
    document_name = strip_document_run_suffix((payload.get("summary") or {}).get("document_name") or result.get("document_name") or "")
    section_heading = plain_label(section_json.get("section_heading") or SECTION_ENGLISH.get(section_id, section_id))

    subsection_index = 0
    item_index = 0
    active_heading = ""
    for entry in entries:
        if not isinstance(entry, dict):
            value = entry
            entry_type = "content"
            heading = ""
        else:
            entry_type = str(entry.get("type") or "content")
            heading = plain_label(entry.get("heading") or "")
            value = entry.get("value")
            if value is None and isinstance(entry.get("values"), list):
                value = entry.get("values")

        if entry_type == "subsection":
            subsection_index += 1
            item_index = 0
            active_heading = heading
        elif entry_type not in {"content", "subsection"}:
            heading = heading or plain_label(entry_type)

        resolved = resolver.resolve(value)
        if not resolved.text:
            continue

        item_index += 1
        hierarchy_numbers = ([subsection_index] if subsection_index else []) + [item_index]
        base_name = f"{prefix}_{format_number_path(hierarchy_numbers)}.txt"
        file_name = ordered_file_name(base_name, order_state=order_state, used_names=used_names)
        hierarchy = section_hierarchy(document_name, section_id, section_heading)
        if active_heading:
            hierarchy.append(
                {
                    "level": 2,
                    "type": "subsection",
                    "index": subsection_index,
                    "index_path": [subsection_index],
                    "title": active_heading,
                }
            )
        indices = {"path": hierarchy_numbers}
        metadata = base_metadata(
            payload=payload,
            result=result,
            section_json=section_json,
            section_id=section_id,
            chunk_type="general_instruction",
            hierarchy=hierarchy,
            indices=indices,
            output_file_name=file_name,
            resolved_values=[resolved],
        )
        label = heading if heading else "Instruction"
        chunks.append(
            write_chunk(
                output_dir=output_dir,
                output_file_name=file_name,
                metadata=metadata,
                content_blocks=[(label, resolved.text)],
            )
        )
    return chunks


def export_rows_or_content(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    result: dict[str, Any],
    section_json: dict[str, Any],
    resolver: PlainTextResolver,
    used_names: set[str],
    order_state: dict[str, int],
) -> list[dict[str, Any]]:
    section_id = str(result.get("section_id") or section_json.get("section_id") or "")
    section_english = repair_text(result.get("section_english") or SECTION_ENGLISH.get(section_id, section_id))
    prefix = section_file_prefix(section_id, section_english)
    document_name = strip_document_run_suffix((payload.get("summary") or {}).get("document_name") or result.get("document_name") or "")
    section_heading = plain_label(section_json.get("section_heading") or SECTION_ENGLISH.get(section_id, section_id))

    content_blocks, resolved_values = resolve_section_content(section_json, resolver)
    if not content_blocks:
        return []

    file_name = ordered_file_name(f"{prefix}.txt", order_state=order_state, used_names=used_names)
    hierarchy = section_hierarchy(document_name, section_id, section_heading)
    metadata = base_metadata(
        payload=payload,
        result=result,
        section_json=section_json,
        section_id=section_id,
        chunk_type="section_content",
        hierarchy=hierarchy,
        indices={"path": []},
        output_file_name=file_name,
        resolved_values=resolved_values,
    )
    return [
        write_chunk(
            output_dir=output_dir,
            output_file_name=file_name,
            metadata=metadata,
            content_blocks=content_blocks,
        )
    ]


def reset_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".txt", ".json"}:
            path.unlink()


def export_rag_txt_files(
    *,
    section_json_payload: dict[str, Any],
    output_dir: Path,
    table_map: dict[str, Any] | None = None,
    cell_map: dict[str, Any] | None = None,
    asset_map: dict[str, Any] | None = None,
    include_sections: set[str] | None = None,
) -> dict[str, Any]:
    reset_output_dir(output_dir)
    resolver = PlainTextResolver(cell_map=cell_map, table_map=table_map, asset_map=asset_map)

    chunks: list[dict[str, Any]] = []
    used_names: set[str] = set()
    order_state = {"value": 0}
    results = section_json_payload.get("results") if isinstance(section_json_payload.get("results"), list) else []
    for result in results:
        if not isinstance(result, dict) or result.get("status") != "resolved":
            continue
        section_id = str(result.get("section_id") or "")
        if include_sections is not None and section_id not in include_sections:
            continue
        section_json = result.get("section_json")
        if isinstance(section_json, dict) and section_json.get("status") == "not_found":
            continue
        if not isinstance(section_json, dict):
            section_json = {"section_id": section_id, "content": section_json}
        entries = section_json.get("entries")
        if section_id in {"SEC12", "SEC13"} and isinstance(entries, list):
            chunks.extend(
                export_workflow_entries(
                    output_dir=output_dir,
                    payload=section_json_payload,
                    result=result,
                    section_json=section_json,
                    resolver=resolver,
                    used_names=used_names,
                    order_state=order_state,
                )
            )
        elif section_id == "SEC11" and isinstance(entries, list):
            chunks.extend(
                export_general_instruction_entries(
                    output_dir=output_dir,
                    payload=section_json_payload,
                    result=result,
                    section_json=section_json,
                    resolver=resolver,
                    used_names=used_names,
                    order_state=order_state,
                )
            )
        else:
            chunks.extend(
                export_rows_or_content(
                    output_dir=output_dir,
                    payload=section_json_payload,
                    result=result,
                    section_json=section_json,
                    resolver=resolver,
                    used_names=used_names,
                    order_state=order_state,
                )
            )

    summary = section_json_payload.get("summary") if isinstance(section_json_payload.get("summary"), dict) else {}
    manifest = {
        "export_type": "rag_txt",
        "document_name": strip_document_run_suffix(summary.get("document_name") or ""),
        "document_path": repair_text(summary.get("document_path") or ""),
        "included_sections": sorted(include_sections) if include_sections is not None else "all",
        "output_dir": str(output_dir),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
