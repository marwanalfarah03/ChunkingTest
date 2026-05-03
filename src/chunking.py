from __future__ import annotations

import argparse
import io
import json
import mimetypes
import re
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable
from xml.etree import ElementTree as ET

import olefile
from oletools import oleobj



NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "v": "urn:schemas-microsoft-com:vml",
    "o": "urn:schemas-microsoft-com:office:office",
}

REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}

ASSETS_DIRNAME = "assets"
DOCUMENTS_DIRNAME = "documents"
DEFAULT_OUTPUT_DIRNAME = "chunks"
DEFAULT_SOURCE_DOCX_NAME = "source.docx"
DEFAULT_TABLE_MAP_NAME = "schema_table_map.json"
DEFAULT_CELL_MAP_NAME = "schema_cell_map.json"
DEFAULT_ASSET_MAP_NAME = "schema_asset_map.json"
PRESERVED_FORMATTING_TAG_PATTERN = re.compile(r"</?(?:strong|em|u)>", re.IGNORECASE)
INVISIBLE_TEXT_CHAR_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060\ufeff]")


@dataclass
class ParagraphEntry:
    body_index: int
    paragraph_index: int
    text: str


@dataclass
class TableCell:
    cell_id: str
    start_row: int
    start_col: int
    rowspan: int
    colspan: int
    text: str
    color: str | None
    element: ET.Element | None
    nested_table_ids: list[str]


@dataclass
class TableGrid:
    row_count: int
    column_count: int
    occupancy: list[list["TableCell"]]
    cells: list["TableCell"]


@dataclass
class TableEntry:
    table_id: str
    table_index: int
    row_count: int
    column_count: int
    location: str
    rendered_table: str
    is_nested: bool
    parent_table_id: str | None
    parent_cell_id: str | None


@dataclass(frozen=True)
class AssetEntry:
    asset_id: str
    kind: str
    original_name: str
    stored_name: str
    relative_path: str
    source_path: str
    content_type: str | None
    relationship_type: str | None


@dataclass
class EntryCounters:
    paragraph_index: int = 0
    table_index: int = 0
    cell_index: int = 0
    asset_index: int = 0


@dataclass
class ChunkEntry:
    chunk_index: int
    chunk_type: str  # "text" or "table"
    content: str
    table_id: str | None = None
    file_name: str | None = None


@dataclass
class ExportArtifacts:
    chunks: list[ChunkEntry]
    table_map: OrderedDict[str, dict[str, object]]
    cell_map: OrderedDict[str, dict[str, object]]
    asset_map: OrderedDict[str, dict[str, object]]


@dataclass(frozen=True)
class DocumentPaths:
    name: str
    directory: Path
    source_docx: Path
    chunks_dir: Path


ReportFn = Callable[[str], None]


@dataclass
class AssetExportContext:
    archive: zipfile.ZipFile
    output_dir: Path
    relative_dir: str
    relationships: dict[str, dict[str, str | None]]
    content_types: dict[str, str]
    counters: EntryCounters
    assets_by_source: dict[str, AssetEntry] = field(default_factory=dict)
    assets_by_id: OrderedDict[str, AssetEntry] = field(default_factory=OrderedDict)


@dataclass(frozen=True)
class NumberingLevel:
    level: int
    start: int
    num_fmt: str
    lvl_text: str
    suff: str | None
    left: int | None
    hanging: int | None


@dataclass
class NumberingContext:
    definitions: dict[str, dict[int, NumberingLevel]] = field(default_factory=dict)
    counters: dict[str, list[int]] = field(default_factory=dict)


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return package_dir().parent


def documents_root() -> Path:
    return project_root() / DOCUMENTS_DIRNAME


def default_output_dir_for_docx(docx_path: Path) -> Path:
    return documents_root() / docx_path.stem


def resolve_source_docx(document_dir: Path) -> Path:
    preferred = document_dir / DEFAULT_SOURCE_DOCX_NAME
    if preferred.exists():
        return preferred

    candidates = sorted(path for path in document_dir.glob("*.docx") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No .docx file found in {document_dir}")

    return candidates[0]


def build_document_paths(document_dir: Path) -> DocumentPaths:
    source_docx = resolve_source_docx(document_dir)

    return DocumentPaths(
        name=document_dir.name,
        directory=document_dir,
        source_docx=source_docx,
        chunks_dir=document_dir / DEFAULT_OUTPUT_DIRNAME,
    )


def list_document_paths(selected_names: list[str] | None = None) -> list[DocumentPaths]:
    root = documents_root()
    if not root.exists():
        return []

    if selected_names:
        document_dirs = [root / name for name in selected_names]
    else:
        document_dirs = sorted(path for path in root.iterdir() if path.is_dir())

    paths: list[DocumentPaths] = []
    missing_requested: list[str] = []

    for document_dir in document_dirs:
        if not document_dir.exists():
            missing_requested.append(document_dir.name)
            continue

        try:
            paths.append(build_document_paths(document_dir))
        except FileNotFoundError:
            if selected_names:
                raise

    if missing_requested:
        missing = ", ".join(sorted(missing_requested))
        raise FileNotFoundError(f"Requested document directories were not found: {missing}")

    return paths


def qn(tag: str) -> str:
    prefix, local_name = tag.split(":", 1)
    return f"{{{NS[prefix]}}}{local_name}"


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def derive_artifact_paths(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        output_dir / DEFAULT_TABLE_MAP_NAME,
        output_dir / DEFAULT_CELL_MAP_NAME,
        output_dir / DEFAULT_ASSET_MAP_NAME,
        output_dir / ASSETS_DIRNAME,
    )


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def next_table_id(counters: EntryCounters) -> str:
    counters.table_index += 1
    return f"TB{counters.table_index:06d}"


def next_cell_id(counters: EntryCounters) -> str:
    counters.cell_index += 1
    return f"CL{counters.cell_index:06d}"


def next_asset_id(counters: EntryCounters) -> str:
    counters.asset_index += 1
    return f"EM{counters.asset_index:06d}"


def read_docx_xml(docx_path: Path, member_name: str) -> ET.Element:
    with zipfile.ZipFile(docx_path) as archive:
        with archive.open(member_name) as handle:
            return ET.parse(handle).getroot()


def read_archive_xml(archive: zipfile.ZipFile, member_name: str) -> ET.Element:
    with archive.open(member_name) as handle:
        return ET.parse(handle).getroot()


def relationship_member_name(member_name: str) -> str:
    path = PurePosixPath(member_name)
    parent = "" if str(path.parent) == "." else f"{path.parent.as_posix()}/"
    return f"{parent}_rels/{path.name}.rels"


def resolve_relationship_target(member_name: str, target: str) -> str:
    base = PurePosixPath(member_name).parent
    resolved_parts: list[str] = []
    for part in [*base.parts, *PurePosixPath(target).parts]:
        if part in {"", "."}:
            continue
        if part == "..":
            if resolved_parts:
                resolved_parts.pop()
            continue
        resolved_parts.append(part)
    return PurePosixPath(*resolved_parts).as_posix()


def read_relationships(archive: zipfile.ZipFile, member_name: str) -> dict[str, dict[str, str | None]]:
    rel_member_name = relationship_member_name(member_name)
    try:
        rels_root = read_archive_xml(archive, rel_member_name)
    except KeyError:
        return {}

    relationships: dict[str, dict[str, str | None]] = {}
    for relationship in rels_root.findall("./pr:Relationship", REL_NS):
        rel_id = relationship.get("Id")
        target = relationship.get("Target")
        if not rel_id or not target:
            continue
        target_mode = relationship.get("TargetMode")
        resolved_target = None if (target_mode or "").lower() == "external" else resolve_relationship_target(member_name, target)
        relationships[rel_id] = {
            "type": relationship.get("Type"),
            "target": resolved_target,
            "target_mode": target_mode,
        }
    return relationships


def read_content_types(archive: zipfile.ZipFile) -> dict[str, str]:
    try:
        content_types_root = read_archive_xml(archive, "[Content_Types].xml")
    except KeyError:
        return {}

    default_types: dict[str, str] = {}
    override_types: dict[str, str] = {}
    for child in list(content_types_root):
        local_name = child.tag.split("}")[-1]
        if local_name == "Default":
            extension = (child.get("Extension") or "").lower()
            content_type = child.get("ContentType")
            if extension and content_type:
                default_types[extension] = content_type
        elif local_name == "Override":
            part_name = (child.get("PartName") or "").lstrip("/")
            content_type = child.get("ContentType")
            if part_name and content_type:
                override_types[part_name] = content_type

    content_types = dict(override_types)
    for part_name in archive.namelist():
        if part_name in content_types:
            continue
        suffix = PurePosixPath(part_name).suffix.lstrip(".").lower()
        if suffix in default_types:
            content_types[part_name] = default_types[suffix]
    return content_types


def theme_alias_to_scheme_name(alias: str) -> str | None:
    aliases = {
        "background1": "lt1",
        "text1": "dk1",
        "background2": "lt2",
        "text2": "dk2",
        "accent1": "accent1",
        "accent2": "accent2",
        "accent3": "accent3",
        "accent4": "accent4",
        "accent5": "accent5",
        "accent6": "accent6",
        "hyperlink": "hlink",
        "followedhyperlink": "folHlink",
    }
    return aliases.get(alias.lower())


def normalize_fill(raw_fill: str | None) -> str | None:
    if not raw_fill:
        return None

    fill = raw_fill.strip().upper()
    if fill in {"AUTO", "FFFFFF", "FFF", "TRANSPARENT", "NONE", "NIL", "CLEAR", "WINDOW", "WINDOWTEXT"}:
        return None
    if re.fullmatch(r"[0-9A-F]{3}", fill):
        fill = "".join(char * 2 for char in fill)
    if re.fullmatch(r"[0-9A-F]{6}", fill):
        return f"#{fill}"
    return fill


def read_theme_colors(docx_path: Path) -> dict[str, str]:
    try:
        theme_root = read_docx_xml(docx_path, "word/theme/theme1.xml")
    except KeyError:
        return {}

    clr_scheme = theme_root.find(".//a:clrScheme", NS)
    if clr_scheme is None:
        return {}

    theme_colors: dict[str, str] = {}
    for child in list(clr_scheme):
        name = child.tag.split("}")[-1]
        if not list(child):
            continue
        value_node = list(child)[0]
        value = value_node.get("lastClr") or value_node.get("val")
        if not value:
            continue
        normalized = normalize_fill(value)
        if normalized:
            theme_colors[name] = normalized
    return theme_colors


def parse_numbering_level(level_element: ET.Element, fallback_level: int = 0) -> NumberingLevel:
    level = parse_int(level_element.get(qn("w:ilvl")) or level_element.get("ilvl"))
    if level is None:
        level = fallback_level

    start_element = level_element.find("./w:start", NS)
    num_fmt_element = level_element.find("./w:numFmt", NS)
    lvl_text_element = level_element.find("./w:lvlText", NS)
    suff_element = level_element.find("./w:suff", NS)
    indent_element = level_element.find("./w:pPr/w:ind", NS)

    return NumberingLevel(
        level=level,
        start=parse_int(start_element.get(qn("w:val")) if start_element is not None else None) or 1,
        num_fmt=(num_fmt_element.get(qn("w:val")) if num_fmt_element is not None else None) or "decimal",
        lvl_text=(lvl_text_element.get(qn("w:val")) if lvl_text_element is not None else None) or f"%{level + 1}.",
        suff=(suff_element.get(qn("w:val")) if suff_element is not None else None),
        left=parse_int(indent_element.get(qn("w:left")) or indent_element.get("left") or indent_element.get(qn("w:start")) or indent_element.get("start")) if indent_element is not None else None,
        hanging=parse_int(indent_element.get(qn("w:hanging")) or indent_element.get("hanging")) if indent_element is not None else None,
    )


def read_numbering_definitions(docx_path: Path) -> dict[str, dict[int, NumberingLevel]]:
    try:
        numbering_root = read_docx_xml(docx_path, "word/numbering.xml")
    except KeyError:
        return {}

    abstract_levels: dict[str, dict[int, NumberingLevel]] = {}
    for abstract in numbering_root.findall("./w:abstractNum", NS):
        abstract_id = abstract.get(qn("w:abstractNumId")) or abstract.get("abstractNumId")
        if not abstract_id:
            continue
        levels: dict[int, NumberingLevel] = {}
        for level_element in abstract.findall("./w:lvl", NS):
            level = parse_numbering_level(level_element)
            levels[level.level] = level
        abstract_levels[abstract_id] = levels

    definitions: dict[str, dict[int, NumberingLevel]] = {}
    for num in numbering_root.findall("./w:num", NS):
        num_id = num.get(qn("w:numId")) or num.get("numId")
        abstract_ref = num.find("./w:abstractNumId", NS)
        abstract_id = abstract_ref.get(qn("w:val")) if abstract_ref is not None else None
        if not num_id or not abstract_id:
            continue

        levels = {index: replace(level) for index, level in abstract_levels.get(abstract_id, {}).items()}
        for override in num.findall("./w:lvlOverride", NS):
            level_index = parse_int(override.get(qn("w:ilvl")) or override.get("ilvl"))
            if level_index is None:
                continue

            level_override = override.find("./w:lvl", NS)
            if level_override is not None:
                levels[level_index] = parse_numbering_level(level_override, level_index)
                continue

            start_override = override.find("./w:startOverride", NS)
            if start_override is None or level_index not in levels:
                continue

            start_value = parse_int(start_override.get(qn("w:val")) or start_override.get("val"))
            if start_value is None:
                continue
            levels[level_index] = replace(levels[level_index], start=start_value)

        definitions[num_id] = levels

    return definitions


def bool_property(
    element: ET.Element | None,
    default: bool = False,
    false_values: Iterable[str] | None = None,
) -> bool:
    if element is None:
        return default
    value = element.get(qn("w:val")) or element.get("val")
    if value is None:
        return True
    normalized_false_values = {"0", "false", "off"}
    if false_values is not None:
        normalized_false_values.update(item.lower() for item in false_values)
    return value.lower() not in normalized_false_values


def to_alpha(value: int, uppercase: bool = False) -> str:
    if value <= 0:
        return "0"
    letters: list[str] = []
    current = value
    while current > 0:
        current -= 1
        letters.append(chr(ord("A") + (current % 26)))
        current //= 26
    result = "".join(reversed(letters))
    return result if uppercase else result.lower()


def to_roman(value: int, uppercase: bool = False) -> str:
    if value <= 0:
        return "0"
    numerals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    parts: list[str] = []
    current = value
    for amount, symbol in numerals:
        while current >= amount:
            parts.append(symbol)
            current -= amount
    result = "".join(parts)
    return result if uppercase else result.lower()


def format_number_value(value: int, num_fmt: str) -> str:
    match num_fmt:
        case "decimal" | "decimalZero":
            return str(value)
        case "lowerLetter":
            return to_alpha(value, uppercase=False)
        case "upperLetter":
            return to_alpha(value, uppercase=True)
        case "lowerRoman":
            return to_roman(value, uppercase=False)
        case "upperRoman":
            return to_roman(value, uppercase=True)
        case _:
            return str(value)


def normalize_bullet_glyph(value: str) -> str:
    bullet_map = {
        "\uf0b7": "•",
        "\uf0a7": "▪",
        "o": "○",
    }
    return bullet_map.get(value, value)


def format_list_label(num_id: str, level_index: int, numbering_context: NumberingContext) -> tuple[str, NumberingLevel] | None:
    levels = numbering_context.definitions.get(num_id)
    if not levels:
        return None

    level = levels.get(level_index)
    if level is None:
        return None

    counters = numbering_context.counters.setdefault(num_id, [0] * 9)
    for reset_index in range(level_index + 1, len(counters)):
        counters[reset_index] = 0
    if counters[level_index] < level.start - 1:
        counters[level_index] = level.start - 1
    counters[level_index] += 1

    if level.num_fmt == "bullet":
        label = normalize_bullet_glyph(level.lvl_text)
    else:
        def replace_level(match: re.Match[str]) -> str:
            reference_index = int(match.group(1)) - 1
            reference_level = levels.get(reference_index, level)
            reference_value = counters[reference_index]
            if reference_value <= 0:
                reference_value = reference_level.start
            return format_number_value(reference_value, reference_level.num_fmt)

        label = re.sub(r"%(\d+)", replace_level, level.lvl_text)

    suffix = {"tab": "\t", "space": " ", "nothing": ""}.get((level.suff or "").lower(), "")
    return label + suffix, level


def sanitize_asset_filename(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", value).strip()
    return cleaned or "asset.bin"


def build_stored_asset_name(output_dir: Path, original_name: str) -> str:
    base_name = sanitize_asset_filename(Path(original_name or "").name)
    suffix = "".join(Path(base_name).suffixes)
    stem = (base_name[: -len(suffix)] if suffix else base_name) or "asset"
    candidate = base_name
    index = 2
    while (output_dir / candidate).exists():
        candidate = f"{stem}_{index}{suffix}" if suffix else f"{stem}_{index}"
        index += 1
    return candidate


def guess_content_type(value: str) -> str | None:
    content_type, _ = mimetypes.guess_type(value, strict=False)
    return content_type


def infer_embedded_payload_details(raw_bytes: bytes) -> tuple[str | None, str | None]:
    if raw_bytes.startswith(b"%PDF-"):
        return ".pdf", "application/pdf"
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if raw_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif", "image/gif"
    if raw_bytes.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile:
            return ".zip", "application/zip"

        if "[Content_Types].xml" in names:
            if any(name.startswith("word/") for name in names):
                return ".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if any(name.startswith("xl/") for name in names):
                return ".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if any(name.startswith("ppt/") for name in names):
                return ".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        return ".zip", "application/zip"
    if raw_bytes.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return ".ole", "application/x-ole-storage"
    return None, None


def apply_extension_to_name(value: str, extension: str | None, fallback_stem: str) -> str:
    name = sanitize_asset_filename(Path(value).name)
    current_suffix = Path(name).suffix.lower()
    if not extension:
        return name
    if current_suffix and current_suffix not in {".bin", ".ole", ".dat", ".tmp"}:
        return name
    base_name = Path(name).stem if current_suffix else Path(fallback_stem).stem
    return sanitize_asset_filename(f"{base_name}{extension}")


def find_ole_native_stream_path(ole: object) -> list[str] | None:
    for stream_path in ole.listdir():
        if stream_path and stream_path[-1] == "\x01Ole10Native":
            return stream_path
    return None


def find_ole_contents_stream_path(ole: object) -> list[str] | None:
    for stream_path in ole.listdir():
        if stream_path and stream_path[-1].upper() == "CONTENTS":
            return stream_path
    return None


def extract_embedded_ole_asset(raw_bytes: bytes, fallback_name: str) -> tuple[str, bytes, str | None] | None:
    if olefile is None or oleobj is None:
        return None

    try:
        if not olefile.isOleFile(io.BytesIO(raw_bytes)):
            return None
    except Exception:
        return None

    try:
        with olefile.OleFileIO(io.BytesIO(raw_bytes)) as ole:
            stream_path = find_ole_native_stream_path(ole)
            if stream_path is not None:
                stream_bytes = ole.openstream(stream_path).read()
            else:
                contents_path = find_ole_contents_stream_path(ole)
                if contents_path is None:
                    return None
                contents_bytes = ole.openstream(contents_path).read()
                extension, content_type = infer_embedded_payload_details(contents_bytes)
                effective_name = apply_extension_to_name(fallback_name, extension, fallback_name)
                effective_content_type = content_type or guess_content_type(effective_name)
                return effective_name, contents_bytes, effective_content_type
    except Exception:
        return None

    for is_package in (False, True):
        try:
            native_stream = oleobj.OleNativeStream(stream_bytes, package=is_package)
        except Exception:
            continue

        if native_stream.is_link or native_stream.data is None:
            continue

        embedded_bytes = native_stream.data.read() if hasattr(native_stream.data, "read") else native_stream.data
        if not embedded_bytes:
            continue

        extension, inferred_content_type = infer_embedded_payload_details(embedded_bytes)
        embedded_name = apply_extension_to_name(native_stream.filename or fallback_name, extension, fallback_name)
        embedded_content_type = inferred_content_type or guess_content_type(embedded_name)
        return embedded_name, embedded_bytes, embedded_content_type

    return None


def detect_asset_kind(relationship_type: str | None, target_path: str, content_type: str | None) -> str | None:
    rel_type = (relationship_type or "").lower()
    target_lower = target_path.lower()
    content_type_lower = (content_type or "").lower()
    if "image" in rel_type or target_lower.startswith("word/media/") or content_type_lower.startswith("image/"):
        return "image"
    if "oleobject" in rel_type or rel_type.endswith("/package") or "/embeddings/" in target_lower:
        return "embedded"
    return None


def export_relationship_asset(asset_context: AssetExportContext, rel_id: str, preferred_kind: str | None = None) -> str | None:
    relationship = asset_context.relationships.get(rel_id)
    if relationship is None:
        return None

    target_path = relationship.get("target")
    if not target_path or relationship.get("target_mode"):
        return None

    if target_path in asset_context.assets_by_source:
        return asset_context.assets_by_source[target_path].asset_id

    content_type = asset_context.content_types.get(target_path)
    kind = preferred_kind or detect_asset_kind(relationship.get("type"), target_path, content_type)
    if kind is None:
        return None

    original_name = PurePosixPath(target_path).name or "asset.bin"
    with asset_context.archive.open(target_path) as source_handle:
        stored_bytes = source_handle.read()

    effective_content_type = content_type or guess_content_type(original_name)
    if kind != "image":
        extracted_asset = extract_embedded_ole_asset(stored_bytes, original_name)
        if extracted_asset is not None:
            original_name, stored_bytes, extracted_content_type = extracted_asset
            effective_content_type = extracted_content_type or effective_content_type

    asset_id = next_asset_id(asset_context.counters)
    stored_name = build_stored_asset_name(asset_context.output_dir, original_name)
    destination_path = asset_context.output_dir / stored_name
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("wb") as destination_handle:
        destination_handle.write(stored_bytes)

    relative_path = PurePosixPath(asset_context.relative_dir, stored_name).as_posix()
    entry = AssetEntry(
        asset_id=asset_id,
        kind=kind,
        original_name=original_name,
        stored_name=stored_name,
        relative_path=relative_path,
        source_path=target_path,
        content_type=effective_content_type,
        relationship_type=relationship.get("type"),
    )
    asset_context.assets_by_source[target_path] = entry
    asset_context.assets_by_id[asset_id] = entry
    return asset_id


def extract_asset_placeholder(element: ET.Element, asset_context: AssetExportContext | None) -> str | None:
    if asset_context is None:
        return None

    ole_object = element.find(".//o:OLEObject", NS)
    if ole_object is not None:
        rel_id = ole_object.get(qn("r:id")) or ole_object.get("id")
        if rel_id:
            asset_id = export_relationship_asset(asset_context, rel_id, preferred_kind="embedded")
            if asset_id:
                return f"<{asset_id}>"

    for xpath in (".//a:blip", ".//v:imagedata"):
        for node in element.findall(xpath, NS):
            rel_id = node.get(qn("r:embed")) or node.get(qn("r:id")) or node.get("embed") or node.get("id")
            if not rel_id:
                continue
            asset_id = export_relationship_asset(asset_context, rel_id, preferred_kind="image")
            if asset_id:
                return f"<{asset_id}>"

    return None


def iter_paragraph_runs(paragraph: ET.Element) -> Iterable[ET.Element]:
    for child in list(paragraph):
        if child.tag == qn("w:r"):
            yield child
        elif child.tag == qn("w:hyperlink"):
            for run in child.findall("./w:r", NS):
                yield run


def format_inline_text(text: str, bold: bool, italic: bool, underline: bool) -> str:
    if not text or not text.strip() or (not bold and not italic and not underline):
        return text
    leading_length = len(text) - len(text.lstrip(" \t"))
    trailing_length = len(text) - len(text.rstrip(" \t"))
    leading = text[:leading_length]
    core_end = len(text) - trailing_length if trailing_length else len(text)
    core = text[leading_length:core_end]
    trailing = text[core_end:]
    if not core:
        return text

    formatted = core
    if underline:
        formatted = f"<u>{formatted}</u>"
    if italic:
        formatted = f"<em>{formatted}</em>"
    if bold:
        formatted = f"<strong>{formatted}</strong>"
    return f"{leading}{formatted}{trailing}"


def render_paragraph_runs(paragraph: ET.Element, asset_context: AssetExportContext | None = None) -> str:
    paragraph_properties = paragraph.find("./w:pPr/w:rPr", NS)
    paragraph_bold = bool_property(paragraph_properties.find("./w:b", NS) if paragraph_properties is not None else None)
    paragraph_italic = bool_property(paragraph_properties.find("./w:i", NS) if paragraph_properties is not None else None)
    paragraph_underline = bool_property(
        paragraph_properties.find("./w:u", NS) if paragraph_properties is not None else None,
        false_values={"none"},
    )

    segments: list[list[object]] = []
    for run in iter_paragraph_runs(paragraph):
        run_properties = run.find("./w:rPr", NS)
        bold = bool_property(run_properties.find("./w:b", NS) if run_properties is not None else None, paragraph_bold)
        italic = bool_property(run_properties.find("./w:i", NS) if run_properties is not None else None, paragraph_italic)
        underline = bool_property(
            run_properties.find("./w:u", NS) if run_properties is not None else None,
            paragraph_underline,
            false_values={"none"},
        )

        for child in list(run):
            text = ""
            if child.tag == qn("w:t"):
                text = child.text or ""
            elif child.tag == qn("w:tab"):
                text = "\t"
            elif child.tag in {qn("w:br"), qn("w:cr")}:
                text = "\n"
            elif child.tag == qn("w:noBreakHyphen"):
                text = "-"
            elif child.tag == qn("w:sym"):
                char = child.get(qn("w:char")) or child.get("char")
                if char:
                    try:
                        text = chr(int(char, 16))
                    except ValueError:
                        text = ""
            elif child.tag in {qn("w:drawing"), qn("w:object"), qn("w:pict")}:
                placeholder = extract_asset_placeholder(child, asset_context)
                if placeholder:
                    segments.append([False, False, False, placeholder])
                continue

            if not text:
                continue
            if (
                segments
                and segments[-1][0] == bold
                and segments[-1][1] == italic
                and segments[-1][2] == underline
            ):
                segments[-1][3] = f"{segments[-1][3]}{text}"
            else:
                segments.append([bold, italic, underline, text])

    return "".join(
        format_inline_text(text, bool(bold), bool(italic), bool(underline))
        for bold, italic, underline, text in segments
    )


def cleanup_preserved_text(value: str) -> str:
    if not value:
        return ""
    return "\n".join(line.rstrip() for line in value.splitlines()).strip("\n")


def compute_paragraph_indent(paragraph: ET.Element, level: NumberingLevel | None) -> str:
    indent_element = paragraph.find("./w:pPr/w:ind", NS)
    direct_values = [
        parse_int(indent_element.get(qn("w:start")) or indent_element.get("start")) if indent_element is not None else None,
        parse_int(indent_element.get(qn("w:left")) or indent_element.get("left")) if indent_element is not None else None,
        parse_int(indent_element.get(qn("w:end")) or indent_element.get("end")) if indent_element is not None else None,
        parse_int(indent_element.get(qn("w:right")) or indent_element.get("right")) if indent_element is not None else None,
    ]
    base_indent = max((value for value in direct_values if value is not None), default=0)
    if level is not None and level.left is not None:
        base_indent = max(base_indent, max(0, level.left - (level.hanging or 0)))
    if level is not None:
        base_indent = max(base_indent, level.level * 360)
    return " " * max(0, base_indent // 180)


def paragraph_prefix(paragraph: ET.Element, numbering_context: NumberingContext) -> str:
    num_properties = paragraph.find("./w:pPr/w:numPr", NS)
    level: NumberingLevel | None = None
    label = ""
    if num_properties is not None:
        num_id_element = num_properties.find("./w:numId", NS)
        level_element = num_properties.find("./w:ilvl", NS)
        num_id = num_id_element.get(qn("w:val")) if num_id_element is not None else None
        level_index = parse_int(level_element.get(qn("w:val")) if level_element is not None else None) or 0
        if num_id:
            formatted = format_list_label(num_id, level_index, numbering_context)
            if formatted is not None:
                label, level = formatted

    indent = compute_paragraph_indent(paragraph, level)
    prefix = f"{indent}{label}"
    if label and not prefix.endswith((" ", "\t")):
        prefix += " "
    return prefix


def paragraph_text(
    paragraph: ET.Element,
    numbering_context: NumberingContext | None = None,
    asset_context: AssetExportContext | None = None,
) -> str:
    numbering_context = numbering_context or NumberingContext()
    text = cleanup_preserved_text(render_paragraph_runs(paragraph, asset_context))
    prefix = paragraph_prefix(paragraph, numbering_context)
    if not text:
        return prefix.rstrip()
    if not prefix:
        return text

    lines = text.split("\n")
    continuation_indent = " " * len(prefix.replace("\t", "    "))
    formatted_lines = [f"{prefix}{lines[0]}"]
    formatted_lines.extend(f"{continuation_indent}{line}" if line else "" for line in lines[1:])
    return "\n".join(formatted_lines)


def iter_block_children(container: ET.Element) -> Iterable[ET.Element]:
    for child in list(container):
        if child.tag in {qn("w:p"), qn("w:tbl")}:
            yield child


def cell_text(
    cell: ET.Element,
    numbering_context: NumberingContext | None = None,
    asset_context: AssetExportContext | None = None,
) -> str:
    texts = [paragraph_text(paragraph, numbering_context, asset_context) for paragraph in iter_block_children(cell) if paragraph.tag == qn("w:p")]
    return "\n".join(text for text in texts if text)


def resolve_theme_fill(theme_fill: str | None, theme_colors: dict[str, str]) -> str | None:
    if not theme_fill:
        return None

    resolved = theme_colors.get(theme_fill)
    if resolved:
        return resolved

    scheme_name = theme_alias_to_scheme_name(theme_fill)
    if not scheme_name:
        return None
    return theme_colors.get(scheme_name)


def cell_fill(cell: ET.Element, theme_colors: dict[str, str]) -> str | None:
    shading = cell.find("./w:tcPr/w:shd", NS)
    if shading is None:
        return None

    explicit_fill = normalize_fill(shading.get(qn("w:fill")) or shading.get("fill"))
    if explicit_fill:
        return explicit_fill

    theme_fill = shading.get(qn("w:themeFill")) or shading.get("themeFill")
    resolved_theme = resolve_theme_fill(theme_fill, theme_colors)
    return normalize_fill(resolved_theme)


def cell_content_with_tags(
    cell: ET.Element,
    numbering_context: NumberingContext | None = None,
    asset_context: AssetExportContext | None = None,
) -> str:
    parts: list[str] = []
    for child in iter_block_children(cell):
        if child.tag == qn("w:p"):
            text = paragraph_text(child, numbering_context, asset_context)
            if text:
                parts.append(text)
        elif child.tag == qn("w:tbl"):
            parts.append("{NESTED_TABLE}")
    return "\n".join(parts)


def table_grid_width(table: ET.Element) -> int:
    grid_columns = table.findall("./w:tblGrid/w:gridCol", NS)
    if grid_columns:
        return len(grid_columns)

    row_widths: list[int] = []
    for row in table.findall("./w:tr", NS):
        width = 0
        for cell in row.findall("./w:tc", NS):
            width += grid_span(cell)
        row_widths.append(width)
    return max(row_widths, default=0)


def grid_span(cell: ET.Element) -> int:
    span = cell.find("./w:tcPr/w:gridSpan", NS)
    if span is None:
        return 1
    value = span.get(qn("w:val")) or span.get("val") or "1"
    try:
        return max(1, int(value))
    except ValueError:
        return 1


def vmerge_state(cell: ET.Element) -> str | None:
    merge = cell.find("./w:tcPr/w:vMerge", NS)
    if merge is None:
        return None
    value = merge.get(qn("w:val")) or merge.get("val")
    return value or "continue"


def table_has_visible_fill(table_grid: TableGrid) -> bool:
    return any(cell.color is not None for cell in table_grid.cells)


def summarize_display_text(value: str) -> str:
    return cleanup_preserved_text(value)


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


def build_cell_display_text(cell: TableCell, show_cell_ids: bool) -> str:
    if not cell_has_meaningful_content(cell):
        return ""

    if show_cell_ids and not cell.color:
        return cell.cell_id

    preview = summarize_display_text(cell.text)
    if cell.color:
        color_token = f"[{cell.color}]"
        return append_display_suffix(preview, color_token) if preview else color_token
    return preview


def split_display_lines(value: str) -> list[str]:
    if not value:
        return [""]
    lines = value.splitlines()
    return lines if lines else [value]


def strip_preserved_formatting(value: str) -> str:
    if not value:
        return ""
    return PRESERVED_FORMATTING_TAG_PATTERN.sub("", value)


def strip_invisible_text(value: str) -> str:
    if not value:
        return ""
    return INVISIBLE_TEXT_CHAR_PATTERN.sub("", value)


def cell_has_meaningful_content(cell: TableCell) -> bool:
    normalized = strip_invisible_text(strip_preserved_formatting(cell.text)).strip()
    return bool(normalized)


def trim_blank_table_grid(table_grid: TableGrid) -> TableGrid:
    if table_grid.row_count == 0 or table_grid.column_count == 0:
        return table_grid

    keep_rows = [
        row_index
        for row_index, row in enumerate(table_grid.occupancy)
        if any(cell_has_meaningful_content(cell) for cell in row)
    ]
    keep_columns = [
        column_index
        for column_index in range(table_grid.column_count)
        if any(
            cell_has_meaningful_content(table_grid.occupancy[row_index][column_index])
            for row_index in range(table_grid.row_count)
        )
    ]

    if len(keep_rows) == table_grid.row_count and len(keep_columns) == table_grid.column_count:
        return table_grid
    if not keep_rows or not keep_columns:
        return TableGrid(row_count=0, column_count=0, occupancy=[], cells=[])

    trimmed_occupancy = [
        [table_grid.occupancy[row_index][column_index] for column_index in keep_columns]
        for row_index in keep_rows
    ]
    positions_by_cell: dict[int, list[tuple[int, int]]] = {}
    for row_index, row in enumerate(trimmed_occupancy):
        for column_index, cell in enumerate(row):
            positions_by_cell.setdefault(id(cell), []).append((row_index, column_index))

    trimmed_cells: list[TableCell] = []
    for cell in table_grid.cells:
        positions = positions_by_cell.get(id(cell))
        if not positions:
            continue
        rows = [row_index for row_index, _ in positions]
        columns = [column_index for _, column_index in positions]
        cell.start_row = min(rows)
        cell.start_col = min(columns)
        cell.rowspan = max(rows) - cell.start_row + 1
        cell.colspan = max(columns) - cell.start_col + 1
        trimmed_cells.append(cell)

    return TableGrid(
        row_count=len(trimmed_occupancy),
        column_count=len(keep_columns),
        occupancy=trimmed_occupancy,
        cells=trimmed_cells,
    )


def build_table_grid(
    table: ET.Element,
    theme_colors: dict[str, str],
    counters: EntryCounters,
    numbering_context: NumberingContext | None = None,
    asset_context: AssetExportContext | None = None,
) -> TableGrid:
    total_columns = table_grid_width(table)
    occupancy: list[list[TableCell]] = []
    cells: list[TableCell] = []
    active_vertical: dict[int, TableCell] = {}

    for row_index, row in enumerate(table.findall("./w:tr", NS)):
        current_row: list[TableCell | None] = [None] * total_columns
        extended_in_row: set[int] = set()
        column_index = 0

        for cell in row.findall("./w:tc", NS):
            while column_index < total_columns and current_row[column_index] is not None:
                column_index += 1

            span = grid_span(cell)
            merge = vmerge_state(cell)

            if merge == "continue":
                anchor = active_vertical.get(column_index)
                if anchor is None:
                    anchor = TableCell(
                        cell_id=next_cell_id(counters),
                        start_row=row_index,
                        start_col=column_index,
                        rowspan=1,
                        colspan=span,
                        text=cell_content_with_tags(cell, numbering_context, asset_context),
                        color=cell_fill(cell, theme_colors),
                        element=cell,
                        nested_table_ids=[],
                    )
                    cells.append(anchor)
                elif id(anchor) not in extended_in_row:
                    anchor.rowspan += 1
                    extended_in_row.add(id(anchor))
            else:
                anchor = TableCell(
                    cell_id=next_cell_id(counters),
                    start_row=row_index,
                    start_col=column_index,
                    rowspan=1,
                    colspan=span,
                    text=cell_content_with_tags(cell, numbering_context, asset_context),
                    color=cell_fill(cell, theme_colors),
                    element=cell,
                    nested_table_ids=[],
                )
                cells.append(anchor)

            for offset in range(span):
                target = column_index + offset
                if target >= total_columns:
                    break
                current_row[target] = anchor
                if merge in {"restart", "continue"}:
                    active_vertical[target] = anchor
                else:
                    active_vertical.pop(target, None)

            column_index += span

        for column in range(total_columns):
            if current_row[column] is None and column in active_vertical:
                anchor = active_vertical[column]
                current_row[column] = anchor
                if id(anchor) not in extended_in_row:
                    anchor.rowspan += 1
                    extended_in_row.add(id(anchor))

        for column in range(total_columns):
            if current_row[column] is not None:
                continue
            anchor = TableCell(
                cell_id=next_cell_id(counters),
                start_row=row_index,
                start_col=column,
                rowspan=1,
                colspan=1,
                text="",
                color=None,
                element=None,
                nested_table_ids=[],
            )
            cells.append(anchor)
            current_row[column] = anchor
            active_vertical.pop(column, None)

        occupancy.append([cell for cell in current_row if cell is not None])

    table_grid = TableGrid(
        row_count=len(occupancy),
        column_count=total_columns,
        occupancy=occupancy,
        cells=cells,
    )
    return trim_blank_table_grid(table_grid)


def spanned_inner_width(widths: list[int], start_col: int, colspan: int) -> int:
    return sum(widths[start_col:start_col + colspan]) + (3 * (colspan - 1))


def compute_column_widths(table_grid: TableGrid) -> list[int]:
    show_cell_ids = table_has_visible_fill(table_grid)
    widths = [1] * table_grid.column_count
    for cell in table_grid.cells:
        text = build_cell_display_text(cell, show_cell_ids)
        if not text:
            continue
        current_width = spanned_inner_width(widths, cell.start_col, cell.colspan)
        needed_width = max(len(line) for line in split_display_lines(text))
        if needed_width <= current_width:
            continue
        deficit = needed_width - current_width
        base_increase = deficit // cell.colspan
        remaining = deficit % cell.colspan
        for offset in range(cell.colspan):
            widths[cell.start_col + offset] += base_increase
            if offset < remaining:
                widths[cell.start_col + offset] += 1
    return widths


def render_horizontal_boundary(table_grid: TableGrid, widths: list[int], boundary_row: int) -> str:
    column_count = table_grid.column_count
    above = table_grid.occupancy[boundary_row - 1] if boundary_row > 0 else [None] * column_count
    below = table_grid.occupancy[boundary_row] if boundary_row < table_grid.row_count else [None] * column_count
    edges = [above[column] is not below[column] for column in range(column_count)]

    parts: list[str] = ["+"]
    for column in range(column_count):
        parts.append(("-" if edges[column] else " ") * (widths[column] + 2))
        if column == column_count - 1:
            parts.append("+")
            continue

        vertical_boundary = (above[column] is not above[column + 1]) or (below[column] is not below[column + 1])
        if vertical_boundary or edges[column] != edges[column + 1]:
            parts.append("+")
        elif edges[column] and edges[column + 1]:
            parts.append("-")
        else:
            parts.append(" ")
    return "".join(parts)


def render_content_rows(table_grid: TableGrid, widths: list[int], row_index: int) -> list[str]:
    show_cell_ids = table_has_visible_fill(table_grid)
    row = table_grid.occupancy[row_index]
    segments: list[tuple[int, list[str]]] = []
    column = 0
    row_height = 1
    while column < table_grid.column_count:
        cell = row[column]
        span = 1
        while column + span < table_grid.column_count and row[column + span] is cell:
            span += 1

        inner_width = spanned_inner_width(widths, column, span)
        text = build_cell_display_text(cell, show_cell_ids) if cell.start_row == row_index and cell.start_col == column else ""
        lines = split_display_lines(text)
        row_height = max(row_height, len(lines))
        segments.append((inner_width, lines))
        column += span

    rendered_lines: list[str] = []
    for line_index in range(row_height):
        parts: list[str] = ["|"]
        for inner_width, lines in segments:
            line = lines[line_index] if line_index < len(lines) else ""
            parts.append(f" {line.ljust(inner_width)} ")
            parts.append("|")
        rendered_lines.append("".join(parts))

    return rendered_lines


def render_ascii_table(table_grid: TableGrid) -> str:
    if table_grid.row_count == 0 or table_grid.column_count == 0:
        return "(empty table)"

    widths = compute_column_widths(table_grid)
    lines = [render_horizontal_boundary(table_grid, widths, 0)]
    for row_index in range(table_grid.row_count):
        lines.extend(render_content_rows(table_grid, widths, row_index))
        lines.append(render_horizontal_boundary(table_grid, widths, row_index + 1))
    return "\n".join(lines)


def build_table_entry(
    table: ET.Element,
    theme_colors: dict[str, str],
    counters: EntryCounters,
    location_prefix: str,
    numbering_context: NumberingContext | None = None,
    asset_context: AssetExportContext | None = None,
    *,
    is_nested: bool = False,
    parent_table_id: str | None = None,
    parent_cell_id: str | None = None,
) -> tuple[TableEntry, TableGrid]:
    table_id = next_table_id(counters)
    table_grid = build_table_grid(table, theme_colors, counters, numbering_context, asset_context)
    entry = TableEntry(
        table_id=table_id,
        table_index=counters.table_index,
        row_count=table_grid.row_count,
        column_count=table_grid.column_count,
        location=f"{location_prefix} (table {counters.table_index})",
        rendered_table=render_ascii_table(table_grid),
        is_nested=is_nested,
        parent_table_id=parent_table_id,
        parent_cell_id=parent_cell_id,
    )
    return entry, table_grid


def collect_nested_table_grids(
    table: ET.Element,
    theme_colors: dict[str, str],
    counters: EntryCounters,
    location_prefix: str,
    table_grids: OrderedDict[str, TableGrid],
    numbering_context: NumberingContext | None = None,
    asset_context: AssetExportContext | None = None,
    *,
    is_nested: bool = False,
    parent_table_id: str | None = None,
    parent_cell_id: str | None = None,
) -> list[TableEntry]:
    table_entry, table_grid = build_table_entry(
        table,
        theme_colors,
        counters,
        location_prefix,
        numbering_context,
        asset_context,
        is_nested=is_nested,
        parent_table_id=parent_table_id,
        parent_cell_id=parent_cell_id,
    )
    table_grids[table_entry.table_id] = table_grid
    nested_entries: list[TableEntry] = [table_entry]
    ordered_cells = sorted(table_grid.cells, key=lambda cell: (cell.start_row, cell.start_col))
    for cell in ordered_cells:
        if cell.element is None:
            continue
        cell_location = f"{table_entry.location} > row {cell.start_row + 1}, col {cell.start_col + 1}"
        for child in iter_block_children(cell.element):
            if child.tag != qn("w:tbl"):
                continue
            nested_descendants = collect_nested_table_grids(
                child,
                theme_colors,
                counters,
                cell_location,
                table_grids,
                numbering_context,
                asset_context,
                is_nested=True,
                parent_table_id=table_entry.table_id,
                parent_cell_id=cell.cell_id,
            )
            cell.nested_table_ids.append(nested_descendants[0].table_id)
            cell.text = cell.text.replace("{NESTED_TABLE}", f"<{nested_descendants[0].table_id}>", 1)
            nested_entries.extend(nested_descendants)
    table_entry.rendered_table = render_ascii_table(table_grid)
    return nested_entries


def build_table_map(entries: Iterable[ParagraphEntry | TableEntry]) -> OrderedDict[str, dict[str, object]]:
    table_map: OrderedDict[str, dict[str, object]] = OrderedDict()
    for entry in entries:
        if not isinstance(entry, TableEntry):
            continue
        table_map[entry.table_id] = {
            "table_index": entry.table_index,
            "location": entry.location,
            "row_count": entry.row_count,
            "column_count": entry.column_count,
            "rendered_table": entry.rendered_table,
            "chunk_type": "nested_table" if entry.is_nested else "table",
            "is_nested": entry.is_nested,
            "parent_table_id": entry.parent_table_id,
            "parent_cell_id": entry.parent_cell_id,
            "chunk_file_name": None,
        }
    return table_map


def build_cell_map(entries: Iterable[ParagraphEntry | TableEntry], table_grids: OrderedDict[str, TableGrid]) -> OrderedDict[str, dict[str, object]]:
    cell_map: OrderedDict[str, dict[str, object]] = OrderedDict()
    for entry in entries:
        if not isinstance(entry, TableEntry):
            continue
        table_grid = table_grids.get(entry.table_id)
        if table_grid is None:
            continue
        show_cell_ids = table_has_visible_fill(table_grid)
        for cell in sorted(table_grid.cells, key=lambda value: (value.start_row, value.start_col, value.cell_id)):
            cell_map[cell.cell_id] = {
                "table_id": entry.table_id,
                "row": cell.start_row + 1,
                "col": cell.start_col + 1,
                "rowspan": cell.rowspan,
                "colspan": cell.colspan,
                "text": cell.text,
                "color": cell.color,
                "nested_table_ids": cell.nested_table_ids,
                "display_text": build_cell_display_text(cell, show_cell_ids),
            }
    return cell_map


def build_asset_map(asset_context: AssetExportContext | None) -> OrderedDict[str, dict[str, object]]:
    asset_map: OrderedDict[str, dict[str, object]] = OrderedDict()
    if asset_context is None:
        return asset_map
    for asset_id, asset in asset_context.assets_by_id.items():
        asset_map[asset_id] = {
            "kind": asset.kind,
            "original_name": asset.original_name,
            "stored_name": asset.stored_name,
            "relative_path": asset.relative_path,
            "source_path": asset.source_path,
            "content_type": asset.content_type,
            "relationship_type": asset.relationship_type,
        }
    return asset_map


def build_export_artifacts(docx_path: Path, asset_output_dir: Path, asset_relative_dir: str) -> ExportArtifacts:
    with zipfile.ZipFile(docx_path) as archive:
        document_root = read_archive_xml(archive, "word/document.xml")
        theme_colors = read_theme_colors(docx_path)
        numbering_context = NumberingContext(definitions=read_numbering_definitions(docx_path))
        body = document_root.find("./w:body", NS)
        counters = EntryCounters()
        asset_context = AssetExportContext(
            archive=archive,
            output_dir=asset_output_dir,
            relative_dir=asset_relative_dir,
            relationships=read_relationships(archive, "word/document.xml"),
            content_types=read_content_types(archive),
            counters=counters,
        )

        if body is None:
            return ExportArtifacts(
                chunks=[],
                table_map=OrderedDict(),
                cell_map=OrderedDict(),
                asset_map=OrderedDict(),
            )

        # `entries` drives the normal section-classification chunks.
        entries: list[ParagraphEntry | TableEntry] = []
        # `all_table_entries` collects every table, including nested ones.
        all_table_entries: list[TableEntry] = []
        table_grids: OrderedDict[str, TableGrid] = OrderedDict()

        for body_index, child in enumerate(list(body), start=1):
            if child.tag == qn("w:p"):
                text = paragraph_text(child, numbering_context, asset_context)
                if not text:
                    continue
                counters.paragraph_index += 1
                entries.append(
                    ParagraphEntry(
                        body_index=body_index,
                        paragraph_index=counters.paragraph_index,
                        text=text,
                    )
                )
            elif child.tag == qn("w:tbl"):
                table_entries = collect_nested_table_grids(
                    child,
                    theme_colors,
                    counters,
                    f"body block {body_index}",
                    table_grids,
                    numbering_context,
                    asset_context,
                )
                entries.append(table_entries[0])
                all_table_entries.extend(table_entries)

    chunks: list[ChunkEntry] = []
    pending_texts: list[str] = []
    child_tables_by_parent: dict[str, list[TableEntry]] = {}
    for table_entry in all_table_entries:
        if table_entry.parent_table_id is None:
            continue
        child_tables_by_parent.setdefault(table_entry.parent_table_id, []).append(table_entry)

    for child_entries in child_tables_by_parent.values():
        child_entries.sort(key=lambda entry: entry.table_index)

    def append_nested_table_chunks(parent_table_id: str, parent_file_stem: str) -> None:
        child_entries = child_tables_by_parent.get(parent_table_id, [])
        for child_index, child_entry in enumerate(child_entries, start=1):
            child_file_name = f"{parent_file_stem}_nested_{child_index:04d}.txt"
            chunks.append(
                ChunkEntry(
                    chunk_index=len(chunks),
                    chunk_type="nested_table",
                    content=child_entry.rendered_table.rstrip() + "\n",
                    table_id=child_entry.table_id,
                    file_name=child_file_name,
                )
            )
            append_nested_table_chunks(child_entry.table_id, child_file_name[:-4])

    def flush_pending_texts() -> None:
        nonlocal pending_texts, chunks
        if not pending_texts:
            return
        chunks.append(
            ChunkEntry(
                chunk_index=len(chunks),
                chunk_type="text",
                content="\n\n".join(text.rstrip() for text in pending_texts).rstrip() + "\n",
            )
        )
        pending_texts = []

    for entry in entries:
        if isinstance(entry, ParagraphEntry):
            pending_texts.append(entry.text)
        else:
            flush_pending_texts()
            table_file_name = f"{len(chunks):04d}_table.txt"
            chunks.append(
                ChunkEntry(
                    chunk_index=len(chunks),
                    chunk_type="table",
                    content=entry.rendered_table.rstrip() + "\n",
                    table_id=entry.table_id,
                    file_name=table_file_name,
                )
            )
            append_nested_table_chunks(entry.table_id, table_file_name[:-4])

    flush_pending_texts()

    # Build maps from paragraphs + all tables so that schema_table_map.json
    # and schema_cell_map.json remain fully populated.
    map_entries: list[ParagraphEntry | TableEntry] = [
        e for e in entries if isinstance(e, ParagraphEntry)
    ] + all_table_entries

    return ExportArtifacts(
        chunks=chunks,
        table_map=build_table_map(map_entries),
        cell_map=build_cell_map(map_entries, table_grids),
        asset_map=build_asset_map(asset_context),
    )


def write_chunks(output_dir: Path, chunks: list[ChunkEntry]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    for existing_path in output_dir.glob("*.txt"):
        existing_path.unlink()

    for chunk in chunks:
        filename = chunk.file_name or f"{chunk.chunk_index:04d}_{chunk.chunk_type}.txt"
        path = output_dir / filename
        path.write_text(chunk.content, encoding="utf-8")
        written_paths.append(path)

    return written_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a DOCX file through its XML package and emit multiple TXT chunks."
    )
    parser.add_argument("docx_path", nargs="?", type=Path, help="Path to the .docx file")
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Directory where chunk and schema files will be written",
    )
    parser.add_argument(
        "--document",
        dest="documents",
        action="append",
        help="Document directory name under documents/. Defaults to all documents.",
    )
    return parser.parse_args()


def export_document(document: DocumentPaths, reporter: ReportFn | None = None) -> dict[str, object]:
    docx_path = document.source_docx.expanduser().resolve()
    output_dir = document.chunks_dir.expanduser().resolve()
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX file was not found: {docx_path}")

    report = reporter or print

    output_dir.mkdir(parents=True, exist_ok=True)
    table_map_path, cell_map_path, asset_map_path, asset_dir_path = derive_artifact_paths(output_dir)

    artifacts = build_export_artifacts(docx_path, asset_dir_path, asset_dir_path.name)
    chunk_paths = write_chunks(output_dir, artifacts.chunks)
    for chunk, path in zip(artifacts.chunks, chunk_paths):
        if chunk.table_id is None:
            continue
        table_entry = artifacts.table_map.get(chunk.table_id)
        if table_entry is None:
            continue
        table_entry["chunk_file_name"] = path.name

    table_map_path.write_text(json.dumps(artifacts.table_map, ensure_ascii=False, indent=2), encoding="utf-8")
    cell_map_path.write_text(json.dumps(artifacts.cell_map, ensure_ascii=False, indent=2), encoding="utf-8")
    asset_map_path.write_text(json.dumps(artifacts.asset_map, ensure_ascii=False, indent=2), encoding="utf-8")

    report(f"Chunks written to directory: {output_dir}")
    report(f"Chunk count: {len(chunk_paths)}")
    for path in chunk_paths:
        report(f" - {path.name}")
    report(f"Table map written to: {table_map_path}")
    report(f"Cell map written to: {cell_map_path}")
    report(f"Asset map written to: {asset_map_path}")
    report(f"Asset files written to: {asset_dir_path}")

    return {
        "document": document,
        "chunk_paths": chunk_paths,
        "chunk_count": len(chunk_paths),
        "table_map_path": table_map_path,
        "cell_map_path": cell_map_path,
        "asset_map_path": asset_map_path,
        "asset_dir_path": asset_dir_path,
    }


def main() -> None:
    args = parse_args()
    if args.docx_path is not None:
        docx_path = args.docx_path.expanduser().resolve()
        if args.output_dir is None:
            output_dir = default_output_dir_for_docx(docx_path)
        else:
            output_dir = args.output_dir.expanduser().resolve()

        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX file was not found: {docx_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        table_map_path, cell_map_path, asset_map_path, asset_dir_path = derive_artifact_paths(output_dir)
        artifacts = build_export_artifacts(docx_path, asset_dir_path, asset_dir_path.name)
        chunk_paths = write_chunks(output_dir, artifacts.chunks)
        for chunk, path in zip(artifacts.chunks, chunk_paths):
            if chunk.table_id is None:
                continue
            table_entry = artifacts.table_map.get(chunk.table_id)
            if table_entry is None:
                continue
            table_entry["chunk_file_name"] = path.name
        table_map_path.write_text(json.dumps(artifacts.table_map, ensure_ascii=False, indent=2), encoding="utf-8")
        cell_map_path.write_text(json.dumps(artifacts.cell_map, ensure_ascii=False, indent=2), encoding="utf-8")
        asset_map_path.write_text(json.dumps(artifacts.asset_map, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Chunks written to directory: {output_dir}")
        print(f"Chunk count: {len(chunk_paths)}")
        for path in chunk_paths:
            print(f" - {path.name}")
        print(f"Table map written to: {table_map_path}")
        print(f"Cell map written to: {cell_map_path}")
        print(f"Asset map written to: {asset_map_path}")
        print(f"Asset files written to: {asset_dir_path}")
        return

    documents = list_document_paths(args.documents)
    if not documents:
        raise FileNotFoundError("No document directories with source .docx files were found under documents/")

    for document in documents:
        print(f"\n=== Document: {document.name} ===")
        export_document(document)


if __name__ == "__main__":
    main()
