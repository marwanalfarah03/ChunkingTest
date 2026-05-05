from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from openai import NotFoundError

from classification import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SECTION_INDEX,
    SECTIONS,
    initialize_client,
    project_relative_path,
    resolve_document_path,
    write_json,
)
from header_inspection import (
    DEFAULT_CELL_MAP_NAME,
    DEFAULT_CLASSIFICATION_OUTPUT_NAME,
    DEFAULT_OUTPUT_DIRNAME,
    DEFAULT_TABLE_MAP_NAME,
    build_chunk_records,
    build_inspection_input_content,
    build_section_spans,
    build_single_table_group_component,
    build_span_components,
    find_chunk_artifact_dir,
    load_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECTION_JSON_PROMPTS_DIR = PROJECT_ROOT / "prompts" / "section_json"
DEFAULT_SECTION_JSON_OUTPUT_NAME = "section_json_output.json"
DEFAULT_COLUMN_INSPECTION_OUTPUT_NAME = "column_header_inspection.json"

EXCLUDED_SECTIONS = {"SEC01", "SEC06"}
SECTIONS_TO_PROCESS = [s["id"] for s in SECTIONS if s["id"] not in EXCLUDED_SECTIONS]
SECTIONS_WITH_COLUMN_INSPECTION = {"SEC03", "SEC04", "SEC05", "SEC12", "SEC13", "SEC15"}

# SEC99 returns a JSON array; all other sections return a JSON object.
SEC99_ID = "SEC99"

VALID_SEC99_FORMATS = {"titled_item", "matrix", "empty", "other", "not_found"}


def build_sec99_span_components(
    span_records: tuple[Any, ...],
    cell_map: dict[str, Any],
) -> list[dict[str, Any]]:
    # SEC99 tables are unrelated to each other, so consecutive table chunks must
    # never be merged into one virtual table the way they are for other sections.
    # Each table chunk is rendered independently; text chunks pass through as-is.
    components: list[dict[str, Any]] = []
    for record in span_records:
        if record["chunk_type"] == "table" and record.get("table_id") and record.get("table_entry"):
            components.append(build_single_table_group_component(record, cell_map))
        else:
            components.append(
                {
                    "kind": "text",
                    "source_chunk_file_names": [record["txt_file_name"]],
                    "content": record["content"].rstrip(),
                }
            )
    return components


def load_section_system_prompt(section_id: str) -> str:
    prompt_path = SECTION_JSON_PROMPTS_DIR / f"{section_id}_system.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt not found for {section_id}: {prompt_path}"
        )
    return prompt_path.read_text(encoding="utf-8")


def find_column_inspection_result(
    inspection_results: list[dict[str, Any]],
    section_id: str,
    span_chunk_file_names: list[str],
) -> dict[str, Any] | None:
    span_files = set(span_chunk_file_names)
    for result in inspection_results:
        if result.get("inspected_section_id") != section_id:
            continue
        if result.get("status") == "skipped":
            continue
        source_files = set(result.get("source_chunk_file_names") or [])
        if source_files & span_files:
            return result
    return None


def build_column_inspection_context(
    inspection_result: dict[str, Any] | None,
    section_id: str,
) -> str:
    if not inspection_result or section_id not in SECTIONS_WITH_COLUMN_INSPECTION:
        return ""
    resolution = inspection_result.get("resolution") or {}
    context_payload = {
        "valid_column_order": resolution.get("valid_column_order"),
        "actual_header_row_exists": resolution.get("actual_header_row_exists"),
        "actual_header_cell_ids": resolution.get("actual_header_cell_ids"),
        "header_state": resolution.get("header_state"),
    }
    return (
        "Column header inspection result (use these column labels exactly in your JSON output):\n"
        + json.dumps(context_payload, ensure_ascii=False, indent=2)
    )


def build_user_prompt(
    document_name: str,
    section_id: str,
    span_chunk_file_names: list[str],
    content: str,
    column_inspection_context: str,
    co_sections: set[str] | None = None,
) -> str:
    section_meta = SECTION_INDEX[section_id]
    source_files_str = ", ".join(span_chunk_file_names)

    parts: list[str] = [
        "Extract the target section from this SOP document content and return a structured JSON.",
        "",
        f"Document: {document_name}",
        f"Target section: {section_id} — {section_meta['arabic']} ({section_meta['english']})",
        f"Source files: {source_files_str}",
    ]

    if section_id == "SEC02" and co_sections and "SEC01" in co_sections:
        parts.append("")
        parts.append(
            "Important: This chunk is classified as both SEC01 and SEC02. "
            "It may contain SEC01 content — the policies and procedures department name "
            "and/or a bank header image (asset marker) — in addition to the SEC02 document "
            "header table. Ignore all SEC01 elements entirely: skip any department name text "
            "and any image or asset markers that belong to the department banner. Extract only "
            "the SEC02 document header table rows."
        )

    if column_inspection_context:
        parts.append("")
        parts.append(column_inspection_context)

    parts.append("")
    parts.append("Content:")
    parts.append(content)

    return "\n".join(parts)


def _strip_markdown_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    inner = [line for line in lines if not line.startswith("```")]
    return "\n".join(inner).strip()


def parse_section_json_response(
    response_text: str,
    section_id: str,
) -> dict[str, Any] | list[Any]:
    cleaned = _strip_markdown_fences(response_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response was not valid JSON: {exc}") from exc

    if section_id == SEC99_ID:
        if not isinstance(payload, list):
            raise ValueError("SEC99 response must be a JSON array.")
        if not payload:
            raise ValueError("SEC99 response array must not be empty.")
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(
                    "Each element in the SEC99 response array must be a JSON object."
                )
            fmt = item.get("format")
            if fmt not in VALID_SEC99_FORMATS:
                raise ValueError(
                    f"SEC99 element has unknown format: {fmt!r}. "
                    f"Allowed: {sorted(VALID_SEC99_FORMATS)}"
                )
        return payload

    if not isinstance(payload, dict):
        raise ValueError("Response must be a JSON object.")

    status = payload.get("status")
    if status == "not_found":
        return payload

    resp_section_id = payload.get("section_id")
    if resp_section_id != section_id:
        raise ValueError(
            f"Response section_id {resp_section_id!r} does not match expected {section_id!r}."
        )

    payload = normalize_section_json_payload(payload, section_id)
    return payload


def normalize_section_json_payload(
    payload: dict[str, Any],
    section_id: str,
) -> dict[str, Any]:
    if section_id != "SEC18":
        return payload
    if str(payload.get("format") or "") != "sections":
        return payload

    section_meta = SECTION_INDEX.get(section_id) or {}
    arabic = str(section_meta.get("arabic") or "").strip()
    english = str(section_meta.get("english") or "").strip()
    if arabic and english:
        normalized_heading = f"{arabic} ({english})"
    else:
        normalized_heading = arabic or english or str(payload.get("section_heading") or "").strip()

    normalized = dict(payload)
    normalized["section_heading"] = normalized_heading
    return normalized


def request_section_json(
    client: Any,
    model: str,
    section_id: str,
    system_prompt: str,
    user_prompt: str,
    max_llm_retries: int,
    log_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any] | list[Any] | None, int, list[dict[str, str]]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    invalid_attempts: list[dict[str, str]] = []
    attempt_count = 0

    while True:
        attempt_count += 1
        sent_messages = list(messages)
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=messages,
            )
        except NotFoundError as exc:
            raise RuntimeError(
                "Received 404 Not Found from the chat completions endpoint during section JSON extraction. "
                "Check the base URL and model identifier."
            ) from exc

        response_text = (response.choices[0].message.content or "").strip()

        try:
            parsed = parse_section_json_response(response_text, section_id)
            if log_callback is not None:
                log_callback({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "stage": "extract",
                    "subject": section_id,
                    "model": model,
                    "attempt": attempt_count,
                    "messages": sent_messages,
                    "response": response_text,
                    "success": True,
                    "error": None,
                })
            return parsed, attempt_count, invalid_attempts
        except ValueError as exc:
            error_str = str(exc)
            invalid_attempts.append(
                {
                    "attempt": str(attempt_count),
                    "error": error_str,
                    "response": response_text,
                }
            )
            if log_callback is not None:
                log_callback({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "stage": "extract",
                    "subject": section_id,
                    "model": model,
                    "attempt": attempt_count,
                    "messages": sent_messages,
                    "response": response_text,
                    "success": False,
                    "error": error_str,
                })
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
                            "Return only valid JSON exactly as described in the system prompt."
                        ),
                    },
                ]
            )


def extract_document_sections(
    document_path: Path,
    classification_output_path: Path,
    column_inspection_output_path: Path | None = None,
    output_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "",
    max_llm_retries: int = 6,
    quiet: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    log_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    artifact_dir = find_chunk_artifact_dir(document_path)
    if artifact_dir is None:
        raise FileNotFoundError(
            f"Could not find chunk artifacts under {document_path} or "
            f"{document_path / DEFAULT_OUTPUT_DIRNAME}"
        )

    classification_payload = load_json(classification_output_path)
    table_map = load_json(artifact_dir / DEFAULT_TABLE_MAP_NAME)
    cell_map = load_json(artifact_dir / DEFAULT_CELL_MAP_NAME)

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

    inspection_results: list[dict[str, Any]] = []
    if column_inspection_output_path and column_inspection_output_path.exists():
        try:
            inspection_payload = load_json(column_inspection_output_path)
            inspection_results = inspection_payload.get("results") or []
        except (FileNotFoundError, ValueError):
            pass

    # Build spans for every processed section. A span is a contiguous run of
    # chunk records all classified with the same section_id. Multiple runs of
    # the same section_id (non-adjacent) produce separate spans.
    all_spans: list[dict[str, Any]] = []
    section_order_index = {sid: i for i, sid in enumerate(SECTIONS_TO_PROCESS)}
    for section_id in SECTIONS_TO_PROCESS:
        all_spans.extend(build_section_spans(chunk_records, section_id))

    def _span_sort_key(span: dict[str, Any]) -> tuple[int, int, int]:
        records = span["records"]
        first_chunk_index = records[0]["index"] if records else 999_999
        # Secondary key: position of this section within the first chunk's predicted
        # sections list (document order), falling back to canonical order.
        chunk_sections: list[str] = records[0].get("predicted_sections", []) if records else []
        try:
            section_pos = chunk_sections.index(span["section_id"])
        except ValueError:
            section_pos = section_order_index.get(span["section_id"], 999)
        return first_chunk_index, section_pos, span["span_index"]

    # Sort spans by document position (first chunk's original index) so that
    # the results list preserves the order chunks appear in the document.
    all_spans.sort(key=_span_sort_key)

    # Merge non-adjacent spans of the same section into one.  A Word table that
    # spans multiple pages (with header repetition) often produces separate .txt
    # chunks with a different-section chunk in between, which breaks the
    # contiguous run and creates spurious extra spans.  SEC99 is intentionally
    # excluded because each SEC99 table is an independent, unrelated item.
    merged_spans: list[dict[str, Any]] = []
    seen_section_spans: dict[str, dict[str, Any]] = {}
    for span in all_spans:
        sid = span["section_id"]
        if sid == SEC99_ID:
            merged_spans.append(span)
        elif sid in seen_section_spans:
            seen_section_spans[sid]["records"] = (
                seen_section_spans[sid]["records"] + span["records"]
            )
        else:
            seen_section_spans[sid] = span
            merged_spans.append(span)
    all_spans = merged_spans

    client, resolved_base_url = initialize_client(
        base_url=base_url, api_key=api_key, model=model
    )
    if not quiet and resolved_base_url.rstrip("/") != base_url.rstrip("/"):
        print(
            f"Resolved base URL from '{base_url}' to '{resolved_base_url}' after endpoint probing.",
            file=sys.stderr,
            flush=True,
        )

    # Track processed (section_id, frozenset-of-chunk-names) pairs so that a
    # span is never sent to the LLM more than once even if multiple passes over
    # the span list would include it.
    processed_span_keys: set[tuple[str, frozenset[str]]] = set()

    results: list[dict[str, Any]] = []
    total_spans = len(all_spans)

    if progress_callback is not None:
        progress_callback(
            {
                "phase": "section_json",
                "current": 0,
                "total": total_spans,
                "message": "Preparing final document sections",
            }
        )

    for index, span in enumerate(all_spans, start=1):
        section_id = span["section_id"]
        span_records = span["records"]
        span_chunk_names = [r["txt_file_name"] for r in span_records]
        span_key = (section_id, frozenset(span_chunk_names))

        if span_key in processed_span_keys:
            continue
        processed_span_keys.add(span_key)

        # Build rendered content. For SEC99, tables are unrelated so each is
        # rendered individually. For all other sections, consecutive table chunks
        # are merged as if they were one table in the original document.
        if section_id == SEC99_ID:
            components = build_sec99_span_components(span_records, cell_map)
        else:
            components = build_span_components(span_records, cell_map)
        content = build_inspection_input_content(components)

        # Include column inspection results for tabular sections.
        column_inspection_result: dict[str, Any] | None = None
        if section_id in SECTIONS_WITH_COLUMN_INSPECTION:
            column_inspection_result = find_column_inspection_result(
                inspection_results, section_id, span_chunk_names
            )
        column_inspection_context = build_column_inspection_context(
            column_inspection_result, section_id
        )

        co_sections: set[str] = {
            sid
            for record in span_records
            for sid in record.get("predicted_sections", [])
            if sid != section_id
        }

        system_prompt = load_section_system_prompt(section_id)
        user_prompt = build_user_prompt(
            document_name=document_path.name,
            section_id=section_id,
            span_chunk_file_names=span_chunk_names,
            content=content,
            column_inspection_context=column_inspection_context,
            co_sections=co_sections,
        )

        section_meta = SECTION_INDEX[section_id]

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "section_json",
                    "current": index - 1,
                    "total": total_spans,
                    "section_id": section_id,
                    "message": f"Creating section {index} of {total_spans}",
                }
            )

        parsed_json, attempt_count, invalid_attempts = request_section_json(
            client=client,
            model=model,
            section_id=section_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_llm_retries=max_llm_retries,
            log_callback=log_callback,
        )

        status = "resolved" if parsed_json is not None else "failed"

        results.append(
            {
                "section_id": section_id,
                "section_arabic": section_meta["arabic"],
                "section_english": section_meta["english"],
                "span_index": span["span_index"],
                "source_chunk_file_names": span_chunk_names,
                "source_relative_paths": [r["relative_path"] for r in span_records],
                "column_inspection_used": column_inspection_result is not None,
                "status": status,
                "llm_attempt_count": attempt_count,
                "invalid_attempts": invalid_attempts,
                "section_json": parsed_json,
            }
        )

        if not quiet:
            files_label = ", ".join(span_chunk_names)
            print(
                f"[{index}/{total_spans}] {section_id} span {span['span_index']} | "
                f"{files_label} | status={status} | attempts={attempt_count}",
                file=sys.stderr,
                flush=True,
            )

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "section_json",
                    "current": index,
                    "total": total_spans,
                    "section_id": section_id,
                    "message": f"Created section {index} of {total_spans}",
                }
            )

    payload = {
        "summary": {
            "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "document_name": document_path.name,
            "document_path": project_relative_path(document_path),
            "classification_output_path": project_relative_path(classification_output_path),
            "column_inspection_output_path": (
                project_relative_path(column_inspection_output_path)
                if column_inspection_output_path
                else None
            ),
            "model": model,
            "resolved_base_url": resolved_base_url,
            "total_spans_processed": len(results),
            "resolved_count": sum(1 for r in results if r["status"] == "resolved"),
            "failed_count": sum(1 for r in results if r["status"] == "failed"),
        },
        "results": results,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, payload)
        if not quiet:
            print(
                f"Wrote section JSON output to {output_path}",
                file=sys.stderr,
                flush=True,
            )

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured JSON for each SOP section from classified and inspected chunks."
    )
    parser.add_argument(
        "document",
        help=(
            "Document directory name under documents/, an explicit extracted document directory path, "
            "or a source .docx path whose extracted directory shares the same stem under documents/."
        ),
    )
    parser.add_argument(
        "--documents-root",
        default="documents",
        help="Directory holding extracted document folders.",
    )
    parser.add_argument(
        "--classification-output",
        default=None,
        help="Path to the classification JSON. Defaults to <document>/classification_output.json.",
    )
    parser.add_argument(
        "--column-inspection-output",
        default=None,
        help=(
            "Path to the column header inspection JSON. "
            "Defaults to <document>/column_header_inspection.json. "
            "If the file does not exist, column context is omitted."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the output JSON. Defaults to <document>/section_json_output.json.",
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
        help="Maximum LLM retries per span before marking it as failed.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging on stderr.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    documents_root = Path(args.documents_root)
    if not documents_root.is_absolute():
        documents_root = PROJECT_ROOT / documents_root

    document_path = resolve_document_path(args.document, documents_root)

    classification_output_path = (
        Path(args.classification_output)
        if args.classification_output
        else document_path / DEFAULT_CLASSIFICATION_OUTPUT_NAME
    )
    if not classification_output_path.is_absolute():
        classification_output_path = PROJECT_ROOT / classification_output_path

    column_inspection_output_path = (
        Path(args.column_inspection_output)
        if args.column_inspection_output
        else document_path / DEFAULT_COLUMN_INSPECTION_OUTPUT_NAME
    )
    if not column_inspection_output_path.is_absolute():
        column_inspection_output_path = PROJECT_ROOT / column_inspection_output_path

    output_path = (
        Path(args.output)
        if args.output
        else document_path / DEFAULT_SECTION_JSON_OUTPUT_NAME
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    extract_document_sections(
        document_path=document_path,
        classification_output_path=classification_output_path,
        column_inspection_output_path=column_inspection_output_path,
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
