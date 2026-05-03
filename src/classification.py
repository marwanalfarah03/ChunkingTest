from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from openai import NotFoundError, OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "classification" / "chunk_classification_system.txt"


SECTIONS = [
    {"id": "SEC01", "arabic": "ترويسة دائرة السياسات والاجراءات", "english": "Department Header / Banner"},
    {"id": "SEC02", "arabic": "ترويسة الوثيقة", "english": "Document Header"},
    {"id": "SEC03", "arabic": "ضبط الوثيقة", "english": "Document Control"},
    {"id": "SEC04", "arabic": "إقرار الوثيقة", "english": "Document Approval"},
    {"id": "SEC05", "arabic": "ضبط الإصدار", "english": "Version Control"},
    {"id": "SEC06", "arabic": "جدول / قائمة المحتويات", "english": "Table of Contents"},
    {"id": "SEC07", "arabic": "الهدف من الوثيقة", "english": "Purpose"},
    {"id": "SEC08", "arabic": "مجال التطبيق", "english": "Scope"},
    {"id": "SEC09", "arabic": "المسؤوليات", "english": "Responsibilities"},
    {"id": "SEC10", "arabic": "المراجع الرئيسية", "english": "Key References"},
    {"id": "SEC11", "arabic": "التعليمات العامة", "english": "General Instructions"},
    {"id": "SEC12", "arabic": "الإجراء", "english": "Procedures / Workflow Steps"},
    {"id": "SEC13", "arabic": "الإجراءات الرقابية", "english": "Control Procedures"},
    {"id": "SEC14", "arabic": "التقارير والكشوفات", "english": "Reports & Statements"},
    {"id": "SEC15", "arabic": "الملفات ومدد الحفظ", "english": "Files & Retention Periods"},
    {"id": "SEC16", "arabic": "الملاحق", "english": "Appendices"},
    {"id": "SEC17", "arabic": "النماذج المستخدمة", "english": "Forms Used"},
    {"id": "SEC18", "arabic": "التعريفات والمختصرات والمصطلحات", "english": "Definitions / Terms / Abbreviations"},
    {"id": "SEC19", "arabic": "النماذج", "english": "Forms / Templates"},
    {"id": "SEC99", "arabic": "أخرى", "english": "Other / Unclassified"},
]

SECTION_INDEX = {section["id"]: section for section in SECTIONS}
SECTION_ORDER = {section["id"]: index for index, section in enumerate(SECTIONS)}
VALID_SECTION_IDS = set(SECTION_INDEX)
SECTION_CATALOG = "\n".join(
    f'- {section["id"]}: {section["arabic"]} ({section["english"]})' for section in SECTIONS
)
DEFAULT_BASE_URL = "http://localhost:8020"
DEFAULT_MODEL = "/data/models/gpt-oss-120b"
PREVIOUS_CONTEXT_WINDOW = 8


@dataclass(frozen=True)
class ChunkTarget:
    document_name: str
    txt_file_name: str
    relative_path: str
    file_path: Path
    raw_text: str


def section_sort_key(section_id: str) -> int:
    return SECTION_ORDER.get(section_id, 999)


def project_relative_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved_path)


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace("{catalog}", SECTION_CATALOG)


def build_base_url_candidates(base_url: str) -> list[str]:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("Base URL cannot be empty.")

    candidates = [normalized]
    if normalized.endswith("/v1"):
        without_v1 = normalized[: -len("/v1")].rstrip("/")
        if without_v1 and without_v1 not in candidates:
            candidates.append(without_v1)
    else:
        with_v1 = f"{normalized}/v1"
        if with_v1 not in candidates:
            candidates.append(with_v1)

    return candidates


def preflight_chat_completion(client: OpenAI, model: str) -> None:
    client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=1,
        messages=[{"role": "user", "content": "ping"}],
    )


def initialize_client(base_url: str, api_key: str, model: str) -> tuple[OpenAI, str]:
    candidates = build_base_url_candidates(base_url)
    not_found_candidates: list[str] = []

    for candidate in candidates:
        client = OpenAI(base_url=candidate, api_key=api_key)
        try:
            preflight_chat_completion(client, model)
            return client, candidate
        except NotFoundError:
            not_found_candidates.append(candidate)
            continue
        except Exception as exc:
            raise RuntimeError(
                f"API preflight failed for base URL '{candidate}'. "
                "Verify the server is reachable, the API key is valid, and the model exists."
            ) from exc

    if len(not_found_candidates) == len(candidates):
        checked_urls = ", ".join(not_found_candidates)
        raise RuntimeError(
            "API returned 404 Not Found for all base URL candidates "
            f"({checked_urls}). Try an OpenAI-compatible API root that serves chat completions, "
            "typically ending with '/v1'."
        )

    raise RuntimeError("Failed to initialize API client.")


def format_previous_document_context(previous_predictions: list[tuple[str, list[str]]]) -> str:
    if not previous_predictions:
        return "None."

    seen_sections: list[str] = []
    seen_lookup: set[str] = set()
    for _, sections in previous_predictions:
        for section_id in sections:
            if section_id not in seen_lookup:
                seen_sections.append(section_id)
                seen_lookup.add(section_id)

    recent_predictions = previous_predictions[-PREVIOUS_CONTEXT_WINDOW:]
    lines = [
        "Use this only as weak document context. It may help identify continuations, OCR-damaged headings, or that a canonical section already appeared earlier. Do not copy it blindly if the current chunk's own evidence disagrees.",
        f"Section IDs already seen earlier in this document: {', '.join(seen_sections)}",
        "Most recent classified chunks in this document:",
    ]
    for chunk_file_name, sections in recent_predictions:
        lines.append(f"- {chunk_file_name}: {', '.join(sections)}")

    return "\n".join(lines)


def build_user_prompt(
    target: ChunkTarget,
    previous_predictions: list[tuple[str, list[str]]] | None = None,
) -> str:
    previous_context = format_previous_document_context(previous_predictions or [])
    return (
        "Classify this extracted SOP chunk.\n"
        "Return JSON with this exact shape:\n"
        '{"sections": ["SEC##", "SEC##", ...]}\n\n'
        f"Document: {target.document_name}\n"
        f"Chunk file: {target.txt_file_name}\n"
        f"Relative path: {target.relative_path}\n\n"
        "Previous classifications from earlier chunks in the same document:\n"
        f"{previous_context}\n\n"
        "Raw snippet:\n"
        f"{target.raw_text}\n"
    )


def parse_prediction_payload(response_text: str) -> list[str]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Response JSON must be an object.")

    extra_keys = set(payload) - {"sections"}
    if extra_keys:
        raise ValueError(f"Unexpected keys in response JSON: {sorted(extra_keys)}")

    sections = payload.get("sections")
    if not isinstance(sections, list):
        raise ValueError("The 'sections' field must be a list.")

    normalized_sections: list[str] = []
    seen_sections: set[str] = set()
    for value in sections:
        if not isinstance(value, str):
            raise ValueError("All values in 'sections' must be strings.")
        if value not in VALID_SECTION_IDS:
            raise ValueError(f"Unknown section id returned: {value}")
        if value not in seen_sections:
            normalized_sections.append(value)
            seen_sections.add(value)

    if not normalized_sections:
        raise ValueError("The 'sections' list must not be empty; use SEC99 when nothing fits.")

    normalized_sections.sort(key=section_sort_key)
    return normalized_sections


def request_prediction(
    client: OpenAI,
    system_prompt: str,
    target: ChunkTarget,
    model: str,
    max_json_retries: int,
    previous_predictions: list[tuple[str, list[str]]] | None = None,
    log_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[str], int, list[dict[str, str]]]:
    user_prompt = build_user_prompt(target, previous_predictions=previous_predictions)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    invalid_attempts: list[dict[str, str]] = []
    invalid_count = 0

    while True:
        sent_messages = list(messages)
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=messages,
            )
        except NotFoundError as exc:
            raise RuntimeError(
                "Received 404 Not Found from the chat completions endpoint. "
                "Check that your API base URL points to an OpenAI-compatible route (often ending with '/v1') "
                "and that the selected model is available."
            ) from exc
        response_text = (response.choices[0].message.content or "").strip()

        try:
            predicted_sections = parse_prediction_payload(response_text)
            if log_callback is not None:
                log_callback({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "stage": "classify",
                    "subject": target.txt_file_name,
                    "model": model,
                    "attempt": invalid_count + 1,
                    "messages": sent_messages,
                    "response": response_text,
                    "success": True,
                    "error": None,
                })
            return predicted_sections, invalid_count, invalid_attempts
        except ValueError as exc:
            invalid_count += 1
            error_str = str(exc)
            invalid_attempts.append(
                {
                    "attempt": str(invalid_count),
                    "error": error_str,
                    "response": response_text,
                }
            )
            if log_callback is not None:
                log_callback({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "stage": "classify",
                    "subject": target.txt_file_name,
                    "model": model,
                    "attempt": invalid_count,
                    "messages": sent_messages,
                    "response": response_text,
                    "success": False,
                    "error": error_str,
                })

            if max_json_retries > 0 and invalid_count >= max_json_retries:
                raise RuntimeError(
                    f"Model failed to return valid JSON after {invalid_count} retries for {target.relative_path}."
                ) from exc

            messages.extend(
                [
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid. "
                            f"Validation error: {exc}. "
                            "Return only valid JSON with this exact shape: "
                            '{"sections": ["SEC##", "SEC##", ...]}.'
                        ),
                    },
                ]
            )


def preview_text(raw_text: str, limit: int = 220) -> str:
    compact = " ".join(raw_text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_document_path(document: str, documents_root: Path) -> Path:
    requested_path = Path(document)
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    if requested_path.is_absolute():
        add_candidate(requested_path)
    else:
        add_candidate(PROJECT_ROOT / requested_path)
        add_candidate(documents_root / requested_path)

    # Allow callers to pass a source .docx path and resolve it to the extracted
    # document directory name under documents/.
    if requested_path.suffix.lower() == ".docx":
        add_candidate(documents_root / requested_path.stem)

    for candidate in list(candidates):
        if candidate.suffix.lower() != ".docx":
            continue
        add_candidate(candidate.with_suffix(""))
        add_candidate(documents_root / candidate.stem)

    checked_candidates: list[str] = []
    for candidate in candidates:
        checked_candidates.append(str(candidate))
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    checked_display = "\n".join(checked_candidates)
    raise FileNotFoundError(
        f"Document directory not found for '{document}'. Checked:\n{checked_display}"
    )


def load_document_targets(document_path: Path) -> list[ChunkTarget]:
    chunk_paths = sorted(
        path
        for path in document_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt" and "_nested_" not in path.name
    )
    if not chunk_paths:
        raise FileNotFoundError(f"No non-nested .txt chunks were found in {document_path}")

    return [
        ChunkTarget(
            document_name=document_path.name,
            txt_file_name=chunk_path.name,
            relative_path=project_relative_path(chunk_path),
            file_path=chunk_path,
            raw_text=chunk_path.read_text(encoding="utf-8"),
        )
        for chunk_path in chunk_paths
    ]


def print_running_status(
    target: ChunkTarget,
    predicted_sections: list[str],
    json_retry_count: int,
    index: int,
    total_targets: int,
) -> None:
    print(
        f"[{index}/{total_targets}] {target.txt_file_name} | predicted={','.join(predicted_sections)} | "
        f"json_retries={json_retry_count}",
        file=sys.stderr,
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify one extracted SOP document without source-truth evaluation."
    )
    parser.add_argument(
        "document",
        help=(
            "Document directory name under documents/, an explicit extracted document directory path, "
            "or a source .docx path whose extracted directory shares the same stem under documents/."
        ),
    )
    parser.add_argument("--documents-root", default="documents", help="Directory holding extracted document folders.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="API base URL (for example: http://localhost:8020/v1).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key. Defaults to OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--max-json-retries",
        type=int,
        default=0,
        help="Maximum number of invalid-JSON retries per chunk. Use 0 for unlimited retries.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the output JSON. Defaults to stdout.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-chunk progress logging on stderr.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    documents_root = Path(args.documents_root)
    if not documents_root.is_absolute():
        documents_root = PROJECT_ROOT / documents_root

    document_path = resolve_document_path(args.document, documents_root)
    targets = load_document_targets(document_path)

    client, resolved_base_url = initialize_client(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )
    if resolved_base_url.rstrip("/") != args.base_url.rstrip("/"):
        print(
            f"Resolved base URL from '{args.base_url}' to '{resolved_base_url}' after endpoint probing.",
            file=sys.stderr,
            flush=True,
        )

    system_prompt = load_system_prompt()
    results: list[dict[str, object]] = []
    invalid_json_retry_count = 0
    document_prediction_context: list[tuple[str, list[str]]] = []

    for index, target in enumerate(targets, start=1):
        if not target.raw_text.strip():
            results.append(
                {
                    "document_name": target.document_name,
                    "txt_file_name": target.txt_file_name,
                    "relative_path": target.relative_path,
                    "predicted_sections": [],
                    "json_retry_count": 0,
                    "invalid_attempts": [],
                    "preview": "",
                    "skipped": "empty",
                }
            )
            if not args.quiet:
                print(
                    f"[{index}/{len(targets)}] {target.txt_file_name} | skipped (empty chunk)",
                    file=sys.stderr,
                    flush=True,
                )
            continue

        predicted_sections, json_retry_count, invalid_attempts = request_prediction(
            client=client,
            system_prompt=system_prompt,
            target=target,
            model=args.model,
            max_json_retries=args.max_json_retries,
            previous_predictions=document_prediction_context,
        )
        invalid_json_retry_count += json_retry_count
        document_prediction_context.append((target.txt_file_name, predicted_sections))

        results.append(
            {
                "document_name": target.document_name,
                "txt_file_name": target.txt_file_name,
                "relative_path": target.relative_path,
                "predicted_sections": predicted_sections,
                "json_retry_count": json_retry_count,
                "invalid_attempts": invalid_attempts,
                "preview": preview_text(target.raw_text),
            }
        )

        if not args.quiet:
            print_running_status(
                target=target,
                predicted_sections=predicted_sections,
                json_retry_count=json_retry_count,
                index=index,
                total_targets=len(targets),
            )

    payload = {
        "summary": {
            "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "document_name": document_path.name,
            "document_path": project_relative_path(document_path),
            "prompt_path": project_relative_path(PROMPT_PATH),
            "model": args.model,
            "resolved_base_url": resolved_base_url,
            "total_chunks": len(results),
            "invalid_json_retry_count": invalid_json_retry_count,
        },
        "results": results,
    }

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, payload)
        print(f"Wrote classification output to {output_path}", file=sys.stderr, flush=True)
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
