from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from classification import (
    SECTIONS,
    VALID_SECTION_IDS,
    project_relative_path,
    section_sort_key,
    write_json,
)


PAGE_SIZE = 25

_SECTION_CATALOG = "\n".join(
    f"    {s['id']:6}  {s['english']:<32}  {s['arabic']}"
    for s in SECTIONS
)

_HELP_TEXT = """
  Commands:
    <number>          edit that row's section assignment
    n / p             next / previous page
    p<n>              jump to page n  (e.g. p3)
    ENTER / ok        confirm all and continue
    q                 quit and discard any changes
    sections          list all valid section IDs with descriptions
    ?                 show this help
""".rstrip()


# ---------------------------------------------------------------------------
# Safe print: reconfigure stdout to UTF-8 on Windows if needed so Arabic
# characters in previews and catalog entries don't crash the process.
# ---------------------------------------------------------------------------

def _ensure_utf8_stdout() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _terminal_width() -> int:
    try:
        return min(os.get_terminal_size().columns, 220)
    except OSError:
        return 120


def _trunc(text: str, width: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    return flat[: width - 1] + ">"  # ASCII ellipsis substitute


def _parse_section_ids(raw: str) -> list[str] | None:
    parts = [p.strip().upper() for p in raw.replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None
    invalid = sorted({p for p in parts if p not in VALID_SECTION_IDS})
    if invalid:
        print(f"  ! Unknown section ID(s): {', '.join(invalid)}")
        print(f"  Valid IDs: {', '.join(sorted(VALID_SECTION_IDS, key=section_sort_key))}")
        return None
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return sorted(unique, key=section_sort_key)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _print_table(
    results: list[dict[str, Any]],
    modified_indices: set[int],
    page: int,
    page_size: int,
    document_name: str,
) -> None:
    width = _terminal_width()
    total = len(results)
    if total == 0:
        print("  (no classification results to review)")
        return

    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    row_start = page * page_size
    row_end = min(row_start + page_size, total)

    # Fixed column widths; preview fills the remainder.
    c_num = 4
    c_file = 24
    c_sec = 28
    # Each " | " separator = 3 chars; 3 separators + leading space = 10
    c_prev = max(20, width - c_num - c_file - c_sec - 10)

    sep = "-" * width

    def _row(num: str, fname: str, secs: str, prev: str, marker: str = " ") -> str:
        return (
            f"{marker}{num.rjust(c_num)} | "
            f"{fname:<{c_file}.{c_file}} | "
            f"{secs:<{c_sec}.{c_sec}} | "
            f"{_trunc(prev, c_prev)}"
        )

    print()
    print(f"  Classification Review -- {document_name}  "
          f"[page {page + 1}/{total_pages}, {total} chunks]")
    print(sep)
    print(_row("#", "File", "Sections", "Preview"))
    print(sep)

    for i in range(row_start, row_end):
        r = results[i]
        secs_str = ", ".join(r["predicted_sections"])
        marker = "*" if i in modified_indices else " "
        print(_row(
            str(i + 1),
            r["txt_file_name"],
            secs_str,
            r.get("preview") or "",
            marker=marker,
        ))

    print(sep)
    footer_parts: list[str] = []
    if modified_indices:
        footer_parts.append(f"* = modified ({len(modified_indices)} row(s))")
    if total_pages > 1:
        footer_parts.append("n=next  p=prev  p<n>=jump")
    footer_parts.append("? = help")
    print("  " + "   |   ".join(footer_parts))


# ---------------------------------------------------------------------------
# Row editor
# ---------------------------------------------------------------------------

def _edit_row(results: list[dict[str, Any]], one_idx: int) -> bool:
    """Edit row one_idx (1-based). Returns True if sections were changed."""
    i = one_idx - 1
    r = results[i]
    width = _terminal_width()

    print()
    print("-" * width)
    print(f"  Row {one_idx}: {r['txt_file_name']}")
    print(f"  Current  : {', '.join(r['predicted_sections'])}")
    preview = _trunc(r.get("preview") or "", width - 14)
    print(f"  Preview  : {preview}")
    print()

    raw = input("  New sections (comma-separated), or ENTER to keep, or '?' for catalog: ").strip()

    if raw == "?":
        print(_SECTION_CATALOG)
        print()
        raw = input("  New sections (comma-separated), or ENTER to keep: ").strip()

    if not raw:
        print("  (unchanged)")
        return False

    parsed = _parse_section_ids(raw)
    if parsed is None:
        print("  (no change applied -- fix the input and try again)")
        return False

    old = list(r["predicted_sections"])
    r["predicted_sections"] = parsed
    if old == parsed:
        print("  (same as before -- no change)")
        return False

    removed = sorted(set(old) - set(parsed), key=section_sort_key)
    added = sorted(set(parsed) - set(old), key=section_sort_key)
    diff_parts: list[str] = []
    if removed:
        diff_parts.append(f"removed {', '.join(removed)}")
    if added:
        diff_parts.append(f"added {', '.join(added)}")
    diff_str = " | ".join(diff_parts) if diff_parts else "reordered"
    print(f"  OK  ({diff_str})  =>  {', '.join(parsed)}")
    return True


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def interactive_review(classification_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Interactively display and edit classification results.

    The original LLM predictions are preserved in each result under
    ``llm_predicted_sections`` for auditability. ``predicted_sections`` is
    updated in place with the final human-confirmed values.

    Returns the annotated payload (modified or as-is).
    """
    _ensure_utf8_stdout()

    results: list[dict[str, Any]] = list(classification_payload.get("results") or [])
    if not results:
        print("  No classification results to review.", file=sys.stderr)
        return classification_payload

    document_name = (classification_payload.get("summary") or {}).get("document_name", "?")

    # Preserve original LLM predictions for auditability
    for r in results:
        if "llm_predicted_sections" not in r:
            r["llm_predicted_sections"] = list(r["predicted_sections"])

    total = len(results)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = 0
    modified_indices: set[int] = set()
    confirmed = False

    _print_table(results, modified_indices, page, PAGE_SIZE, document_name)
    print(_HELP_TEXT)

    while True:
        try:
            raw = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted -- discarding all changes.")
            for r in results:
                r["predicted_sections"] = list(r["llm_predicted_sections"])
            modified_indices.clear()
            break

        lower = raw.lower()

        # Confirm
        if lower in ("", "ok", "done", "confirm", "yes", "y", "s", "save"):
            confirmed = True
            break

        # Quit / discard
        if lower in ("q", "quit", "exit"):
            print("  Discarding all changes -- original LLM classifications will be used.")
            for r in results:
                r["predicted_sections"] = list(r["llm_predicted_sections"])
            modified_indices.clear()
            break

        # Help
        if lower in ("?", "help", "h"):
            print(_HELP_TEXT)
            continue

        # Catalog
        if lower == "sections":
            print(_SECTION_CATALOG)
            continue

        # Page navigation: n / next
        if lower in ("n", "next"):
            page = min(page + 1, total_pages - 1)
            _print_table(results, modified_indices, page, PAGE_SIZE, document_name)
            continue

        # Page navigation: p / prev  (only when no digit follows)
        if lower in ("p", "prev", "back"):
            page = max(page - 1, 0)
            _print_table(results, modified_indices, page, PAGE_SIZE, document_name)
            continue

        # Page navigation: p<n>
        if lower.startswith("p") and lower[1:].isdigit():
            target = int(lower[1:]) - 1
            page = max(0, min(target, total_pages - 1))
            _print_table(results, modified_indices, page, PAGE_SIZE, document_name)
            continue

        # Row number
        if raw.isdigit():
            row_num = int(raw)
            if 1 <= row_num <= total:
                changed = _edit_row(results, row_num)
                i = row_num - 1
                if changed:
                    if results[i]["predicted_sections"] == results[i]["llm_predicted_sections"]:
                        modified_indices.discard(i)
                    else:
                        modified_indices.add(i)
                # Jump to the page that contains this row
                page = (row_num - 1) // PAGE_SIZE
                _print_table(results, modified_indices, page, PAGE_SIZE, document_name)
            else:
                print(f"  Row number out of range (1-{total})")
            continue

        print("  Unrecognized command. Type '?' for help.")

    # Annotate each result with its review action
    for i, r in enumerate(results):
        r["review_action"] = "modified" if i in modified_indices else "confirmed"

    n_modified = len(modified_indices)
    if confirmed:
        if n_modified:
            print(f"\n  Review complete -- {n_modified} row(s) modified, "
                  f"{total - n_modified} confirmed.")
        else:
            print(f"\n  Review complete -- all {total} rows confirmed.")
    else:
        print(f"\n  Review exited -- changes discarded, original classifications used.")

    # Build the annotated payload
    updated_payload = dict(classification_payload)
    updated_payload["results"] = results
    summary = dict((classification_payload.get("summary") or {}))
    summary["human_reviewed"] = confirmed
    summary["review_timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary["review_modifications_count"] = n_modified
    updated_payload["summary"] = summary

    return updated_payload


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def review_classified_document(
    classification_output_path: Path,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Load ``classification_output_path``, run the interactive review (unless
    quiet or stdin is not a TTY), save the result back to the same file, and
    return the payload.

    When stdin is not a TTY (piped input, CI) or ``quiet`` is True, the review
    is skipped and the payload is returned unchanged with
    ``human_reviewed: false`` added to the summary.
    """
    raw = json.loads(classification_output_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Classification output must be a JSON object: {classification_output_path}"
        )

    if quiet or not sys.stdin.isatty():
        summary = dict(raw.get("summary") or {})
        summary["human_reviewed"] = False
        raw["summary"] = summary
        return raw

    reviewed = interactive_review(raw)
    write_json(classification_output_path, reviewed)

    n = (reviewed.get("summary") or {}).get("review_modifications_count", 0)
    confirmed = (reviewed.get("summary") or {}).get("human_reviewed", False)
    state = "confirmed" if confirmed else "exited"
    print(
        f"Classification review {state} "
        f"({n} modification(s)) -- saved to "
        f"{project_relative_path(classification_output_path)}",
        file=sys.stderr,
        flush=True,
    )

    return reviewed
