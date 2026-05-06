from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

try:
    from pymilvus import DataType, MilvusClient
except ImportError:  # pragma: no cover - handled at runtime
    DataType = None
    MilvusClient = None

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:  # pragma: no cover - handled at runtime
    CrossEncoder = None
    SentenceTransformer = None


DEFAULT_MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
DEFAULT_MILVUS_DB_NAME = os.environ.get("MILVUS_DB_NAME", "default")
DEFAULT_TOP_K = int(os.environ.get("MILVUS_TOP_K", "12"))
DEFAULT_TOP_N = int(os.environ.get("MILVUS_TOP_N", "5"))
DEFAULT_BATCH_SIZE = int(os.environ.get("MILVUS_EMBED_BATCH_SIZE", "24"))
DEFAULT_HNSW_M = int(os.environ.get("MILVUS_HNSW_M", "32"))
DEFAULT_HNSW_EF_CONSTRUCTION = int(os.environ.get("MILVUS_HNSW_EF_CONSTRUCTION", "400"))
DEFAULT_HNSW_EF_SEARCH = int(os.environ.get("MILVUS_EF_SEARCH", "512"))

EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
RERANKER_MODEL_NAME = "Alibaba-NLP/gte-multilingual-reranker-base"

MAX_COLLECTION_NAME_LENGTH = 255
MAX_DOCUMENT_ID_LENGTH = 256
MAX_DOCUMENT_NAME_LENGTH = 1024
MAX_CHUNK_ID_LENGTH = 256
MAX_FILE_NAME_LENGTH = 512
MAX_SECTION_ID_LENGTH = 32
MAX_CHUNK_TYPE_LENGTH = 64
MAX_HIERARCHY_PATH_LENGTH = 4096
MAX_TEXT_LENGTH = int(os.environ.get("MILVUS_TEXT_MAX_LENGTH", "60000"))
MAX_TEXT_BYTES = int(os.environ.get("MILVUS_TEXT_MAX_BYTES", "60000"))

COLLECTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RagChunk:
    document_id: str
    document_name: str
    chunk_id: str
    file_name: str
    file_path: Path
    section_id: str
    chunk_type: str
    hierarchy_path: str
    text: str


_embedder_lock = threading.Lock()
_embedder: SentenceTransformer | None = None
_embedder_device: str | None = None

_reranker_lock = threading.Lock()
_reranker: CrossEncoder | None = None
_reranker_device: str | None = None


def _repair_position_id_buffers(model_root: Any) -> int:
    repaired = 0
    modules = getattr(model_root, "modules", None)
    if modules is None:
        return repaired
    for module in modules():
        position_ids = getattr(module, "position_ids", None)
        if not isinstance(position_ids, torch.Tensor):
            continue
        expected_prefix: torch.Tensor
        replacement: torch.Tensor
        if position_ids.ndim == 1 and position_ids.numel() > 0:
            expected_prefix = torch.arange(min(position_ids.numel(), 16), device=position_ids.device, dtype=torch.long)
            current_prefix = position_ids[: expected_prefix.numel()].to(dtype=torch.long)
            if torch.equal(current_prefix, expected_prefix):
                continue
            replacement = torch.arange(position_ids.numel(), device=position_ids.device, dtype=torch.long)
        elif position_ids.ndim == 2 and position_ids.shape[0] == 1 and position_ids.shape[1] > 0:
            expected_prefix = torch.arange(min(position_ids.shape[1], 16), device=position_ids.device, dtype=torch.long)
            current_prefix = position_ids[0, : expected_prefix.numel()].to(dtype=torch.long)
            if torch.equal(current_prefix, expected_prefix):
                continue
            replacement = torch.arange(position_ids.shape[1], device=position_ids.device, dtype=torch.long).unsqueeze(0)
        else:
            continue
        module.register_buffer("position_ids", replacement, persistent=False)
        repaired += 1
    return repaired


def dependency_error_message() -> str | None:
    missing: list[str] = []
    if MilvusClient is None or DataType is None:
        missing.append("pymilvus")
    if SentenceTransformer is None or CrossEncoder is None:
        missing.append("sentence-transformers")
    if missing:
        return (
            "Missing Milvus dependencies: "
            + ", ".join(missing)
            + ". Install them in the active Python environment before using the Milvus page."
        )
    return None


def require_dependencies() -> None:
    message = dependency_error_message()
    if message:
        raise RuntimeError(message)


def normalize_milvus_uri(uri: str | None) -> str:
    candidate = (uri or DEFAULT_MILVUS_URI).strip()
    if not candidate:
        return DEFAULT_MILVUS_URI
    if "://" not in candidate:
        return f"http://{candidate}"
    return candidate


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_collection_name(collection_name: str) -> str:
    normalized = (collection_name or "").strip()
    if not normalized:
        raise ValueError("Collection name is required.")
    if len(normalized) > MAX_COLLECTION_NAME_LENGTH:
        raise ValueError(f"Collection name must be at most {MAX_COLLECTION_NAME_LENGTH} characters.")
    if not COLLECTION_NAME_RE.fullmatch(normalized):
        raise ValueError("Collection name must start with a letter or underscore and contain only letters, numbers, and underscores.")
    return normalized


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"RAG TXT manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"RAG TXT manifest is invalid JSON: {manifest_path}") from exc


def load_rag_manifest(document_path: Path) -> dict[str, Any]:
    return _load_manifest(document_path / "rag_txt" / "manifest.json")


def list_rag_documents(document_roots: Iterable[Path]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for document_path in document_roots:
        manifest_path = document_path / "rag_txt" / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _load_manifest(manifest_path)
        chunk_count = int(manifest.get("chunk_count") or 0)
        total_characters = 0
        for chunk in manifest.get("chunks") or []:
            if isinstance(chunk, dict):
                total_characters += int(chunk.get("content_character_count") or 0)
        documents.append({
            "name": document_path.name,
            "rag_chunk_count": chunk_count,
            "total_characters": total_characters,
            "manifest_path": str(manifest_path),
        })
    return documents


def _validate_field_length(label: str, value: str, limit: int) -> None:
    if len(value) > limit:
        raise ValueError(f"{label} exceeds the Milvus field limit ({len(value)} > {limit}).")


def _validate_chunk(chunk: RagChunk) -> None:
    _validate_field_length("Document ID", chunk.document_id, MAX_DOCUMENT_ID_LENGTH)
    _validate_field_length("Document name", chunk.document_name, MAX_DOCUMENT_NAME_LENGTH)
    _validate_field_length("Chunk ID", chunk.chunk_id, MAX_CHUNK_ID_LENGTH)
    _validate_field_length("File name", chunk.file_name, MAX_FILE_NAME_LENGTH)
    _validate_field_length("Section ID", chunk.section_id, MAX_SECTION_ID_LENGTH)
    _validate_field_length("Chunk type", chunk.chunk_type, MAX_CHUNK_TYPE_LENGTH)
    _validate_field_length("Hierarchy path", chunk.hierarchy_path, MAX_HIERARCHY_PATH_LENGTH)
    if len(chunk.text) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"{chunk.document_name} / {chunk.file_name} has {len(chunk.text)} characters, which exceeds the Milvus text limit of {MAX_TEXT_LENGTH}."
        )
    text_bytes = len(chunk.text.encode("utf-8"))
    if text_bytes > MAX_TEXT_BYTES:
        raise ValueError(
            f"{chunk.document_name} / {chunk.file_name} uses {text_bytes} UTF-8 bytes, which exceeds the safety limit of {MAX_TEXT_BYTES}."
        )


def load_document_chunks(
    *,
    document_id: str,
    document_path: Path,
    document_name: str,
) -> list[RagChunk]:
    manifest = load_rag_manifest(document_path)
    rag_dir = document_path / "rag_txt"
    chunks: list[RagChunk] = []
    for raw_chunk in manifest.get("chunks") or []:
        if not isinstance(raw_chunk, dict):
            continue
        file_name = str(raw_chunk.get("file_name") or "").strip()
        if not file_name:
            continue
        file_path = rag_dir / file_name
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"RAG TXT chunk is missing: {file_path}")
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        chunk = RagChunk(
            document_id=document_id,
            document_name=document_name,
            chunk_id=str(raw_chunk.get("chunk_id") or file_path.stem),
            file_name=file_name,
            file_path=file_path,
            section_id=str(raw_chunk.get("section_id") or ""),
            chunk_type=str(raw_chunk.get("chunk_type") or "section_content"),
            hierarchy_path=str(raw_chunk.get("hierarchy_path") or ""),
            text=text,
        )
        _validate_chunk(chunk)
        chunks.append(chunk)
    if not chunks:
        raise RuntimeError(f"No non-empty RAG TXT files were found for {document_name}.")
    return chunks


def collect_chunks(
    *,
    documents: list[dict[str, Any]],
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[RagChunk], list[dict[str, Any]]]:
    all_chunks: list[RagChunk] = []
    summaries: list[dict[str, Any]] = []
    total_docs = len(documents)
    for index, document in enumerate(documents, start=1):
        if progress_callback is not None:
            progress_callback({
                "stage": "validate",
                "current": index - 1,
                "total": total_docs,
                "detail": f"Validating RAG TXT files for {document['name']} ({index} of {total_docs})",
            })
        chunks = load_document_chunks(
            document_id=str(document["id"]),
            document_path=Path(document["path"]),
            document_name=str(document["name"]),
        )
        summaries.append({
            "id": str(document["id"]),
            "name": str(document["name"]),
            "chunk_count": len(chunks),
        })
        all_chunks.extend(chunks)
        if progress_callback is not None:
            progress_callback({
                "stage": "validate",
                "current": index,
                "total": total_docs,
                "detail": f"Validated {len(chunks)} files for {document['name']}",
            })
    return all_chunks, summaries


def get_embedder(device: str) -> SentenceTransformer:
    require_dependencies()
    global _embedder, _embedder_device
    with _embedder_lock:
        if _embedder is None or _embedder_device != device:
            _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device, trust_remote_code=True)
            _repair_position_id_buffers(_embedder)
            _embedder_device = device
        return _embedder


def get_reranker(device: str) -> CrossEncoder:
    require_dependencies()
    global _reranker, _reranker_device
    with _reranker_lock:
        if _reranker is None or _reranker_device != device:
            _reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device, trust_remote_code=True)
            _repair_position_id_buffers(_reranker.model)
            _reranker_device = device
        return _reranker


def embed_texts(texts: list[str], *, device: str, batch_size: int) -> np.ndarray:
    model = get_embedder(device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _safe_json_float(value: Any, *, default: float | None = None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def rerank_hits(question: str, hits: list[dict[str, Any]], *, top_n: int, device: str) -> list[dict[str, Any]]:
    if not hits:
        return []
    reranker = get_reranker(device)
    pairs = [(question, str(hit.get("text") or "")) for hit in hits]
    scores = reranker.predict(pairs, batch_size=min(DEFAULT_BATCH_SIZE, len(pairs)), show_progress_bar=False)
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    ranked_hits = []
    for hit, score in zip(hits, flat_scores, strict=False):
        updated = dict(hit)
        updated["reranker_score"] = _safe_json_float(score, default=None)
        updated["_reranker_sort_score"] = _safe_json_float(score, default=float("-inf"))
        ranked_hits.append(updated)
    ranked_hits.sort(key=lambda item: item.get("_reranker_sort_score", float("-inf")), reverse=True)

    return [
        {
            "document_id": str(hit.get("document_id") or ""),
            "document_name": hit.get("document_name") or "Unknown document",
            "best_chunk_id": hit.get("chunk_id") or "",
            "file_name": hit.get("file_name") or "",
            "section_id": hit.get("section_id") or "",
            "chunk_type": hit.get("chunk_type") or "",
            "hierarchy_path": hit.get("hierarchy_path") or "",
            "text": hit.get("text") or "",
            "milvus_score": _safe_json_float(hit.get("milvus_score"), default=None),
            "reranker_score": _safe_json_float(hit.get("reranker_score"), default=None),
        }
        for hit in ranked_hits[:top_n]
    ]


def _connect_client(*, milvus_uri: str, milvus_token: str, milvus_db_name: str) -> MilvusClient:
    require_dependencies()
    kwargs: dict[str, Any] = {
        "uri": normalize_milvus_uri(milvus_uri),
        "db_name": (milvus_db_name or DEFAULT_MILVUS_DB_NAME).strip() or DEFAULT_MILVUS_DB_NAME,
    }
    if milvus_token.strip():
        kwargs["token"] = milvus_token.strip()
    return MilvusClient(**kwargs)


def _ensure_collection(client: MilvusClient, *, collection_name: str, vector_dim: int) -> bool:
    recreated = False
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        recreated = True

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=MAX_DOCUMENT_ID_LENGTH)
    schema.add_field(field_name="document_name", datatype=DataType.VARCHAR, max_length=MAX_DOCUMENT_NAME_LENGTH)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=MAX_CHUNK_ID_LENGTH)
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=MAX_FILE_NAME_LENGTH)
    schema.add_field(field_name="section_id", datatype=DataType.VARCHAR, max_length=MAX_SECTION_ID_LENGTH)
    schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=MAX_CHUNK_TYPE_LENGTH)
    schema.add_field(field_name="hierarchy_path", datatype=DataType.VARCHAR, max_length=MAX_HIERARCHY_PATH_LENGTH)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={
            "M": DEFAULT_HNSW_M,
            "efConstruction": DEFAULT_HNSW_EF_CONSTRUCTION,
        },
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    client.load_collection(collection_name=collection_name)
    return recreated


def _chunked(items: list[RagChunk], batch_size: int) -> Iterable[list[RagChunk]]:
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def ingest_documents(
    *,
    collection_name: str,
    milvus_uri: str,
    milvus_token: str = "",
    milvus_db_name: str = DEFAULT_MILVUS_DB_NAME,
    documents: list[dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    require_dependencies()
    validated_collection_name = validate_collection_name(collection_name)
    resolved_device = resolve_device()
    clean_batch_size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))

    if not documents:
        raise ValueError("Select at least one document with RAG TXT files.")

    all_chunks, document_summaries = collect_chunks(documents=documents, progress_callback=progress_callback)
    if progress_callback is not None:
        progress_callback({
            "stage": "model",
            "current": 0,
            "total": 1,
            "detail": f"Loading embedding model on {resolved_device}",
        })
    embedder = get_embedder(resolved_device)
    vector_dim = int(embedder.get_sentence_embedding_dimension())
    if progress_callback is not None:
        progress_callback({
            "stage": "connect",
            "current": 0,
            "total": 1,
            "detail": "Connecting to Milvus",
        })
    client = _connect_client(milvus_uri=milvus_uri, milvus_token=milvus_token, milvus_db_name=milvus_db_name)
    recreated = _ensure_collection(client, collection_name=validated_collection_name, vector_dim=vector_dim)

    batches = list(_chunked(all_chunks, clean_batch_size))
    total_batches = max(1, len(batches))
    inserted_count = 0
    for batch_index, batch in enumerate(batches, start=1):
        if progress_callback is not None:
            progress_callback({
                "stage": "embed",
                "current": batch_index - 1,
                "total": total_batches,
                "detail": f"Embedding batch {batch_index} of {total_batches}",
            })
        embeddings = embed_texts([chunk.text for chunk in batch], device=resolved_device, batch_size=clean_batch_size)
        rows = []
        for chunk, embedding in zip(batch, embeddings, strict=False):
            rows.append({
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "chunk_id": chunk.chunk_id,
                "file_name": chunk.file_name,
                "section_id": chunk.section_id,
                "chunk_type": chunk.chunk_type,
                "hierarchy_path": chunk.hierarchy_path,
                "text": chunk.text,
                "embedding": embedding.tolist(),
            })
        if progress_callback is not None:
            progress_callback({
                "stage": "insert",
                "current": batch_index - 1,
                "total": total_batches,
                "detail": f"Inserting batch {batch_index} of {total_batches} into {validated_collection_name}",
            })
        client.insert(collection_name=validated_collection_name, data=rows)
        inserted_count += len(rows)
        if progress_callback is not None:
            progress_callback({
                "stage": "insert",
                "current": batch_index,
                "total": total_batches,
                "detail": f"Inserted {inserted_count} of {len(all_chunks)} chunks",
            })
    client.load_collection(collection_name=validated_collection_name)
    return {
        "collection_name": validated_collection_name,
        "milvus_uri": normalize_milvus_uri(milvus_uri),
        "milvus_db_name": (milvus_db_name or DEFAULT_MILVUS_DB_NAME).strip() or DEFAULT_MILVUS_DB_NAME,
        "device": resolved_device,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "document_count": len(document_summaries),
        "chunk_count": inserted_count,
        "embedding_dimension": vector_dim,
        "recreated": recreated,
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "hnsw_m": DEFAULT_HNSW_M,
        "hnsw_ef_construction": DEFAULT_HNSW_EF_CONSTRUCTION,
        "documents": document_summaries,
    }


def _normalize_hit(raw_hit: dict[str, Any], rank: int) -> dict[str, Any]:
    entity = raw_hit.get("entity") if isinstance(raw_hit, dict) else None
    if not isinstance(entity, dict):
        entity = raw_hit
    text = str(entity.get("text") or "")
    return {
        "rank": rank,
        "id": raw_hit.get("id") if isinstance(raw_hit, dict) else None,
        "document_id": str(entity.get("document_id") or ""),
        "document_name": str(entity.get("document_name") or ""),
        "chunk_id": str(entity.get("chunk_id") or ""),
        "file_name": str(entity.get("file_name") or ""),
        "section_id": str(entity.get("section_id") or ""),
        "chunk_type": str(entity.get("chunk_type") or ""),
        "hierarchy_path": str(entity.get("hierarchy_path") or ""),
        "text": text,
        "milvus_score": _safe_json_float(raw_hit.get("distance") if isinstance(raw_hit, dict) else 0.0, default=None),
    }


_QUERY_OUTPUT_FIELDS = [
    "document_id",
    "document_name",
    "chunk_id",
    "file_name",
    "section_id",
    "chunk_type",
    "hierarchy_path",
    "text",
]


def _quote_milvus_string(value: str) -> str:
    return json.dumps(str(value))


def _normalize_document_ids(document_ids: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_document_id in document_ids or []:
        document_id = str(raw_document_id or "").strip()
        if not document_id or document_id in seen:
            continue
        normalized.append(document_id)
        seen.add(document_id)
    return normalized


def _build_document_filter(document_ids: list[str]) -> str:
    if not document_ids:
        return ""
    if len(document_ids) == 1:
        return f'document_id == {_quote_milvus_string(document_ids[0])}'
    return f"document_id in {json.dumps(document_ids, ensure_ascii=False)}"


def query_collection(
    *,
    question: str,
    collection_name: str,
    milvus_uri: str,
    milvus_token: str = "",
    milvus_db_name: str = DEFAULT_MILVUS_DB_NAME,
    top_k: int = DEFAULT_TOP_K,
    top_n: int = DEFAULT_TOP_N,
    document_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    require_dependencies()
    query_text = (question or "").strip()
    if not query_text:
        raise ValueError("Question is required.")
    validated_collection_name = validate_collection_name(collection_name)
    clean_top_k = max(1, int(top_k or DEFAULT_TOP_K))
    clean_top_n = max(1, int(top_n or DEFAULT_TOP_N))
    resolved_device = resolve_device()
    normalized_document_ids = _normalize_document_ids(document_ids)
    document_filter = _build_document_filter(normalized_document_ids)
    ef_search = max(DEFAULT_HNSW_EF_SEARCH, clean_top_k)

    client = _connect_client(milvus_uri=milvus_uri, milvus_token=milvus_token, milvus_db_name=milvus_db_name)
    if not client.has_collection(collection_name=validated_collection_name):
        raise RuntimeError(f"Collection {validated_collection_name} was not found in Milvus.")
    client.load_collection(collection_name=validated_collection_name)
    if len(normalized_document_ids) == 1:
        raw_hits = client.query(
            collection_name=validated_collection_name,
            filter=document_filter,
            output_fields=_QUERY_OUTPUT_FIELDS,
        )
    else:
        query_embedding = embed_texts([query_text], device=resolved_device, batch_size=1)[0].tolist()
        raw_results = client.search(
            collection_name=validated_collection_name,
            data=[query_embedding],
            filter=document_filter,
            limit=clean_top_k,
            output_fields=_QUERY_OUTPUT_FIELDS,
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": ef_search},
            },
        )
        raw_hits = raw_results[0] if raw_results else []
    milvus_hits = [_normalize_hit(hit, rank) for rank, hit in enumerate(raw_hits, start=1)]
    reranked_documents = rerank_hits(query_text, milvus_hits, top_n=clean_top_n, device=resolved_device)
    return {
        "question": query_text,
        "collection_name": validated_collection_name,
        "top_k": clean_top_k,
        "top_n": clean_top_n,
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "ef_search": ef_search,
        "milvus_hits": milvus_hits,
        "reranked_documents": reranked_documents,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "device": resolved_device,
    }