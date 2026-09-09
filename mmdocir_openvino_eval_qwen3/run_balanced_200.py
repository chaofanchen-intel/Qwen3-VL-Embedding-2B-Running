#!/usr/bin/env python3
"""Run an isolated Qwen3-VL-Embedding-2B MMDocIR balanced_200 evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import openvino as ov
import openvino_genai
import pyarrow.parquet as parquet
import psutil
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_DATASET = Path("/home/chaofan/datasets/MMDocIR_Evaluation_Dataset")
DEFAULT_SUBSET = HERE / "subsets" / "balanced_200.json"
DEFAULT_MODELS = REPO.parent / "benchmark" / "models"
DEFAULT_RESULTS = HERE / "results" / "balanced_200"
DOCUMENT_PROMPT = "Represent the user's input."
QUERY_PROMPT = "Represent the user's input."


@dataclass(frozen=True)
class RowRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid row range {self.start}..{self.end}")


@dataclass(frozen=True)
class GroundTruthLayout:
    page_id: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Query:
    query_id: int
    text: str
    relevant_page_ids: tuple[int, ...]
    relevant_layouts: tuple[GroundTruthLayout, ...]
    doc_name: str
    page_rows: RowRange
    layout_rows: RowRange


@dataclass(frozen=True)
class Document:
    doc_name: str
    page_rows: RowRange
    layout_rows: RowRange
    queries: tuple[Query, ...]


@dataclass(frozen=True)
class LayoutCandidate:
    doc_name: str
    page_id: int
    bbox: tuple[float, float, float, float]


def pair(value: Sequence[Any]) -> RowRange:
    if len(value) != 2:
        raise ValueError(f"expected a [start, end] range, got {value!r}")
    return RowRange(int(value[0]), int(value[1]))


def load_subset_ids(path: Path) -> tuple[dict[str, Any], set[int]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    query_ids = {
        int(query_id)
        for domain in manifest.get("domains", [])
        for query_id in domain.get("query_ids", [])
    }
    declared = manifest.get("query_count")
    if declared is not None and int(declared) != len(query_ids):
        raise ValueError(f"subset query_count={declared}, actual={len(query_ids)}")
    if not query_ids:
        raise ValueError(f"subset has no query ids: {path}")
    return manifest, query_ids


def load_documents(path: Path, selected_ids: set[int]) -> tuple[Document, ...]:
    documents: list[Document] = []
    found: set[int] = set()
    global_query_id = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            page_rows = pair(item["page_indices"])
            layout_rows = pair(item.get("layout_indices", item.get("layout_indinces")))
            queries: list[Query] = []
            for question in item["questions"]:
                if global_query_id in selected_ids:
                    layouts = tuple(
                        GroundTruthLayout(
                            page_id=int(layout["page"]),
                            bbox=tuple(float(value) for value in layout["bbox"]),
                        )
                        for layout in question["layout_mapping"]
                    )
                    queries.append(
                        Query(
                            query_id=global_query_id,
                            text=str(question["Q"]),
                            relevant_page_ids=tuple(int(value) for value in question["page_id"]),
                            relevant_layouts=layouts,
                            doc_name=str(item["doc_name"]),
                            page_rows=page_rows,
                            layout_rows=layout_rows,
                        )
                    )
                    found.add(global_query_id)
                global_query_id += 1
            if queries:
                documents.append(
                    Document(str(item["doc_name"]), page_rows, layout_rows, tuple(queries))
                )
    missing = sorted(selected_ids - found)
    if missing:
        raise ValueError(f"subset query ids missing from annotations: {missing[:10]}")
    return tuple(documents)


def selected_count(documents: Sequence[Document], kind: str) -> int:
    ranges = [document.page_rows if kind == "page" else document.layout_rows for document in documents]
    return sum(item.end - item.start + 1 for item in ranges)


def iter_selected_rows(
    path: Path,
    columns: Sequence[str],
    documents: Sequence[Document],
    kind: str,
) -> Iterable[tuple[int, dict[str, Any]]]:
    ranges = sorted(
        (document.page_rows if kind == "page" else document.layout_rows for document in documents),
        key=lambda item: item.start,
    )
    if not ranges:
        return
    range_index = 0
    row_offset = 0
    data = parquet.ParquetFile(path)
    for row_group_index in range(data.num_row_groups):
        row_count = data.metadata.row_group(row_group_index).num_rows
        row_end = row_offset + row_count - 1
        while range_index < len(ranges) and row_offset > ranges[range_index].end:
            range_index += 1
        if range_index >= len(ranges):
            return
        if ranges[range_index].start <= row_end:
            batch_offset = 0
            for record_batch in data.iter_batches(
                batch_size=256, row_groups=[row_group_index], columns=list(columns)
            ):
                rows = record_batch.to_pylist()
                for local_index, row in enumerate(rows):
                    global_index = row_offset + batch_offset + local_index
                    while range_index < len(ranges) and global_index > ranges[range_index].end:
                        range_index += 1
                    if range_index >= len(ranges):
                        return
                    if ranges[range_index].start <= global_index <= ranges[range_index].end:
                        yield global_index, row
                batch_offset += len(rows)
        row_offset += row_count


def owner_maps(documents: Sequence[Document]) -> tuple[dict[int, tuple[str, int]], dict[int, str]]:
    pages: dict[int, tuple[str, int]] = {}
    layouts: dict[int, str] = {}
    for document in documents:
        for index in range(document.page_rows.start, document.page_rows.end + 1):
            pages[index] = (document.doc_name, index - document.page_rows.start)
        for index in range(document.layout_rows.start, document.layout_rows.end + 1):
            layouts[index] = document.doc_name
    return pages, layouts


def load_rgb(binary: bytes | bytearray | memoryview) -> Image.Image:
    with Image.open(BytesIO(bytes(binary))) as image:
        return image.convert("RGB").copy()


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def result_vectors(result: Any) -> np.ndarray:
    tensor = result.embeddings
    return np.asarray(tensor.data, dtype=np.float32).reshape(tuple(tensor.shape))


class Qwen3Backend:
    """Self-contained Qwen3 backend; no dependency on another model repository."""

    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = model_dir
        self.device = device
        self.pipeline = openvino_genai.EmbeddingPipeline(str(model_dir), device)
        self.embedding_dim = 2048

    def text(self, texts: Sequence[str], prompt: str) -> np.ndarray:
        return result_vectors(self.pipeline.embed(list(texts), prompt=prompt))

    def image_one(self, image: Image.Image) -> np.ndarray:
        tensor = ov.Tensor(np.asarray(image, dtype=np.uint8))
        return result_vectors(self.pipeline.embed(images=[tensor], prompt=DOCUMENT_PROMPT))[0]

    def fused_one(self, text: str, image: Image.Image) -> np.ndarray:
        tensor = ov.Tensor(np.asarray(image, dtype=np.uint8))
        return result_vectors(self.pipeline.embed(text, images=[tensor], prompt=DOCUMENT_PROMPT))[0]


def load_or_encode(
    path: Path,
    count: int,
    dim: int,
    producer: callable,
    label: str,
    force: bool,
) -> tuple[np.ndarray, float, bool]:
    meta_path = path.with_suffix(".json")
    if not force and path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("count") == count and meta.get("dim") == dim:
            return np.load(path), float(meta.get("seconds", 0.0)), True
    started = time.perf_counter()
    vectors = np.asarray(producer(), dtype=np.float32)
    if vectors.shape != (count, dim):
        raise RuntimeError(f"{label} returned {vectors.shape}, expected {(count, dim)}")
    elapsed = time.perf_counter() - started
    np.save(path, vectors)
    meta_path.write_text(
        json.dumps({"count": count, "dim": dim, "seconds": elapsed}, indent=2) + "\n",
        encoding="utf-8",
    )
    return vectors, elapsed, False


def page_metrics(ranked: Sequence[int], relevant: Sequence[int]) -> dict[str, float]:
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("query has no relevant pages")
    output: dict[str, float] = {}
    for k in (5, 10):
        gains = [1.0 if value in relevant_set else 0.0 for value in ranked[:k]]
        ideal = [1.0] * min(k, len(relevant_set))
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal_dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
        output[f"recall@{k}"] = len(set(ranked[:k]) & relevant_set) / len(relevant_set)
        output[f"ndcg@{k}"] = dcg / ideal_dcg if ideal_dcg else 0.0
    return output


def area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def intersection(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def layout_metrics(
    ranked: Sequence[LayoutCandidate],
    all_candidates: Sequence[LayoutCandidate],
    truth: Sequence[LayoutCandidate],
) -> dict[str, float] | None:
    total_area = sum(area(item.bbox) for item in truth)
    if total_area <= 0:
        return None

    def gain(candidate: LayoutCandidate) -> float:
        return sum(
            intersection(candidate.bbox, target.bbox)
            for target in truth
            if candidate.doc_name == target.doc_name and candidate.page_id == target.page_id
        )

    ranked_gains = [gain(item) for item in ranked]
    ideal_gains = sorted((gain(item) for item in all_candidates), reverse=True)
    output: dict[str, float] = {}
    for k in (5, 10):
        ideal = sum(value / math.log2(index + 2) for index, value in enumerate(ideal_gains[:k]))
        actual = sum(value / math.log2(index + 2) for index, value in enumerate(ranked_gains[:k]))
        output[f"recall@{k}"] = sum(ranked_gains[:k]) / total_area
        output[f"ndcg@{k}"] = actual / ideal if ideal else 0.0
    return output


def aggregate(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def evaluate_scope(
    scope: str,
    queries: Sequence[Query],
    query_vectors: np.ndarray,
    page_vectors: np.ndarray,
    page_rows: Sequence[dict[str, Any]],
    layout_vectors: np.ndarray,
    layout_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    page_by_doc: dict[str, list[int]] = {}
    layout_by_doc: dict[str, list[int]] = {}
    layout_by_doc_page: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(page_rows):
        page_by_doc.setdefault(row["doc_name"], []).append(index)
    candidates = [
        LayoutCandidate(row["doc_name"], row["page_id"], tuple(row["bbox"]))
        for row in layout_rows
    ]
    for index, candidate in enumerate(candidates):
        layout_by_doc.setdefault(candidate.doc_name, []).append(index)
        layout_by_doc_page.setdefault((candidate.doc_name, candidate.page_id), []).append(index)

    page_metrics_rows: list[dict[str, float]] = []
    layout_metrics_rows: list[dict[str, float]] = []
    exclusions: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    max_k = 10
    for query_index, query in enumerate(queries):
        if scope == "close":
            page_indices = page_by_doc[query.doc_name]
            layout_indices = layout_by_doc[query.doc_name]
            relevant_pages = list(query.relevant_page_ids)
        else:
            page_indices = list(range(len(page_rows)))
            layout_indices = list(range(len(candidates)))
            relevant_pages = [query.page_rows.start + value for value in query.relevant_page_ids]
        page_scores = page_vectors[page_indices] @ query_vectors[query_index]
        page_order = np.argsort(-page_scores, kind="stable")[:max_k]
        ranked_page_rows = [page_rows[page_indices[position]] for position in page_order]
        page_ids = [row["page_id"] if scope == "close" else row["global_index"] for row in ranked_page_rows]
        page_result = page_metrics(page_ids, relevant_pages)
        page_metrics_rows.append(page_result)

        layout_scores = layout_vectors[layout_indices] @ query_vectors[query_index]
        layout_order = np.argsort(-layout_scores, kind="stable")[:max_k]
        ranked_layouts = [candidates[layout_indices[position]] for position in layout_order]
        truth = [LayoutCandidate(query.doc_name, item.page_id, item.bbox) for item in query.relevant_layouts]
        ideal_indices = {
            index
            for target in truth
            for index in layout_by_doc_page.get((target.doc_name, target.page_id), [])
        }
        layout_result = layout_metrics(ranked_layouts, [candidates[index] for index in ideal_indices], truth)
        if layout_result is None:
            exclusions.append({"query_id": query.query_id, "doc_name": query.doc_name, "bboxes": [list(item.bbox) for item in truth]})
        else:
            layout_metrics_rows.append(layout_result)
        details.append({"query_id": query.query_id, "page": page_result, "layout": layout_result})

    return {
        "scope": scope,
        "queries": len(queries),
        "page": aggregate(page_metrics_rows),
        "layout": aggregate(layout_metrics_rows),
        "layout_queries": len(layout_metrics_rows),
        "layout_excluded_queries": len(exclusions),
        "layout_exclusions": exclusions,
    }, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--precisions", default="INT8,INT4")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset, query_ids = load_subset_ids(args.subset.resolve())
    dataset_dir = args.dataset_dir.resolve()
    annotations = dataset_dir / "MMDocIR_annotations.jsonl"
    pages_path = dataset_dir / "MMDocIR_pages.parquet"
    layouts_path = dataset_dir / "MMDocIR_layouts.parquet"
    documents = load_documents(annotations, query_ids)
    queries = [query for document in documents for query in document.queries]
    queries.sort(key=lambda item: item.query_id)
    selected_pages = selected_count(documents, "page")
    selected_layouts = selected_count(documents, "layout")
    page_owners, layout_owners = owner_maps(documents)
    page_rows_raw = list(iter_selected_rows(pages_path, ["doc_name", "passage_id"], documents, "page"))
    layout_rows_raw = list(iter_selected_rows(layouts_path, ["doc_name", "type", "layout_id", "page_id", "bbox", "text", "vlm_text"], documents, "layout"))
    if len(page_rows_raw) != selected_pages or len(layout_rows_raw) != selected_layouts:
        raise RuntimeError(f"selected parquet counts page={len(page_rows_raw)}/{selected_pages}, layout={len(layout_rows_raw)}/{selected_layouts}")
    page_rows = [
        {"global_index": global_index, "doc_name": page_owners[global_index][0], "page_id": page_owners[global_index][1], "passage_id": int(row["passage_id"])}
        for global_index, row in page_rows_raw
    ]
    layout_rows: list[dict[str, Any]] = []
    for global_index, row in layout_rows_raw:
        layout_type = str(row["type"]).lower()
        layout_rows.append({"global_index": global_index, "doc_name": layout_owners[global_index], "layout_id": int(row["layout_id"]), "page_id": int(row["page_id"]), "bbox": [float(value) for value in row["bbox"]], "type": layout_type, "text": row["text"] or "", "vlm_text": row["vlm_text"] or ""})

    summary: dict[str, Any] = {
        "model": "Qwen3-VL-Embedding-2B",
        "dataset": {"path": str(dataset_dir), "subset": str(args.subset.resolve()), "subset_manifest": subset, "queries": len(queries), "pages": selected_pages, "layouts": selected_layouts},
        "runtime": {"openvino": ov.__version__, "device": args.device, "precisions": [value.strip().upper() for value in args.precisions.split(",")], "embedding_dim": 2048, "document_prompt": DOCUMENT_PROMPT, "query_prompt": QUERY_PROMPT, "onednn_primitive_cache_capacity": os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY")},
        "runs": {},
    }
    for precision in [value.strip().upper() for value in args.precisions.split(",") if value.strip()]:
        model_dir = args.models_root.resolve() / f"Qwen3-VL-Embedding-2B-{precision}"
        precision_dir = args.output_dir / precision.lower()
        precision_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{precision}] loading {model_dir}", flush=True)
        started = time.perf_counter()
        backend = Qwen3Backend(model_dir, args.device)
        load_seconds = time.perf_counter() - started
        cache_dir = precision_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        query_vectors, query_seconds, query_hit = load_or_encode(cache_dir / "queries.npy", len(queries), 2048, lambda: normalize(backend.text([query.text for query in queries], QUERY_PROMPT)), "queries", args.force)
        def encode_pages() -> np.ndarray:
            output = np.empty((len(page_rows), 2048), dtype=np.float32)
            for index, (_, row) in enumerate(
                iter_selected_rows(pages_path, ["image_binary"], documents, "page")
            ):
                output[index] = backend.image_one(load_rgb(row["image_binary"]))
                if (index + 1) % 25 == 0:
                    print(f"[{precision}] pages {index + 1}/{len(page_rows)}", flush=True)
            return normalize(output)

        page_vectors, page_seconds, page_hit = load_or_encode(
            cache_dir / "pages.npy", len(page_rows), 2048, encode_pages, "pages", args.force
        )
        def encode_layouts() -> np.ndarray:
            output = np.empty((len(layout_rows), 2048), dtype=np.float32)
            text_positions = [index for index, row in enumerate(layout_rows) if row["type"] not in {"image", "table"}]
            for start in range(0, len(text_positions), args.text_batch_size):
                positions = text_positions[start:start + args.text_batch_size]
                output[positions] = normalize(backend.text([layout_rows[index]["text"] for index in positions], DOCUMENT_PROMPT))
            for index, (_, row) in enumerate(
                iter_selected_rows(layouts_path, ["type", "image_binary", "vlm_text"], documents, "layout")
            ):
                if str(row["type"]).lower() in {"image", "table"}:
                    output[index] = backend.fused_one(row["vlm_text"] or "", load_rgb(row["image_binary"]))
                if (index + 1) % 50 == 0:
                    print(f"[{precision}] layouts {index + 1}/{len(layout_rows)}", flush=True)
            return normalize(output)
        layout_vectors, layout_seconds, layout_hit = load_or_encode(cache_dir / "layouts.npy", len(layout_rows), 2048, encode_layouts, "layouts", args.force)
        evaluations: dict[str, Any] = {}
        for scope in ("close", "open"):
            evaluation, details = evaluate_scope(scope, queries, query_vectors, page_vectors, page_rows, layout_vectors, layout_rows)
            evaluations[scope] = evaluation
            (precision_dir / f"per_query_{scope}.jsonl").write_text("\n".join(json.dumps(item) for item in details) + "\n", encoding="utf-8")
        summary["runs"][precision] = {"model_dir": str(model_dir), "load_seconds": load_seconds, "encoding": {"queries": {"seconds": query_seconds, "cache_hit": query_hit}, "pages": {"seconds": page_seconds, "cache_hit": page_hit}, "layouts": {"seconds": layout_seconds, "cache_hit": layout_hit}}, "evaluation": evaluations}
        (precision_dir / "summary.json").write_text(json.dumps(summary["runs"][precision], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        del backend
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
