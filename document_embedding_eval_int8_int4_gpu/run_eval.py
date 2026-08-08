#!/usr/bin/env python3
"""Benchmark Qwen3-VL-Embedding-2B on a PDF document with GPU INT8/INT4."""
from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import fitz
import numpy as np
import openvino as ov
import openvino_genai
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parent
RUNNING = ROOT.parent
PDF_PATH = RUNNING / "060_OT_Downlook_PR_Error.pdf"
MODELS_ROOT = RUNNING.parent / "benchmark" / "models"
RESULTS = ROOT / "results"
PAGES = ROOT / "pages"
BATCHES = (1, 4, 8, 14)
REPEATS = 3
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
DPI = 150
DOCUMENT_PROMPT = "Represent the user's input."
QUERY_PROMPT = "Represent the user's input."

QUALITY_QUERIES = [
    {"id": "halo", "text": "What causes the halo effect on the edge bump?", "pages": [2]},
    {"id": "pr_error", "text": "What should be checked when Downlook PR fails?", "pages": [2, 3, 4, 5]},
    {"id": "flux", "text": "How should flux coverage be checked?", "pages": [3]},
    {"id": "substrate", "text": "What substrate should be used for PR learning?", "pages": [3]},
    {"id": "threshold", "text": "How should the PR threshold and area be adjusted?", "pages": [4]},
    {"id": "lighting", "text": "How should the PR lighting be adjusted?", "pages": [4]},
    {"id": "inspection_log", "text": "How do I turn on the Vision inspection log?", "pages": [5, 6]},
    {"id": "dump_files", "text": "How do I save Vision Dump files and system files?", "pages": [11, 12, 13]},
]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(float(statistics.mean(values)), 3),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "min": round(float(min(values)), 3),
        "max": round(float(max(values)), 3),
    }


def token_count(tokenizer: openvino_genai.Tokenizer, text: str) -> int:
    return int(tokenizer.encode(text).input_ids.shape[1])


def extract_pdf() -> tuple[list[str], dict[str, Any]]:
    started = time.perf_counter()
    document = fitz.open(PDF_PATH)
    open_ms = (time.perf_counter() - started) * 1000
    texts: list[str] = []
    extract_times: list[float] = []
    for page in document:
        page_started = time.perf_counter()
        texts.append(page.get_text("text", sort=True).strip())
        extract_times.append((time.perf_counter() - page_started) * 1000)
    document.close()
    return texts, {
        "path": str(PDF_PATH),
        "pages": len(texts),
        "file_bytes": PDF_PATH.stat().st_size,
        "open_ms": round(open_ms, 3),
        "text_extract_ms": stats(extract_times),
        "text_chars": sum(len(text) for text in texts),
    }


def render_pages() -> tuple[list[Path], dict[str, Any]]:
    PAGES.mkdir(parents=True, exist_ok=True)
    document = fitz.open(PDF_PATH)
    paths: list[Path] = []
    render_times: list[float] = []
    matrix = fitz.Matrix(DPI / 72.0, DPI / 72.0)
    for index, page in enumerate(document, 1):
        path = PAGES / f"page_{index:02d}_{DPI}dpi.png"
        started = time.perf_counter()
        page.get_pixmap(matrix=matrix, alpha=False).save(path)
        render_times.append((time.perf_counter() - started) * 1000)
        paths.append(path)
    document.close()
    first_size = Image.open(paths[0]).size
    return paths, {
        "dpi": DPI,
        "render_ms": stats(render_times),
        "image_size_wh": list(first_size),
        "total_bytes": sum(path.stat().st_size for path in paths),
    }


def make_chunks(page_texts: list[str], tokenizer: openvino_genai.Tokenizer) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page_index, text in enumerate(page_texts, 1):
        encoded = tokenizer.encode(text).input_ids
        ids = np.asarray(encoded.data).reshape(-1).tolist()
        if not ids:
            chunks.append({"id": f"page{page_index}_chunk0", "page": page_index, "text": "", "tokens": 0})
            continue
        start = 0
        chunk_index = 0
        while start < len(ids):
            end = min(start + CHUNK_SIZE, len(ids))
            chunk_ids = ids[start:end]
            chunks.append({
                "id": f"page{page_index}_chunk{chunk_index}",
                "page": page_index,
                "text": tokenizer.decode(chunk_ids, skip_special_tokens=True),
                "tokens": len(chunk_ids),
            })
            if end == len(ids):
                break
            start = end - CHUNK_OVERLAP
            chunk_index += 1
    return chunks


def build_inputs(
    page_texts: list[str],
    page_images: list[Path],
    tokenizer: openvino_genai.Tokenizer,
) -> dict[str, Any]:
    page_tokens = [token_count(tokenizer, text) for text in page_texts]
    chunks = make_chunks(page_texts, tokenizer)
    return {
        "page_text": [
            {"id": f"page{index}", "page": index, "text": text, "tokens": page_tokens[index - 1]}
            for index, text in enumerate(page_texts, 1)
        ],
        "text_chunk_256": chunks,
        "page_image_150": [
            {"id": f"page{index}", "page": index, "image": str(path)}
            for index, path in enumerate(page_images, 1)
        ],
        "page_fused_150": [
            {"id": f"page{index}", "page": index, "text": text, "image": str(image), "tokens": page_tokens[index - 1]}
            for index, (text, image) in enumerate(zip(page_texts, page_images), 1)
        ],
        "token_stats": {
            "page_text": stats_int(page_tokens),
            "text_chunk_256": stats_int([item["tokens"] for item in chunks]),
        },
    }


def stats_int(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "p50": round(percentile([float(value) for value in values], 50), 3),
        "mean": round(float(statistics.mean(values)), 3),
        "max": max(values),
    }


def image_tensor(path: str) -> ov.Tensor:
    return ov.Tensor(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))


def embedding_array(result: Any) -> np.ndarray:
    tensor = result.embeddings
    return np.asarray(tensor.data, dtype=np.float32).reshape(tuple(tensor.shape))


def text_call(pipeline: Any, texts: list[str], prompt: str) -> np.ndarray:
    return embedding_array(pipeline.embed(texts, prompt=prompt))


def image_call(pipeline: Any, image: ov.Tensor) -> np.ndarray:
    return embedding_array(pipeline.embed("", images=[image], prompt=DOCUMENT_PROMPT))


def fused_call(pipeline: Any, text: str, image: ov.Tensor) -> np.ndarray:
    return embedding_array(pipeline.embed(text, images=[image], prompt=DOCUMENT_PROMPT))


def benchmark_text(
    pipeline: Any,
    items: list[dict[str, Any]],
    repeat: int,
    prompt: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    texts = [item["text"] for item in items]
    total_tokens = sum(item.get("tokens", 0) for item in items)
    for batch_size in BATCHES:
        def invoke() -> np.ndarray:
            vectors = [text_call(pipeline, texts[start:start + batch_size], prompt)
                       for start in range(0, len(texts), batch_size)]
            return np.concatenate(vectors, axis=0)

        invoke()
        times: list[float] = []
        peak_rss = rss_mb()
        vectors = None
        for _ in range(repeat):
            gc.collect()
            started = time.perf_counter()
            vectors = invoke()
            times.append((time.perf_counter() - started) * 1000)
            peak_rss = max(peak_rss, rss_mb())
        mean_seconds = statistics.mean(times) / 1000
        rows.append({
            "batch_size": batch_size,
            "items": len(items),
            "embedding_dim": int(vectors.shape[1]),
            "output_shape": list(vectors.shape),
            "time_ms": stats(times),
            "items_per_sec": round(len(items) / mean_seconds, 3),
            "tokens": total_tokens,
            "tokens_per_sec": round(total_tokens / mean_seconds, 3) if total_tokens else None,
            "rss_peak_mb": round(peak_rss, 3),
            "batch_semantics": "true text batch; one embed call per chunk",
        })
    return {"kind": "text", "items": len(items), "rows": rows}


def benchmark_paired_multimodal(
    pipeline: Any,
    kind: str,
    items: list[dict[str, Any]],
    repeat: int,
) -> dict[str, Any]:
    """Benchmark paired page image/fused inputs without cross-attaching images."""
    rows: list[dict[str, Any]] = []
    for batch_size in BATCHES:
        def invoke() -> np.ndarray:
            vectors: list[np.ndarray] = []
            for start in range(0, len(items), batch_size):
                for item in items[start:start + batch_size]:
                    image = image_tensor(item["image"])
                    vector = image_call(pipeline, image) if kind == "page_image_150" else fused_call(pipeline, item["text"], image)
                    vectors.append(vector)
            return np.concatenate(vectors, axis=0)

        invoke()
        times: list[float] = []
        peak_rss = rss_mb()
        vectors = None
        for _ in range(repeat):
            gc.collect()
            started = time.perf_counter()
            vectors = invoke()
            times.append((time.perf_counter() - started) * 1000)
            peak_rss = max(peak_rss, rss_mb())
        mean_seconds = statistics.mean(times) / 1000
        rows.append({
            "batch_size": batch_size,
            "items": len(items),
            "embedding_dim": int(vectors.shape[1]),
            "output_shape": list(vectors.shape),
            "time_ms": stats(times),
            "items_per_sec": round(len(items) / mean_seconds, 3),
            "tokens": sum(item.get("tokens", 0) for item in items),
            "tokens_per_sec": None,
            "rss_peak_mb": round(peak_rss, 3),
            "batch_semantics": "paired per-item image/fused calls grouped by batch_size; Qwen3 API has no text_i/image_i pair batch input",
        })
    return {"kind": kind, "items": len(items), "rows": rows}


def cosine_matrix(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    query_norm = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    document_norm = documents / np.maximum(np.linalg.norm(documents, axis=1, keepdims=True), 1e-12)
    return query_norm @ document_norm.T


def quality_metrics(pipeline: Any, inputs: dict[str, Any], embeddings: dict[str, np.ndarray]) -> dict[str, Any]:
    queries = [item["text"] for item in QUALITY_QUERIES]
    query_vectors = text_call(pipeline, queries, QUERY_PROMPT)
    output: dict[str, Any] = {
        "queries": QUALITY_QUERIES,
        "query_prompt": QUERY_PROMPT,
        "document_prompt": DOCUMENT_PROMPT,
    }
    for kind in ("page_text", "text_chunk_256", "page_image_150", "page_fused_150"):
        items = inputs[kind]
        scores = cosine_matrix(query_vectors, embeddings[kind])
        hits = []
        reciprocal: list[float] = []
        for query_index, query in enumerate(QUALITY_QUERIES):
            relevant = set(query["pages"])
            order = np.argsort(-scores[query_index])
            ranked_pages = [items[index]["page"] for index in order]
            rank = next((index + 1 for index, page in enumerate(ranked_pages) if page in relevant), len(ranked_pages) + 1)
            reciprocal.append(1.0 / rank if rank <= len(ranked_pages) else 0.0)
            hits.append({
                "id": query["id"],
                "top_pages": ranked_pages[:5],
                "top_scores": [round(float(scores[query_index][index]), 6) for index in order[:5]],
                "first_relevant_rank": rank,
            })
        output[kind] = {
            "recall_at_1": round(float(np.mean([hit["first_relevant_rank"] <= 1 for hit in hits])), 6),
            "recall_at_3": round(float(np.mean([hit["first_relevant_rank"] <= 3 for hit in hits])), 6),
            "recall_at_5": round(float(np.mean([hit["first_relevant_rank"] <= 5 for hit in hits])), 6),
            "mrr": round(float(np.mean(reciprocal)), 6),
            "per_query": hits,
        }
    cross: dict[str, Any] = {}
    for left, right in (("page_text", "page_image_150"), ("page_text", "page_fused_150"), ("page_image_150", "page_fused_150")):
        diagonal = np.diag(cosine_matrix(embeddings[left], embeddings[right]))
        cross[f"{left}_vs_{right}"] = {
            "mean": round(float(diagonal.mean()), 6),
            "min": round(float(diagonal.min()), 6),
            "max": round(float(diagonal.max()), 6),
        }
    output["cross_representation_cosine"] = cross
    return output


def run_precision(precision: str, inputs: dict[str, Any], page_images: list[Path], repeat: int) -> dict[str, Any]:
    model_dir = MODELS_ROOT / f"Qwen3-VL-Embedding-2B-{precision}"
    started = time.perf_counter()
    pipeline = openvino_genai.EmbeddingPipeline(str(model_dir), "GPU")
    load_ms = (time.perf_counter() - started) * 1000
    embeddings: dict[str, np.ndarray] = {}
    performance: dict[str, Any] = {}
    for kind in ("page_text", "text_chunk_256"):
        performance[kind] = benchmark_text(pipeline, inputs[kind], repeat, DOCUMENT_PROMPT)
        batch_size = BATCHES[-1]
        texts = [item["text"] for item in inputs[kind]]
        embeddings[kind] = np.concatenate(
            [text_call(pipeline, texts[start:start + batch_size], DOCUMENT_PROMPT)
             for start in range(0, len(texts), batch_size)], axis=0
        )
    for kind in ("page_image_150", "page_fused_150"):
        performance[kind] = benchmark_paired_multimodal(pipeline, kind, inputs[kind], repeat)
        vectors = []
        for item in inputs[kind]:
            image = image_tensor(item["image"])
            vectors.append(image_call(pipeline, image) if kind == "page_image_150" else fused_call(pipeline, item["text"], image))
        embeddings[kind] = np.concatenate(vectors, axis=0)
    return {
        "model_dir": str(model_dir),
        "precision": precision,
        "device": "GPU",
        "model_load_ms": round(load_ms, 3),
        "embedding_dim": 2048,
        "performance": performance,
        "quality": quality_metrics(pipeline, inputs, embeddings),
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-VL-Embedding-2B Document Embedding Benchmark",
        "",
        f"Document: `{result['pdf']['path']}` ({result['pdf']['pages']} pages)",
        "",
        "All model measurements use patched OpenVINO GenAI 2026.3, GPU, native 2048-dimensional output.",
        "",
        "## Performance",
        "",
        "Text representations use true text batches. Image/fused representations preserve page pairs with per-item calls because the Qwen3 multimodal API does not accept paired `(text_i, image_i)` batches.",
        "",
    ]
    for precision, data in result["models"].items():
        lines += [f"### {precision} (model load {data['model_load_ms']:.1f} ms)", "", "| Representation | Batch | Mean ms | P50 ms | Items/s | Tokens/s | Dim |", "|---|---:|---:|---:|---:|---:|---:|"]
        for kind, benchmark in data["performance"].items():
            for row in benchmark["rows"]:
                lines.append(
                    f"| {kind} | {row['batch_size']} | {row['time_ms']['mean']:.1f} | {row['time_ms']['p50']:.1f} | {row['items_per_sec']:.2f} | {row['tokens_per_sec'] if row['tokens_per_sec'] is not None else '-'} | {row['embedding_dim']} |"
                )
        lines.append("")
    lines += ["## Quality", "", "Recall and MRR use the manually labeled page sets in `QUALITY_QUERIES`; scores are retrieval diagnostics, not a substitute for a production relevance benchmark.", ""]
    for precision, data in result["models"].items():
        lines += [f"### {precision}", "", "| Representation | Recall@1 | Recall@3 | Recall@5 | MRR |", "|---|---:|---:|---:|---:|"]
        for kind in ("page_text", "text_chunk_256", "page_image_150", "page_fused_150"):
            quality = data["quality"][kind]
            lines.append(f"| {kind} | {quality['recall_at_1']:.3f} | {quality['recall_at_3']:.3f} | {quality['recall_at_5']:.3f} | {quality['mrr']:.3f} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    page_texts, pdf_stats = extract_pdf()
    page_images, render_stats = render_pages()
    tokenizer = openvino_genai.Tokenizer(str(MODELS_ROOT / "Qwen3-VL-Embedding-2B-INT8"))
    inputs = build_inputs(page_texts, page_images, tokenizer)
    result: dict[str, Any] = {
        "pdf": pdf_stats,
        "render": render_stats,
        "settings": {
            "device": "GPU",
            "precisions": ["INT8", "INT4"],
            "batches": list(BATCHES),
            "repeats": REPEATS,
            "chunk_size_tokens": CHUNK_SIZE,
            "chunk_overlap_tokens": CHUNK_OVERLAP,
            "dpi": DPI,
            "document_prompt": DOCUMENT_PROMPT,
            "query_prompt": QUERY_PROMPT,
        },
        "inputs": inputs,
        "models": {},
    }
    for precision in ("INT8", "INT4"):
        print(f"Loading {precision} on GPU")
        result["models"][precision] = run_precision(precision, inputs, page_images, REPEATS)
    (RESULTS / "benchmark.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (RESULTS / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    print(f"wrote {RESULTS / 'benchmark.json'}")
    print(f"wrote {RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
