#!/usr/bin/env python3
"""Benchmark Qwen3-VL-Embedding-2B through the patched OpenVINO GenAI API."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

import openvino as ov
import openvino_genai

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_ROOT = ROOT.parent / "benchmark" / "models"
DEFAULT_INSTALL = ROOT.parent / "benchmark" / "openvino" / "install"
DEFAULT_RESULTS = ROOT / "benchmark_results"
TEXT_LENGTHS = [32, 64, 128, 256, 512, 1024]
IMAGE_SIZES = [(64, 64), (224, 224), (448, 448), (672, 672), (960, 480)]
PRECISIONS = ["FP32", "INT8", "INT4"]
DEVICES = ["CPU", "GPU"]
MODES = ["text", "image", "fused"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
TEXT = "What causes the halo effect on the edge bump?"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def make_assets(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Path] = {}
    for width, height in IMAGE_SIZES:
        path = directory / f"gradient_{width}x{height}.png"
        if not path.exists():
            x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
            y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
            pixels = np.stack(
                [np.broadcast_to(x, (height, width)),
                 np.broadcast_to(y, (height, width)),
                 (x + y) / 2],
                axis=-1,
            ).astype(np.uint8)
            Image.fromarray(pixels, mode="RGB").save(path)
        assets[f"{width}x{height}"] = path
    return assets


def make_text(tokenizer: openvino_genai.Tokenizer, target: int) -> tuple[str, int]:
    """Create ASCII filler whose GenAI tokenizer length is exactly target."""
    for words in range(max(1, target - 2), target * 3 + 16):
        text = " ".join(["a"] * words)
        length = int(tokenizer.encode(text).input_ids.shape[1])
        if length == target:
            return text, length
        if length > target:
            break
    raise RuntimeError(f"Could not construct a text workload with {target} tokens")


def load_image_tensor(path: Path) -> ov.Tensor:
    image = Image.open(path).convert("RGB")
    return ov.Tensor(np.asarray(image, dtype=np.uint8))


def extract_vector(result: Any) -> np.ndarray:
    tensor = result.embeddings
    return np.asarray(tensor.data, dtype=np.float32).reshape(tuple(tensor.shape))


def timed_embedding(
    invoke: Callable[[], Any],
    repeat: int,
    warmup: int,
    batch_size: int = 1,
) -> dict[str, Any]:
    for _ in range(warmup):
        extract_vector(invoke())
    samples: list[float] = []
    final_shape: list[int] | None = None
    for _ in range(repeat):
        started = time.perf_counter()
        projected = extract_vector(invoke())
        samples.append((time.perf_counter() - started) * 1000)
        final_shape = list(projected.shape)
    return {
        "output_shape": final_shape,
        "output_dimension": 2048,
        "batch_size": batch_size,
        "mean_ms": round(statistics.mean(samples), 3),
        "mean_ms_per_embedding": round(statistics.mean(samples) / batch_size, 3),
        "stdev_ms": round(statistics.stdev(samples), 3) if len(samples) > 1 else 0.0,
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
        "repeat": repeat,
        "warmup": warmup,
        "timing_scope": "warm native 2048-dimensional embed; model construction excluded",
    }


def workload_call(pipeline: Any, mode: str, text: str, image: ov.Tensor) -> Callable[[], Any]:
    if mode == "text":
        return lambda: pipeline.embed(text)
    if mode == "image":
        return lambda: pipeline.embed(images=[image])
    if mode == "fused":
        return lambda: pipeline.embed(text, images=[image])
    raise ValueError(f"Unknown mode: {mode}")


def make_environment(core: ov.Core) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in ("openvino", "openvino-genai", "numpy", "Pillow")
        },
        "available_devices": list(core.available_devices),
    }


def run_baseline(
    models_root: Path,
    image_path: Path,
    devices: list[str],
    precisions: list[str],
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    core = ov.Core()
    image = load_image_tensor(image_path)
    tokenizer = openvino_genai.Tokenizer(str(models_root / "Qwen3-VL-Embedding-2B-INT8"))
    text, token_count = make_text(tokenizer, 32)
    result: dict[str, Any] = {
        "suite": "baseline",
        "workload": {
            "text": text,
            "token_count": token_count,
            "image": str(image_path),
            "image_size_wh": list(Image.open(image_path).size),
            "image_shape": [Image.open(image_path).size[1], Image.open(image_path).size[0], 3],
            "image_dtype": "uint8 RGB",
            "output_shape": [1, 2048],
        },
        "results": {},
    }
    for device in devices:
        if device not in core.available_devices:
            result["results"][device] = {"status": "unavailable"}
            continue
        for precision in precisions:
            model_dir = models_root / f"Qwen3-VL-Embedding-2B-{precision}"
            started = time.perf_counter()
            pipeline = openvino_genai.EmbeddingPipeline(str(model_dir), device)
            construct_ms = (time.perf_counter() - started) * 1000
            cell: dict[str, Any] = {"construct_ms": round(construct_ms, 3), "modes": {}}
            for mode in MODES:
                call = workload_call(pipeline, mode, text, image)
                cell["modes"][mode] = timed_embedding(call, repeat, warmup)
            result["results"].setdefault(device, {})[precision] = cell
            print(f"baseline {device} {precision} construct_ms={construct_ms:.1f}")
    return result


def run_batch(
    models_root: Path,
    image_path: Path,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    core = ov.Core()
    result: dict[str, Any] = {
        "suite": "batch",
        "device": "GPU",
        "precisions": ["INT8", "INT4"],
        "batch_sizes": BATCH_SIZES,
        "workload": {
            "text": TEXT,
            "token_count": 32,
            "image": str(image_path),
            "image_size_wh": list(Image.open(image_path).size),
            "image_shape": [Image.open(image_path).size[1], Image.open(image_path).size[0], 3],
            "image_dtype": "uint8 RGB",
            "semantics": "one API call returns N rows; image/fused use one shared image and N text rows",
        },
        "results": {},
    }
    if "GPU" not in core.available_devices:
        result["status"] = "GPU unavailable"
        return result

    image = load_image_tensor(image_path)
    for precision in ("INT8", "INT4"):
        model_dir = models_root / f"Qwen3-VL-Embedding-2B-{precision}"
        started = time.perf_counter()
        pipeline = openvino_genai.EmbeddingPipeline(str(model_dir), "GPU")
        precision_result: dict[str, Any] = {
            "construct_ms": round((time.perf_counter() - started) * 1000, 3),
            "cells": {},
        }
        result["results"][precision] = precision_result
        for batch_size in BATCH_SIZES:
            texts = [TEXT] * batch_size
            calls = {
                "text": lambda texts=texts: pipeline.embed(texts),
                "image": lambda batch_size=batch_size: pipeline.embed([""] * batch_size, images=[image]),
                "fused": lambda texts=texts: pipeline.embed(texts, images=[image]),
            }
            precision_result["cells"][str(batch_size)] = {
                mode: timed_embedding(call, repeat, warmup, batch_size)
                for mode, call in calls.items()
            }
            print(f"batch GPU {precision} N={batch_size}")
    return result


def run_detailed(
    models_root: Path,
    assets: dict[str, Path],
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    core = ov.Core()
    tokenizer = openvino_genai.Tokenizer(str(models_root / "Qwen3-VL-Embedding-2B-INT8"))
    texts = {str(target): make_text(tokenizer, target) for target in TEXT_LENGTHS}
    result: dict[str, Any] = {
        "suite": "detailed",
        "devices": ["GPU"],
        "precisions": ["INT8", "INT4"],
        "text_lengths": TEXT_LENGTHS,
        "output_dimension": 2048,
        "image_sizes": [f"{width}x{height}" for width, height in IMAGE_SIZES],
        "workloads": {
            "texts": {
                length: {"text": text, "token_count": token_count}
                for length, (text, token_count) in texts.items()
            },
            "images": {name: {"path": str(path)} for name, path in assets.items()},
        },
        "results": {},
    }
    if "GPU" not in core.available_devices:
        result["status"] = "GPU unavailable"
        return result

    for precision in ("INT8", "INT4"):
        model_dir = models_root / f"Qwen3-VL-Embedding-2B-{precision}"
        started = time.perf_counter()
        pipeline = openvino_genai.EmbeddingPipeline(str(model_dir), "GPU")
        construct_ms = (time.perf_counter() - started) * 1000
        precision_result: dict[str, Any] = {"construct_ms": round(construct_ms, 3), "text": {}, "image": {}, "fused": {}}
        result["results"][precision] = precision_result
        print(f"detailed GPU {precision} construct_ms={construct_ms:.1f}")

        for length, (text, _) in texts.items():
            call = workload_call(pipeline, "text", text, load_image_tensor(next(iter(assets.values()))))
            precision_result["text"][length] = timed_embedding(call, repeat, warmup)

        for image_name, image_path in assets.items():
            image = load_image_tensor(image_path)
            for mode in ("image", "fused"):
                call = workload_call(pipeline, mode, TEXT, image)
                precision_result[mode][image_name] = timed_embedding(call, repeat, warmup)
                print(f"  {precision} {mode} {image_name}")
    return result


def fmt(value: Any) -> str:
    return "-" if value is None else f"{value:.1f}"


def markdown_report(
    baseline: dict[str, Any] | None,
    detailed: dict[str, Any] | None,
    batch: dict[str, Any] | None,
) -> str:
    lines = [
        "# Qwen3-VL-Embedding-2B Benchmark Results",
        "",
        "Generated by `benchmark_qwen3_vl.py`. Times are warm native 2048-dimensional end-to-end calls; model construction is reported separately.",
        "",
    ]
    if baseline is not None:
        workload = baseline["workload"]
        image_path = Path(workload["image"])
        image_size = workload.get("image_size_wh", list(Image.open(image_path).size))
        image_shape = workload.get("image_shape", [image_size[1], image_size[0], 3])
        image_dtype = workload.get("image_dtype", "uint8 RGB")
        lines += [
            "## Baseline input: exact workload",
            "",
            f"- Text: `{workload['text']}`",
            f"- Text tokenizer length: **{workload['token_count']} tokens**",
            f"- Image: `{workload['image']}`",
            f"- Image size: **{image_size[0]}x{image_size[1]}**, shape `{image_shape}`, dtype `{image_dtype}`",
            "- Modes: `text` uses only the text; `image` uses only the image; `fused` uses both in one API call",
            "- Output: native `[1,2048]`; timings are warm end-to-end calls, excluding model construction",
            "",
            "## Baseline: 2048 dimensions",
            "",
            "| Device | Precision | Text | Image | Fused | Construct |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for device, device_result in baseline["results"].items():
            if not isinstance(device_result, dict) or device_result.get("status"):
                continue
            for precision, cell in device_result.items():
                modes = cell["modes"]
                lines.append(
                    f"| {device} | {precision} | {fmt(modes['text']['mean_ms'])} | {fmt(modes['image']['mean_ms'])} | {fmt(modes['fused']['mean_ms'])} | {fmt(cell['construct_ms'])} ms |"
                )
        lines.append("")
    if detailed is not None and detailed.get("results"):
        lines += ["## Detailed text: GPU INT8/INT4, native 2048 dimensions", "", "Each cell is mean milliseconds; output shape is recorded in JSON.", ""]
        for precision, precision_result in detailed["results"].items():
            lines += [f"### {precision}", "", "| Tokens | Native 2048 ms |", "|---:|---:|"]
            for length in detailed["text_lengths"]:
                cell = precision_result["text"][str(length)]
                lines.append(f"| {length} | {fmt(cell['mean_ms'])} |")
            lines.append("")
        lines += ["## Detailed image/fused: GPU INT8/INT4, native 2048 dimensions", "", "Each table reports `image / fused` mean milliseconds.", ""]
        for precision, precision_result in detailed["results"].items():
            lines += [f"### {precision}", "", "| Image | Image ms | Fused ms |", "|---|---:|---:|"]
            for image_name in detailed["image_sizes"]:
                image_cell = precision_result["image"][image_name]
                fused_cell = precision_result["fused"][image_name]
                lines.append(f"| {image_name} | {fmt(image_cell['mean_ms'])} | {fmt(fused_cell['mean_ms'])} |")
            lines.append("")
    if batch is not None and batch.get("results"):
        lines += [
            "## Batch: GPU INT8/INT4",
            "",
            "Each row is one `EmbeddingPipeline.embed(...)` call returning N rows. `image` and `fused` use one shared 448x448 image with N text rows; `ms/call` is total call latency and `ms/embedding` is divided by N.",
            "",
        ]
        for precision, precision_result in batch["results"].items():
            lines += [f"### {precision}", "", "| N | Text ms/call | Text ms/emb | Image ms/call | Image ms/emb | Fused ms/call | Fused ms/emb |", "|---:|---:|---:|---:|---:|---:|---:|"]
            for batch_size in batch["batch_sizes"]:
                cells = precision_result["cells"][str(batch_size)]
                lines.append(
                    f"| {batch_size} | {fmt(cells['text']['mean_ms'])} | {fmt(cells['text']['mean_ms_per_embedding'])} | {fmt(cells['image']['mean_ms'])} | {fmt(cells['image']['mean_ms_per_embedding'])} | {fmt(cells['fused']['mean_ms'])} | {fmt(cells['fused']['mean_ms_per_embedding'])} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("baseline", "detailed", "batch", "all"), default="all")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--image", type=Path, help="baseline image; default is the generated 448x448 workload")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        raise SystemExit("--repeat must be positive and --warmup must be non-negative")
    models_root = args.models_root.expanduser().resolve()
    results_dir = args.results.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    assets = make_assets(results_dir / "assets")
    core = ov.Core()
    environment = make_environment(core)
    baseline = None
    detailed = None
    batch = None
    if args.suite in ("baseline", "all"):
        baseline_image = args.image.expanduser().resolve() if args.image else assets["448x448"]
        baseline = run_baseline(models_root, baseline_image, DEVICES, PRECISIONS, args.repeat, args.warmup)
        baseline["environment"] = environment
        (results_dir / "baseline.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif (results_dir / "baseline.json").is_file():
        baseline = json.loads((results_dir / "baseline.json").read_text(encoding="utf-8"))
    if args.suite in ("detailed", "all"):
        detailed = run_detailed(models_root, assets, args.repeat, args.warmup)
        detailed["environment"] = environment
        (results_dir / "detailed.json").write_text(json.dumps(detailed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif (results_dir / "detailed.json").is_file():
        detailed = json.loads((results_dir / "detailed.json").read_text(encoding="utf-8"))
    if args.suite in ("batch", "all"):
        batch_image = args.image.expanduser().resolve() if args.image else assets["448x448"]
        batch = run_batch(models_root, batch_image, args.repeat, args.warmup)
        batch["environment"] = environment
        (results_dir / "batch.json").write_text(json.dumps(batch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif (results_dir / "batch.json").is_file():
        batch = json.loads((results_dir / "batch.json").read_text(encoding="utf-8"))
    report_path = results_dir / "BENCHMARK_RESULTS.md"
    report_path.write_text(markdown_report(baseline, detailed, batch), encoding="utf-8")
    manifest = {
        "script": str(Path(__file__).resolve()),
        "suite": args.suite,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "baseline_output_dimension": 2048,
        "detailed_output_dimension": 2048,
        "batch_sizes": BATCH_SIZES,
        "note": "OpenVINO GenAI returns native 2048 dimensions. MRL prefix truncation is post-processing and is not a separate latency case.",
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
