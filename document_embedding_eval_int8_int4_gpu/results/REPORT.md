# Qwen3-VL-Embedding-2B Document Embedding Benchmark

Document: `/home/chaofan/model_benchmark/Qwen3-VL-Embedding-2B-Running/060_OT_Downlook_PR_Error.pdf` (14 pages)

All model measurements use patched OpenVINO GenAI 2026.3, GPU, native 2048-dimensional output.

## Performance

Text representations use true text batches. Image/fused representations preserve page pairs with per-item calls because the Qwen3 multimodal API does not accept paired `(text_i, image_i)` batches.

### INT8 (model load 6718.1 ms)

| Representation | Batch | Mean ms | P50 ms | Items/s | Tokens/s | Dim |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 854.7 | 855.1 | 16.38 | 1245.988 | 2048 |
| page_text | 4 | 539.7 | 534.3 | 25.94 | 1973.316 | 2048 |
| page_text | 8 | 551.0 | 537.3 | 25.41 | 1932.983 | 2048 |
| page_text | 14 | 632.1 | 621.1 | 22.15 | 1684.732 | 2048 |
| text_chunk_256 | 1 | 856.7 | 856.3 | 16.34 | 1243.201 | 2048 |
| text_chunk_256 | 4 | 525.7 | 525.6 | 26.63 | 2025.893 | 2048 |
| text_chunk_256 | 8 | 533.8 | 533.0 | 26.23 | 1995.039 | 2048 |
| text_chunk_256 | 14 | 620.0 | 619.6 | 22.58 | 1717.754 | 2048 |
| page_image_150 | 1 | 50655.7 | 50733.9 | 0.28 | - | 2048 |
| page_image_150 | 4 | 50437.5 | 50438.3 | 0.28 | - | 2048 |
| page_image_150 | 8 | 50678.6 | 50700.8 | 0.28 | - | 2048 |
| page_image_150 | 14 | 50745.9 | 50736.2 | 0.28 | - | 2048 |
| page_fused_150 | 1 | 51366.3 | 51365.3 | 0.27 | - | 2048 |
| page_fused_150 | 4 | 51233.5 | 51214.8 | 0.27 | - | 2048 |
| page_fused_150 | 8 | 51178.7 | 51174.1 | 0.27 | - | 2048 |
| page_fused_150 | 14 | 51187.7 | 51178.6 | 0.27 | - | 2048 |

### INT4 (model load 5509.9 ms)

| Representation | Batch | Mean ms | P50 ms | Items/s | Tokens/s | Dim |
|---|---:|---:|---:|---:|---:|---:|
| page_text | 1 | 777.0 | 756.0 | 18.02 | 1370.71 | 2048 |
| page_text | 4 | 514.4 | 500.5 | 27.21 | 2070.263 | 2048 |
| page_text | 8 | 528.5 | 515.9 | 26.49 | 2015.143 | 2048 |
| page_text | 14 | 627.5 | 614.2 | 22.31 | 1697.341 | 2048 |
| text_chunk_256 | 1 | 739.5 | 740.0 | 18.93 | 1440.159 | 2048 |
| text_chunk_256 | 4 | 499.9 | 500.0 | 28.00 | 2130.35 | 2048 |
| text_chunk_256 | 8 | 516.1 | 516.3 | 27.12 | 2063.364 | 2048 |
| text_chunk_256 | 14 | 612.6 | 612.5 | 22.85 | 1738.483 | 2048 |
| page_image_150 | 1 | 50624.1 | 50596.9 | 0.28 | - | 2048 |
| page_image_150 | 4 | 50463.6 | 50444.1 | 0.28 | - | 2048 |
| page_image_150 | 8 | 50500.4 | 50502.6 | 0.28 | - | 2048 |
| page_image_150 | 14 | 50459.5 | 50467.5 | 0.28 | - | 2048 |
| page_fused_150 | 1 | 51092.1 | 51113.2 | 0.27 | - | 2048 |
| page_fused_150 | 4 | 51105.7 | 51094.1 | 0.27 | - | 2048 |
| page_fused_150 | 8 | 51132.8 | 51124.1 | 0.27 | - | 2048 |
| page_fused_150 | 14 | 51131.7 | 51144.7 | 0.27 | - | 2048 |

## Quality

Recall and MRR use the manually labeled page sets in `QUALITY_QUERIES`; scores are retrieval diagnostics, not a substitute for a production relevance benchmark.

### INT8

| Representation | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| page_text | 0.125 | 0.625 | 0.625 | 0.385 |
| text_chunk_256 | 0.125 | 0.625 | 0.625 | 0.385 |
| page_image_150 | 0.250 | 0.750 | 0.875 | 0.507 |
| page_fused_150 | 0.125 | 0.625 | 0.625 | 0.406 |

### INT4

| Representation | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| page_text | 0.125 | 0.500 | 0.750 | 0.380 |
| text_chunk_256 | 0.125 | 0.500 | 0.750 | 0.380 |
| page_image_150 | 0.250 | 0.750 | 0.750 | 0.497 |
| page_fused_150 | 0.250 | 0.500 | 0.750 | 0.461 |
