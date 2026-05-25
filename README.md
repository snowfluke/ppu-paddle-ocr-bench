# ppu-paddle-ocr-bench

Standalone benchmark for [`ppu-paddle-ocr`](https://github.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr) on Node/Bun, with **automatic GPU execution-provider detection**.

It measures the *published* library (installed via `bun add`), runs a real OCR
pass per loop (cache disabled), and reports per-loop and per-image timings.

## Quick start

```bash
bun install
bun run warm        # one-time: download models into ~/.cache/ppu-paddle-ocr
# drop your test images into ./images
bun run bench
```

## GPU auto-detection

ONNX Runtime on Node defaults to **CPU only** — it does *not* use a GPU unless
you ask for one. This benchmark picks the best execution provider for the
platform and always keeps `cpu` as a fallback:

| Platform              | Providers tried (in order) |
| --------------------- | -------------------------- |
| macOS (Apple Silicon) | `cpu`                      |
| macOS (Intel)         | `coreml` → `cpu`           |
| Windows               | `cuda` → `dml` → `cpu`     |
| Linux                 | `cuda` → `cpu`             |

- **DirectML (`dml`)** works on any DX12 GPU on Windows (NVIDIA/AMD/Intel,
  including integrated) — this is usually what a "laptop with a VGA" wants.
- **CUDA** needs the NVIDIA CUDA toolkit installed and a CUDA-enabled
  `onnxruntime-node` build; if it can't init, the bench logs a warning and
  falls back to CPU automatically.
- **CoreML is not the default on Apple Silicon.** Measured on an M1, CoreML runs
  these mobile OCR models ~4–5× **slower** than CPU: ORT can place only part of
  the graph on CoreML and splits it into 10–26 partitions, paying CPU↔ANE
  conversion at every boundary. FP32 on the CPU (NEON) wins. Try it with
  `EP=coreml bun run bench`. Intel Macs still default to CoreML.

The startup banner prints both the **requested** and the **active** provider so
you can confirm what actually ran.

> Note: a GPU provider only helps if `onnxruntime-node` was built with it and
> the system libraries are present, and only pays off for a real discrete GPU
> (NVIDIA CUDA, DX12 DirectML). FP32-on-CPU is the baseline everywhere and the
> winner on Apple Silicon. Do **not** use INT8 models on Apple Silicon — FP32
> NEON is faster there too.

## Configuration (env vars)

| Var         | Default       | Meaning                                            |
| ----------- | ------------- | -------------------------------------------------- |
| `EP`        | auto          | Override providers, e.g. `EP=cpu` or `EP=dml,cpu`  |
| `IMAGE_DIR` | `./images`    | Folder of test images (`.png/.jpg/.jpeg/.webp`)    |
| `NUM_LOOPS` | `7`           | Number of timed passes over all images             |
| `STRATEGY`  | `cross-line`  | `per-box` \| `per-line` \| `cross-line`            |
| `ENGINE`    | `opencv`      | `opencv` \| `canvas-native`                        |
| `THREADS`   | auto          | CPU `intraOpNumThreads` cap (see below)            |

Examples:

```bash
EP=cpu bun run bench                 # force CPU baseline
EP=coreml bun run bench              # try Apple CoreML (usually slower)
STRATEGY=cross-line bun run bench    # throughput-oriented
THREADS=4 bun run bench              # cap intra-op threads
IMAGE_DIR=../receipts bun run bench  # point at another folder
```

### Tuning `THREADS` on Apple Silicon

ORT's default intra-op thread count includes the efficiency cores, which drag
down a latency-bound workload. On an M1 (4 performance + 4 efficiency cores),
capping to the performance-core count is fastest:

| `THREADS` | ms / image (M1, CPU) |
| --------- | -------------------- |
| 8         | ~674                 |
| auto      | ~347                 |
| **4**     | **~312**             |
| 2         | ~509                 |

Match this to your chip's performance-core count (e.g. `THREADS=4` for an M1,
higher for M-series Pro/Max).

## Why cache is disabled

`recognize()` caches results in-memory keyed by image bytes. Reusing the same
buffers across loops would make every loop after the first a cache hit and the
average meaningless. This bench passes `{ noCache: true }` so each loop is a
real inference pass, and runs an excluded warmup first to absorb ONNX Runtime's
one-time graph-build cost.
