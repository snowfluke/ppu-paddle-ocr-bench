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

| Platform | Providers tried (in order) |
| -------- | -------------------------- |
| macOS    | `coreml` → `cpu`           |
| Windows  | `cuda` → `dml` → `cpu`     |
| Linux    | `cuda` → `cpu`             |

- **DirectML (`dml`)** works on any DX12 GPU on Windows (NVIDIA/AMD/Intel,
  including integrated) — this is usually what a "laptop with a VGA" wants.
- **CUDA** needs the NVIDIA CUDA toolkit installed and a CUDA-enabled
  `onnxruntime-node` build; if it can't init, the bench logs a warning and
  falls back to CPU automatically.
- **CoreML** targets the Apple GPU / Neural Engine on macOS.

The startup banner prints both the **requested** and the **active** provider so
you can confirm what actually ran.

> Note: a GPU provider only helps if `onnxruntime-node` was built with it and
> the system libraries are present. FP32-on-CPU is the baseline everywhere.
> On Apple Silicon, do **not** use INT8 models — FP32 NEON is faster there.

## Configuration (env vars)

| Var         | Default       | Meaning                                            |
| ----------- | ------------- | -------------------------------------------------- |
| `EP`        | auto          | Override providers, e.g. `EP=cpu` or `EP=dml,cpu`  |
| `IMAGE_DIR` | `./images`    | Folder of test images (`.png/.jpg/.jpeg/.webp`)    |
| `NUM_LOOPS` | `7`           | Number of timed passes over all images             |
| `STRATEGY`  | `per-line`    | `per-box` \| `per-line` \| `cross-line`            |
| `ENGINE`    | `opencv`      | `opencv` \| `canvas-native`                        |

Examples:

```bash
EP=cpu bun run bench                 # force CPU baseline
STRATEGY=cross-line bun run bench    # throughput-oriented
IMAGE_DIR=../receipts bun run bench  # point at another folder
```

## Why cache is disabled

`recognize()` caches results in-memory keyed by image bytes. Reusing the same
buffers across loops would make every loop after the first a cache hit and the
average meaningless. This bench passes `{ noCache: true }` so each loop is a
real inference pass, and runs an excluded warmup first to absorb ONNX Runtime's
one-time graph-build cost.
