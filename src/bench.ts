import { readdirSync } from "fs";
import path from "path";
import { performance } from "perf_hooks";
import { PaddleOcrService } from "ppu-paddle-ocr";
import type { PaddleOptions, RecognitionStrategy } from "ppu-paddle-ocr";
import {
  autoExecutionProviders,
  describeProviders,
  platformLabel,
  type ExecutionProvider,
} from "./providers.ts";

const IMAGE_DIR = process.env.IMAGE_DIR ?? "./images";
const NUM_LOOPS = Number(process.env.NUM_LOOPS ?? 7);
const STRATEGY = (process.env.STRATEGY as RecognitionStrategy) ?? "cross-line";
const ENGINE = (process.env.ENGINE as "opencv" | "canvas-native") ?? "opencv";
// Optional CPU intra-op thread cap. On Apple Silicon, capping to the
// performance-core count (e.g. THREADS=4 on M1) beats ORT's default, which
// also schedules the slower efficiency cores. Unset = let ORT decide.
const THREADS = process.env.THREADS ? Number(process.env.THREADS) : undefined;

const IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"];

function loadImagePaths(dir: string): string[] {
  return readdirSync(dir)
    .filter((f) => IMAGE_EXTS.includes(path.extname(f).toLowerCase()))
    .sort()
    .map((f) => path.join(dir, f));
}

/** Build + initialize a service, falling back to CPU if a GPU provider fails. */
async function createService(
  providers: ExecutionProvider[]
): Promise<{ service: PaddleOcrService; used: ExecutionProvider[] }> {
  // `strategy` is passed per recognize() call below — the published
  // PaddleOptions.recognition type wrongly requires an internal
  // `charactersDictionary` field, so we keep it out of the constructor.
  const base: PaddleOptions = {
    debugging: { debug: false, verbose: false },
    processing: { engine: ENGINE },
  };
  const session = (eps: ExecutionProvider[]) => ({
    executionProviders: eps,
    graphOptimizationLevel: "all" as const,
    ...(THREADS ? { intraOpNumThreads: THREADS } : {}),
  });

  try {
    const service = new PaddleOcrService({ ...base, session: session(providers) });
    await service.initialize();
    return { service, used: providers };
  } catch (err) {
    if (providers.length === 1 && providers[0] === "cpu") throw err;
    console.warn(
      `⚠ GPU providers [${describeProviders(providers)}] failed to init, falling back to CPU.`
    );
    console.warn(`  reason: ${(err as Error).message}`);
    const service = new PaddleOcrService({ ...base, session: session(["cpu"]) });
    await service.initialize();
    return { service, used: ["cpu"] };
  }
}

function stats(xs: number[]) {
  const sorted = [...xs].sort((a, b) => a - b);
  const n = sorted.length;
  const min = sorted[0]!;
  const max = sorted[n - 1]!;
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const median = n % 2 ? sorted[(n - 1) / 2]! : (sorted[n / 2 - 1]! + sorted[n / 2]!) / 2;
  const stddev = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  return { min, max, mean, median, stddev };
}

async function main() {
  const requested = autoExecutionProviders();

  console.log("=== ppu-paddle-ocr benchmark ===");
  console.log(`platform   : ${platformLabel()}`);
  console.log(`requested  : ${describeProviders(requested)}`);
  console.log(`engine     : ${ENGINE}`);
  console.log(`strategy   : ${STRATEGY}`);
  console.log(`threads    : ${THREADS ?? "auto (ORT default)"}`);
  console.log(`loops      : ${NUM_LOOPS}`);
  console.log(`image dir  : ${IMAGE_DIR}\n`);

  const imagePaths = loadImagePaths(IMAGE_DIR);
  if (imagePaths.length === 0) {
    console.error(`No images (${IMAGE_EXTS.join(", ")}) found in ${IMAGE_DIR}.`);
    process.exit(1);
  }
  console.log(`Loaded ${imagePaths.length} image(s).`);

  const buffers = await Promise.all(
    imagePaths.map((p) => Bun.file(p).arrayBuffer())
  );

  const { service, used } = await createService(requested);
  console.log(`active EP  : ${describeProviders(used)}\n`);

  // Warmup: first inference pays the ORT graph-build cost. noCache so it does
  // not seed the result cache and turn timed loops into cache hits.
  process.stdout.write("Warming up… ");
  for (const buf of buffers) await service.recognize(buf, { strategy: STRATEGY, noCache: true });
  console.log("done.\n");

  // --- Sequential: one recognize() per image, in order. ---
  console.log("--- Sequential (recognize per image) ---");
  const seqLoops: number[] = [];
  const seqPerImage: number[] = [];

  for (let i = 0; i < NUM_LOOPS; i++) {
    const loopStart = performance.now();
    for (const buf of buffers) {
      const t = performance.now();
      await service.recognize(buf, { strategy: STRATEGY, noCache: true });
      seqPerImage.push((performance.now() - t) / 1000);
    }
    const dur = (performance.now() - loopStart) / 1000;
    seqLoops.push(dur);
    console.log(`Loop ${i + 1}/${NUM_LOOPS}: ${dur.toFixed(2)} s`);
  }

  // --- Batch: all images handed to batchRecognize() in one call per loop. ---
  console.log("\n--- Batch (batchRecognize all) ---");
  const batchLoops: number[] = [];

  for (let i = 0; i < NUM_LOOPS; i++) {
    const loopStart = performance.now();
    await service.batchRecognize(buffers, { strategy: STRATEGY, noCache: true });
    const dur = (performance.now() - loopStart) / 1000;
    batchLoops.push(dur);
    console.log(`Loop ${i + 1}/${NUM_LOOPS}: ${dur.toFixed(2)} s`);
  }

  const seqLoop = stats(seqLoops);
  const seqImg = stats(seqPerImage);
  const batchLoop = stats(batchLoops);
  const n = imagePaths.length;

  console.log(`\n=== Sequential per-loop (${n} images each) ===`);
  console.log(`Median : ${seqLoop.median.toFixed(2)} s  (±${seqLoop.stddev.toFixed(2)} s)`);
  console.log(`Min/Max: ${seqLoop.min.toFixed(2)} / ${seqLoop.max.toFixed(2)} s`);
  console.log(`Thrupt : ${(n / seqLoop.median).toFixed(2)} img/s`);

  console.log(`\n=== Sequential per-image ===`);
  console.log(`Median : ${(seqImg.median * 1000).toFixed(0)} ms`);
  console.log(`Min/Max: ${(seqImg.min * 1000).toFixed(0)} / ${(seqImg.max * 1000).toFixed(0)} ms`);

  console.log(`\n=== Batch per-loop (${n} images each) ===`);
  console.log(`Median : ${batchLoop.median.toFixed(2)} s  (±${batchLoop.stddev.toFixed(2)} s)`);
  console.log(`Min/Max: ${batchLoop.min.toFixed(2)} / ${batchLoop.max.toFixed(2)} s`);
  console.log(`Thrupt : ${(n / batchLoop.median).toFixed(2)} img/s`);

  const speedup = seqLoop.median / batchLoop.median;
  console.log(`\nBatch vs sequential: ${speedup.toFixed(2)}× (median per-loop)`);

  await service.destroy();
}

main().catch((err) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
