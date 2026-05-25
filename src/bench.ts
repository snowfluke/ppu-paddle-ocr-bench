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
const STRATEGY = (process.env.STRATEGY as RecognitionStrategy) ?? "per-line";
const ENGINE = (process.env.ENGINE as "opencv" | "canvas-native") ?? "opencv";

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

  try {
    const service = new PaddleOcrService({
      ...base,
      session: { executionProviders: providers, graphOptimizationLevel: "all" },
    });
    await service.initialize();
    return { service, used: providers };
  } catch (err) {
    if (providers.length === 1 && providers[0] === "cpu") throw err;
    console.warn(
      `⚠ GPU providers [${describeProviders(providers)}] failed to init, falling back to CPU.`
    );
    console.warn(`  reason: ${(err as Error).message}`);
    const service = new PaddleOcrService({
      ...base,
      session: { executionProviders: ["cpu"], graphOptimizationLevel: "all" },
    });
    await service.initialize();
    return { service, used: ["cpu"] };
  }
}

function stats(xs: number[]) {
  const min = Math.min(...xs);
  const max = Math.max(...xs);
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  return { min, max, mean };
}

async function main() {
  const requested = autoExecutionProviders();

  console.log("=== ppu-paddle-ocr benchmark ===");
  console.log(`platform   : ${platformLabel()}`);
  console.log(`requested  : ${describeProviders(requested)}`);
  console.log(`engine     : ${ENGINE}`);
  console.log(`strategy   : ${STRATEGY}`);
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

  const loopDurations: number[] = [];
  const perImage: number[] = [];

  for (let i = 0; i < NUM_LOOPS; i++) {
    const loopStart = performance.now();
    for (const buf of buffers) {
      const t = performance.now();
      await service.recognize(buf, { strategy: STRATEGY, noCache: true });
      perImage.push((performance.now() - t) / 1000);
    }
    const dur = (performance.now() - loopStart) / 1000;
    loopDurations.push(dur);
    console.log(`Loop ${i + 1}/${NUM_LOOPS}: ${dur.toFixed(2)} s`);
  }

  const loop = stats(loopDurations);
  const img = stats(perImage);

  console.log(`\n=== Per-loop (${imagePaths.length} images each) ===`);
  console.log(`Min  : ${loop.min.toFixed(2)} s`);
  console.log(`Max  : ${loop.max.toFixed(2)} s`);
  console.log(`Mean : ${loop.mean.toFixed(2)} s`);

  console.log(`\n=== Per-image ===`);
  console.log(`Min  : ${(img.min * 1000).toFixed(0)} ms`);
  console.log(`Max  : ${(img.max * 1000).toFixed(0)} ms`);
  console.log(`Mean : ${(img.mean * 1000).toFixed(0)} ms`);

  await service.destroy();
}

main().catch((err) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
