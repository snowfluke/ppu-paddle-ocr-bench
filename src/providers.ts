import os from "os";

export type ExecutionProvider = string | { name: string; [k: string]: unknown };

/**
 * Pick execution providers best-to-worst for the current platform.
 * ONNX Runtime walks the list and uses the first provider that can run each
 * node, so keeping "cpu" last guarantees a working fallback.
 *
 * Override with the EP env var, e.g. `EP=cpu` or `EP=cuda,cpu`.
 */
export function autoExecutionProviders(): ExecutionProvider[] {
  const override = process.env.EP?.trim();
  if (override) return override.split(",").map((s) => s.trim()).filter(Boolean);

  switch (process.platform) {
    case "darwin":
      // CoreML targets the Apple GPU / Neural Engine.
      return ["coreml", "cpu"];
    case "win32":
      // DirectML runs on any DX12 GPU (NVIDIA/AMD/Intel, incl. integrated);
      // CUDA is tried first for NVIDIA setups that have the toolkit.
      return ["cuda", "dml", "cpu"];
    default:
      // Linux: CUDA for NVIDIA, otherwise CPU.
      return ["cuda", "cpu"];
  }
}

export function describeProviders(providers: ExecutionProvider[]): string {
  return providers
    .map((p) => (typeof p === "string" ? p : p.name))
    .join(" → ");
}

export function platformLabel(): string {
  return `${process.platform}/${process.arch} · ${os.cpus()[0]?.model ?? "unknown CPU"} · ${os.cpus().length} threads`;
}
