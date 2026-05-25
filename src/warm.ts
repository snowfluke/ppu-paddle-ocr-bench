import { PaddleOcrService } from "ppu-paddle-ocr";

// Pre-download the default models into ~/.cache/ppu-paddle-ocr so the first
// timed run never measures network latency.
await PaddleOcrService.downloadModels({ verbose: true });
console.log("Models cached.");
