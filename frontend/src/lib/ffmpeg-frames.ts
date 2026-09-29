import { FFmpeg, FFFSType } from "@ffmpeg/ffmpeg";
import { toBlobURL } from "@ffmpeg/util";

// Декодер кадров на ffmpeg.wasm для форматов, которые браузер не умеет показывать сам
// (HEVC без аппаратной поддержки, ProRes, MPEG-4 Part 2, AVI и т. п.).

type LoadProgress = (ratio: number | null) => void;

let assetsPromise: Promise<{ coreURL: string; wasmURL: string }> | null = null;
const progressListeners = new Set<LoadProgress>();

function loadAssets() {
  if (!assetsPromise) {
    assetsPromise = Promise.all([
      toBlobURL("/ffmpeg/ffmpeg-core.js", "text/javascript"),
      toBlobURL("/ffmpeg/ffmpeg-core.wasm", "application/wasm", true, ({ received, total }) => {
        const ratio = total > 0 ? Math.min(1, received / total) : null;
        progressListeners.forEach((listener) => listener(ratio));
      }),
    ])
      .then(([coreURL, wasmURL]) => ({ coreURL, wasmURL }))
      .catch((cause) => {
        assetsPromise = null;
        throw cause;
      });
  }
  return assetsPromise;
}

export type VideoInfo = { duration: number; fps: number; width: number; height: number; codec: string };

export type FrameOptions = {
  /** Максимальная длина большей стороны результата. */
  maxSide: number;
  /** Быстрый режим: ближайший ключевой кадр до указанного времени, без точного декодирования. */
  keyframe?: boolean;
  timeoutMs?: number;
};

let instanceCounter = 0;

export class VideoFrameDecoder {
  private ffmpeg: FFmpeg | null = null;
  private starting: Promise<FFmpeg> | null = null;
  private logs: string[] = [];
  private chain: Promise<unknown> = Promise.resolve();
  private disposed = false;
  private outputCounter = 0;
  private readonly mountDir = `/input-${++instanceCounter}`;

  constructor(private readonly file: File) {}

  private get inputPath() {
    return `${this.mountDir}/${this.file.name}`;
  }

  private async start(): Promise<FFmpeg> {
    if (this.ffmpeg) return this.ffmpeg;
    if (!this.starting) {
      this.starting = (async () => {
        const { coreURL, wasmURL } = await loadAssets();
        if (this.disposed) throw new Error("Декодер закрыт.");
        const ffmpeg = new FFmpeg();
        ffmpeg.on("log", ({ message }) => {
          this.logs.push(message);
          if (this.logs.length > 300) this.logs.splice(0, this.logs.length - 300);
        });
        await ffmpeg.load({ coreURL, wasmURL });
        await ffmpeg.createDir(this.mountDir);
        await ffmpeg.mount(FFFSType.WORKERFS, { files: [this.file] }, this.mountDir);
        if (this.disposed) {
          ffmpeg.terminate();
          throw new Error("Декодер закрыт.");
        }
        this.ffmpeg = ffmpeg;
        return ffmpeg;
      })().finally(() => {
        this.starting = null;
      });
    }
    return this.starting;
  }

  /** Полностью перезапускает wasm-экземпляр: после Aborted() или таймаута он непригоден. */
  private restart() {
    this.ffmpeg?.terminate();
    this.ffmpeg = null;
  }

  /** Все операции ffmpeg выполняются строго по очереди. */
  private serial<T>(task: () => Promise<T>): Promise<T> {
    const next = this.chain.then(task, task);
    this.chain = next.catch(() => undefined);
    return next;
  }

  async load(onProgress?: LoadProgress) {
    if (onProgress) progressListeners.add(onProgress);
    try {
      await this.serial(() => this.start());
    } finally {
      if (onProgress) progressListeners.delete(onProgress);
    }
  }

  private lastError(fallback: string) {
    const meaningful = this.logs
      .filter((line) => !/^\s*(Aborted\(\)|frame=|size=|video:|Input #|Output #|Stream mapping|Press \[q\])/.test(line))
      .slice(-2)
      .join(" ")
      .trim();
    return meaningful || fallback;
  }

  /** Читает параметры видео и заодно проверяет, что ffmpeg способен декодировать первый кадр. */
  probe(): Promise<VideoInfo> {
    return this.serial(async () => {
      for (let attempt = 0; attempt < 2; attempt += 1) {
        const ffmpeg = await this.start();
        this.logs = [];
        let code = -1;
        try {
          code = await ffmpeg.exec(["-hide_banner", "-i", this.inputPath, "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"], 60_000);
        } catch {
          code = -1;
        }
        const text = this.logs.join("\n");
        const durationMatch = text.match(/Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/);
        const streamLine = text.split("\n").find((line) => /Stream #\d+:\d+.*Video:/.test(line));
        if (!streamLine) {
          if (code !== 0 && attempt === 0) {
            this.restart();
            continue;
          }
          throw new Error(text.includes("Invalid data") ? "Файл повреждён или это не видео." : "В файле не найдена видеодорожка.");
        }
        const codec = streamLine.match(/Video:\s*([\w-]+)/)?.[1] ?? "video";
        const size = streamLine.match(/\b(\d{2,5})x(\d{2,5})\b/);
        const fps = Number(streamLine.match(/([\d.]+)\s*fps/)?.[1] ?? streamLine.match(/([\d.]+)\s*tbr/)?.[1] ?? 30);
        const duration = durationMatch ? Number(durationMatch[1]) * 3600 + Number(durationMatch[2]) * 60 + Number(durationMatch[3]) : NaN;
        if (code !== 0) this.restart();
        return {
          codec,
          duration,
          fps: Number.isFinite(fps) && fps > 1 && fps < 1000 ? fps : 30,
          width: size ? Number(size[1]) : 0,
          height: size ? Number(size[2]) : 0,
        };
      }
      throw new Error("Не удалось прочитать видео.");
    });
  }

  /**
   * Декодирует один кадр и возвращает PNG.
   * Встроенный в ffmpeg.wasm кодировщик MJPEG падает с «memory access out of bounds» / «null function»,
   * поэтому ffmpeg отдаёт PNG — API поиска принимает его напрямую.
   */
  frame(time: number, options: FrameOptions): Promise<Blob> {
    return this.serial(async () => {
      const max = Math.round(options.maxSide);
      const filters = `scale=w='min(${max},iw)':h='min(${max},ih)':force_original_aspect_ratio=decrease,format=rgb24`;
      const seek = Math.max(0, time).toFixed(4);
      const timeout = options.timeoutMs ?? 90_000;
      const args = [
        "-hide_banner",
        ...(options.keyframe ? ["-skip_frame", "nokey", "-noaccurate_seek"] : []),
        "-ss", seek,
        "-i", this.inputPath,
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        "-frames:v", "1",
        "-vf", filters,
        "-c:v", "png",
        "-compression_level", "1",
        "-f", "image2",
        "-update", "1",
        "-y",
      ];
      let lastError = "Не удалось декодировать кадр.";
      for (let attempt = 0; attempt < 2; attempt += 1) {
        const output = `frame-${++this.outputCounter}.png`;
        const ffmpeg = await this.start();
        this.logs = [];
        let code = -1;
        const startedAt = performance.now();
        try {
          code = await ffmpeg.exec([...args, output], timeout);
        } catch (cause) {
          code = -1;
          this.logs.push(`exec error: ${cause instanceof Error ? cause.message : String(cause)}`);
        }
        const timedOut = code !== 0 && performance.now() - startedAt >= timeout - 50;
        try {
          const data = await ffmpeg.readFile(output);
          await ffmpeg.deleteFile(output).catch(() => undefined);
          if (typeof data !== "string" && data.byteLength > 0) {
            if (code !== 0) this.restart();
            return new Blob([new Uint8Array(data)], { type: "image/png" });
          }
        } catch {
          // Файла нет — кадр не получен, разберёмся ниже.
        }
        console.warn(`[ffmpeg] кадр ${seek}s: код ${code}`, this.logs.slice(-20).join("\n"));
        lastError = timedOut ? "Декодирование заняло слишком много времени." : this.lastError(lastError);
        // Ошибка или таймаут: экземпляр мог упасть в abort(), перезапускаем его и пробуем ещё раз.
        this.restart();
      }
      throw new Error(lastError);
    });
  }

  dispose() {
    this.disposed = true;
    this.restart();
  }
}
