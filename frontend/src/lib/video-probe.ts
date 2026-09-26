// Лёгкий разбор контейнеров MP4/MOV без чтения файла целиком:
// читаем только заголовки боксов и сам moov (обычно это несколько мегабайт).

export type VideoProbe = { duration: number | null; fps: number | null };

const CONTAINERS = new Set(["moov", "trak", "mdia", "minf", "stbl"]);

async function readBytes(file: Blob, start: number, end: number) {
  return new DataView(await file.slice(start, end).arrayBuffer());
}

const boxType = (view: DataView, offset: number) =>
  String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));

async function findTopLevelBox(file: File, type: string) {
  let offset = 0;
  while (offset + 8 <= file.size) {
    const header = await readBytes(file, offset, Math.min(file.size, offset + 16));
    let size = header.getUint32(0);
    const name = boxType(header, 4);
    let headerSize = 8;
    if (size === 1 && header.byteLength >= 16) {
      size = header.getUint32(8) * 4294967296 + header.getUint32(12);
      headerSize = 16;
    } else if (size === 0) {
      size = file.size - offset;
    }
    if (size < headerSize) return null;
    if (name === type) return { start: offset + headerSize, end: Math.min(file.size, offset + size) };
    offset += size;
  }
  return null;
}

type Track = { handler?: string; timescale?: number; mediaDuration?: number; samples?: number; sampleTime?: number };

function walk(view: DataView, start: number, end: number, track: Track | null, tracks: Track[], movie: { timescale?: number; duration?: number }) {
  let offset = start;
  while (offset + 8 <= end) {
    let size = view.getUint32(offset);
    const name = boxType(view, offset + 4);
    let headerSize = 8;
    if (size === 1) {
      size = view.getUint32(offset + 8) * 4294967296 + view.getUint32(offset + 12);
      headerSize = 16;
    } else if (size === 0) {
      size = end - offset;
    }
    if (size < headerSize || offset + size > end) break;
    const body = offset + headerSize;
    if (name === "trak") {
      const next: Track = {};
      tracks.push(next);
      walk(view, body, offset + size, next, tracks, movie);
    } else if (CONTAINERS.has(name)) {
      walk(view, body, offset + size, track, tracks, movie);
    } else if (name === "mvhd") {
      const version = view.getUint8(body);
      movie.timescale = view.getUint32(body + (version === 1 ? 20 : 12));
      movie.duration = version === 1
        ? view.getUint32(body + 24) * 4294967296 + view.getUint32(body + 28)
        : view.getUint32(body + 16);
    } else if (track && name === "mdhd") {
      const version = view.getUint8(body);
      track.timescale = view.getUint32(body + (version === 1 ? 20 : 12));
      track.mediaDuration = version === 1
        ? view.getUint32(body + 24) * 4294967296 + view.getUint32(body + 28)
        : view.getUint32(body + 16);
    } else if (track && name === "hdlr") {
      track.handler = boxType(view, body + 8);
    } else if (track && name === "stts") {
      const entries = view.getUint32(body + 4);
      let samples = 0;
      let sampleTime = 0;
      for (let index = 0; index < entries && body + 16 + index * 8 <= offset + size; index += 1) {
        const count = view.getUint32(body + 8 + index * 8);
        const delta = view.getUint32(body + 12 + index * 8);
        samples += count;
        sampleTime += count * delta;
      }
      track.samples = samples;
      track.sampleTime = sampleTime;
    }
    offset += size;
  }
}

export async function probeIsoVideo(file: File): Promise<VideoProbe> {
  const empty = { duration: null, fps: null };
  if (!/\.(mp4|mov|m4v|3gp)$/i.test(file.name) && !/mp4|quicktime|3gpp/i.test(file.type)) return empty;
  try {
    const moov = await findTopLevelBox(file, "moov");
    if (!moov || moov.end - moov.start > 256 * 1024 * 1024) return empty;
    const view = await readBytes(file, moov.start, moov.end);
    const tracks: Track[] = [];
    const movie: { timescale?: number; duration?: number } = {};
    walk(view, 0, view.byteLength, null, tracks, movie);
    const video = tracks.find((track) => track.handler === "vide");
    let fps: number | null = null;
    if (video?.timescale && video.samples && video.sampleTime) {
      const value = (video.samples * video.timescale) / video.sampleTime;
      if (Number.isFinite(value) && value > 1 && value < 1000) fps = value;
    }
    let duration: number | null = null;
    if (video?.timescale && video.mediaDuration) duration = video.mediaDuration / video.timescale;
    else if (movie.timescale && movie.duration) duration = movie.duration / movie.timescale;
    return { duration: duration && Number.isFinite(duration) && duration > 0 ? duration : null, fps };
  } catch {
    return empty;
  }
}
