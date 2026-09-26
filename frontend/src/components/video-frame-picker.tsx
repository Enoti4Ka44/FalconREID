import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { AlertCircle, Camera, Check, Film, Loader2, Pause, Play, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { VideoFrameDecoder, type VideoInfo } from "@/lib/ffmpeg-frames";
import { probeIsoVideo } from "@/lib/video-probe";

type PickerProps = { file: File; onCancel: () => void; onCapture: (file: File) => void };

const THUMB_COUNT = 10;
const DEFAULT_FPS = 30;

function clock(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00.00";
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(2).padStart(5, "0")}`;
}

const frameFileName = (file: File, frame: number, time: number, extension = "jpg") =>
  `${file.name.replace(/\.[^.]+$/, "")}-frame-${frame + 1}-${time.toFixed(2)}s.${extension}`;

const lastFrameOf = (duration: number, fps: number) => Math.max(0, Math.round(duration * fps) - 1);

/**
 * Выбор кадра из видео. Сначала пробуем показать ролик нативно (быстро, покадрово),
 * а если браузер не умеет декодировать этот кодек — переключаемся на ffmpeg.wasm,
 * который рисует настоящие кадры по мере перемещения по шкале.
 */
export function VideoFramePicker(props: PickerProps) {
  const [mode, setMode] = useState<"native" | "decoder">("native");
  return mode === "native"
    ? <NativeVideoPicker {...props} onUnsupported={() => setMode("decoder")} />
    : <DecoderVideoPicker {...props} />;
}

/* ───────────────────────────── Общая оболочка ───────────────────────────── */

function PickerShell({ file, onCancel, badge, viewport, timeline, controls, footer }: {
  file: File;
  onCancel: () => void;
  badge?: ReactNode;
  viewport: ReactNode;
  timeline: ReactNode;
  controls: ReactNode;
  footer: ReactNode;
}) {
  return <div className="overflow-hidden rounded-2xl border border-blue-300/20 bg-[#050d16] shadow-[0_24px_80px_rgba(2,8,23,.4)]">
    <div className="flex items-center justify-between gap-3 border-b border-white/[.07] px-5 py-4">
      <div className="flex min-w-0 items-center gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-blue-300/10 text-blue-200"><Film className="h-4 w-4" /></div>
        <div className="min-w-0">
          <p className="text-sm font-medium text-slate-200">Выбор кадра из видео</p>
          <p className="truncate text-[11px] text-slate-500">{file.name}</p>
        </div>
        {badge}
      </div>
      <button type="button" onClick={onCancel} className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-slate-500 hover:bg-white/5 hover:text-white" aria-label="Закрыть видео"><X className="h-4 w-4" /></button>
    </div>
    {viewport}
    <div className="space-y-4 border-t border-white/[.07] p-4 sm:p-5">
      {timeline}
      {controls}
      {footer}
    </div>
  </div>;
}

function Timeline({ frame, lastFrame, fps, thumbnails, disabled, onChange, onDragStart, onDragEnd }: {
  frame: number;
  lastFrame: number;
  fps: number;
  thumbnails: (string | null)[];
  disabled: boolean;
  onChange: (frame: number) => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
}) {
  const progress = lastFrame > 0 ? Math.min(100, Math.max(0, (frame / lastFrame) * 100)) : 0;
  return <div className="relative overflow-hidden rounded-xl border border-white/10 bg-[#020811] shadow-inner">
    <div className="grid h-16 overflow-hidden opacity-70 sm:h-20" style={{ gridTemplateColumns: `repeat(${THUMB_COUNT}, minmax(0, 1fr))` }}>
      {Array.from({ length: THUMB_COUNT }, (_, index) => thumbnails[index]
        ? <img key={index} src={thumbnails[index]!} alt="" draggable={false} className="h-full w-full object-cover" />
        : <div key={index} className={`border-r border-white/[.05] ${index % 2 ? "bg-blue-300/[.04]" : "bg-white/[.025]"}`} />)}
    </div>
    <div className="pointer-events-none absolute inset-y-0 left-0 bg-blue-400/15" style={{ width: `${progress}%` }} />
    <div className="pointer-events-none absolute inset-y-0 w-px bg-blue-200 shadow-[0_0_12px_rgba(147,197,253,.9)]" style={{ left: `${progress}%` }}>
      <span className="absolute -left-2 -top-1 h-4 w-4 rounded-full border-2 border-[#07131f] bg-blue-200 shadow-lg" />
    </div>
    <input
      aria-label="Выбор момента видео"
      type="range"
      min={0}
      max={lastFrame}
      step={1}
      value={Math.min(frame, lastFrame)}
      disabled={disabled}
      onChange={(event) => onChange(Number(event.currentTarget.value))}
      onPointerDown={onDragStart}
      onPointerUp={onDragEnd}
      onPointerCancel={onDragEnd}
      onBlur={onDragEnd}
      className="timeline-range absolute inset-0 h-full w-full cursor-ew-resize opacity-0 disabled:cursor-not-allowed"
    />
    <div className="pointer-events-none absolute bottom-1 left-2 rounded-md bg-black/65 px-2 py-1 font-mono text-[10px] text-blue-100 backdrop-blur">{clock(frame / fps)}</div>
    <div className="pointer-events-none absolute bottom-1 right-2 rounded-md bg-black/65 px-2 py-1 font-mono text-[10px] text-slate-400 backdrop-blur">{lastFrame > 0 ? clock((lastFrame + 1) / fps) : "--:--.--"}</div>
  </div>;
}

function StepControls({ frame, lastFrame, fps, disabled, playing, onTogglePlay, onStep }: {
  frame: number;
  lastFrame: number;
  fps: number;
  disabled: boolean;
  playing?: boolean;
  onTogglePlay?: () => void;
  onStep: (delta: number) => void;
}) {
  const fiveSeconds = Math.max(1, Math.round(fps * 5));
  return <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
    <div className={`grid gap-2 sm:flex sm:flex-wrap sm:items-center ${onTogglePlay ? "grid-cols-[40px_repeat(4,minmax(0,1fr))]" : "grid-cols-4"}`}>
      {onTogglePlay && <Button type="button" variant="outline" size="sm" onClick={onTogglePlay} disabled={disabled} className="w-10 px-0" aria-label={playing ? "Пауза" : "Воспроизвести"}>{playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}</Button>}
      <Button type="button" variant="outline" size="sm" className="px-2 sm:px-3" onClick={() => onStep(-fiveSeconds)} disabled={disabled || frame <= 0}>−5 сек</Button>
      <Button type="button" variant="outline" size="sm" className="px-2 sm:px-3" onClick={() => onStep(-1)} disabled={disabled || frame <= 0}>−1 кадр</Button>
      <Button type="button" variant="outline" size="sm" className="px-2 sm:px-3" onClick={() => onStep(1)} disabled={disabled || frame >= lastFrame}>+1 кадр</Button>
      <Button type="button" variant="outline" size="sm" className="px-2 sm:px-3" onClick={() => onStep(fiveSeconds)} disabled={disabled || frame >= lastFrame}>+5 сек</Button>
    </div>
    <div className="self-start rounded-lg border border-blue-300/15 bg-blue-300/[.06] px-3 py-1.5 font-mono text-xs text-blue-100 sm:self-auto sm:text-sm">
      {clock(frame / fps)} <span className="text-slate-600">/</span> <span className="text-slate-400">{lastFrame > 0 ? clock((lastFrame + 1) / fps) : "--:--.--"}</span>
      <span className="ml-2 text-[10px] text-slate-500">кадр {frame + 1}/{lastFrame + 1}</span>
    </div>
  </div>;
}

function CaptureFooter({ message, error, hint, disabled, capturing, onCapture }: {
  message: string;
  error: boolean;
  hint: string;
  disabled: boolean;
  capturing: boolean;
  onCapture: () => void;
}) {
  return <div className={`flex flex-col items-stretch gap-3 rounded-xl border px-4 py-3 sm:flex-row sm:items-center sm:justify-between ${error ? "border-red-400/20 bg-red-400/[.07]" : "border-white/[.06] bg-white/[.02]"}`}>
    <div className="min-w-0 flex-1">
      <p className={`flex items-start gap-2 text-xs ${error ? "text-red-200" : "text-slate-300"}`}>{error && <AlertCircle className="mt-px h-3.5 w-3.5 shrink-0" />}{message}</p>
      <p className="mt-1 text-[10px] text-slate-600">{hint}</p>
    </div>
    <Button type="button" onClick={onCapture} disabled={disabled || capturing} size="sm" className="rounded-full bg-blue-100 px-5 text-slate-950 shadow-[0_0_24px_rgba(147,197,253,.16)] hover:bg-white">
      {capturing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Camera className="h-3.5 w-3.5" />}
      {capturing ? "Создаём снимок…" : "Использовать этот кадр"}
    </Button>
  </div>;
}

/* ─────────────────────── Нативное воспроизведение ─────────────────────── */

function NativeVideoPicker({ file, onCancel, onCapture, onUnsupported }: PickerProps & { onUnsupported: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const unsupportedRef = useRef(false);
  const infiniteDurationRef = useRef(false);
  // URL создаётся внутри эффекта: в StrictMode эффект монтируется дважды, и ссылка из useState
  // оказывалась отозванной ещё до загрузки видео.
  const [url, setUrl] = useState<string | null>(null);
  const [fps, setFps] = useState(DEFAULT_FPS);
  const [fpsKnown, setFpsKnown] = useState(false);
  const [duration, setDuration] = useState(0);
  const [time, setTime] = useState(0);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [thumbnails, setThumbnails] = useState<(string | null)[]>([]);

  useEffect(() => {
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);

  useEffect(() => {
    let cancelled = false;
    void probeIsoVideo(file).then((probe) => {
      if (cancelled || !probe.fps) return;
      setFps(probe.fps);
      setFpsKnown(true);
    });
    return () => { cancelled = true; };
  }, [file]);

  const unsupported = useCallback(() => {
    if (unsupportedRef.current) return;
    unsupportedRef.current = true;
    onUnsupported();
  }, [onUnsupported]);

  // Если за разумное время не удалось получить ни одного кадра — браузер этот кодек не тянет.
  useEffect(() => {
    if (!url || ready) return;
    const timeout = window.setTimeout(unsupported, 10_000);
    return () => window.clearTimeout(timeout);
  }, [ready, unsupported, url]);

  // Плавное обновление позиции во время воспроизведения.
  useEffect(() => {
    if (!playing) return;
    let handle = 0;
    const tick = () => {
      if (videoRef.current) setTime(videoRef.current.currentTime);
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing]);

  const lastFrame = lastFrameOf(duration, fps);
  const frame = Math.min(lastFrame, Math.max(0, Math.floor(time * fps + 1e-3)));

  const seekFrame = (next: number) => {
    const video = videoRef.current;
    if (!video || !duration) return;
    const target = Math.max(0, Math.min(lastFrame, Math.round(next)));
    video.pause();
    // Середина интервала кадра — так браузер гарантированно покажет именно его.
    const seconds = Math.min(duration - 0.001, (target + 0.5) / fps);
    video.currentTime = seconds;
    setTime(seconds);
    setError(null);
  };

  const togglePlayback = async () => {
    const video = videoRef.current;
    if (!video || !ready) return;
    try {
      if (video.paused) {
        if (video.ended || video.currentTime >= duration - 0.05) video.currentTime = 0;
        await video.play();
      } else {
        video.pause();
      }
    } catch {
      setPlaying(false);
    }
  };

  const handleMetadata = (video: HTMLVideoElement) => {
    // Браузер открыл контейнер, но не видеодорожку (например, HEVC без поддержки) — играет только звук.
    if (!video.videoWidth || !video.videoHeight) {
      unsupported();
      return;
    }
    if (video.duration === Infinity) {
      // WebM из MediaRecorder не содержит длительности: заставляем браузер её вычислить.
      infiniteDurationRef.current = true;
      video.currentTime = 1e101;
      return;
    }
    if (Number.isFinite(video.duration) && video.duration > 0) setDuration(video.duration);
  };

  const handleDurationChange = (video: HTMLVideoElement) => {
    if (!Number.isFinite(video.duration) || video.duration <= 0) return;
    setDuration(video.duration);
    if (infiniteDurationRef.current) {
      infiniteDurationRef.current = false;
      video.currentTime = 0;
      setTime(0);
    }
  };

  const waitForFrame = (video: HTMLVideoElement) => new Promise<void>((resolve) => {
    const done = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("seeked", onSeeked);
      resolve();
    };
    const onSeeked = () => {
      if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(() => done());
      else done();
    };
    const timeout = window.setTimeout(done, 3000);
    if (!video.seeking && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) done();
    else video.addEventListener("seeked", onSeeked, { once: true });
  });

  const capture = async () => {
    const video = videoRef.current;
    if (!video) return;
    setCapturing(true);
    setError(null);
    try {
      video.pause();
      await waitForFrame(video);
      if (!video.videoWidth || !video.videoHeight) throw new Error("Кадр видео ещё не готов.");
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) throw new Error("Не удалось подготовить изображение.");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.94));
      if (!blob) throw new Error("Не удалось сохранить выбранный кадр.");
      const currentFrame = Math.floor(video.currentTime * fps + 1e-3);
      onCapture(new File([blob], frameFileName(file, currentFrame, video.currentTime), { type: "image/jpeg" }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Не удалось извлечь кадр из видео.");
    } finally {
      setCapturing(false);
    }
  };

  // Лента миниатюр строится отдельным невидимым <video>, чтобы не мешать основному.
  useEffect(() => {
    if (!ready || !duration || !url) return;
    let cancelled = false;
    const preview = document.createElement("video");
    preview.src = url;
    preview.muted = true;
    preview.playsInline = true;
    preview.preload = "auto";
    const waitFor = (eventName: "loadeddata" | "seeked") => new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error("thumbnail timeout")), 4000);
      preview.addEventListener(eventName, () => { window.clearTimeout(timeout); resolve(); }, { once: true });
    });
    void (async () => {
      try {
        if (preview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) await waitFor("loadeddata");
        const ratio = preview.videoWidth && preview.videoHeight ? preview.videoWidth / preview.videoHeight : 16 / 9;
        const canvas = document.createElement("canvas");
        canvas.width = 192;
        canvas.height = Math.max(1, Math.round(192 / ratio));
        const context = canvas.getContext("2d");
        if (!context) return;
        const images: (string | null)[] = Array(THUMB_COUNT).fill(null);
        for (let index = 0; index < THUMB_COUNT && !cancelled; index += 1) {
          preview.currentTime = Math.min(duration - 0.01, (duration * (index + 0.5)) / THUMB_COUNT);
          await waitFor("seeked");
          context.drawImage(preview, 0, 0, canvas.width, canvas.height);
          images[index] = canvas.toDataURL("image/jpeg", 0.6);
          if (!cancelled) setThumbnails([...images]);
        }
      } catch {
        // Шкала остаётся рабочей и без миниатюр.
      }
    })();
    return () => { cancelled = true; preview.removeAttribute("src"); preview.load(); };
  }, [duration, ready, url]);

  const controlsDisabled = !ready || !duration || capturing;

  return <PickerShell
    file={file}
    onCancel={onCancel}
    viewport={<div className="relative bg-black">
      {url && <video
        ref={videoRef}
        src={url}
        preload="auto"
        playsInline
        onClick={() => void togglePlayback()}
        onLoadedMetadata={(event) => handleMetadata(event.currentTarget)}
        onDurationChange={(event) => handleDurationChange(event.currentTarget)}
        onLoadedData={(event) => { if (event.currentTarget.videoWidth) setReady(true); }}
        onSeeked={(event) => { if (!infiniteDurationRef.current) setTime(event.currentTarget.currentTime); }}
        onPlay={() => setPlaying(true)}
        onPause={(event) => { setPlaying(false); setTime(event.currentTarget.currentTime); }}
        onEnded={() => setPlaying(false)}
        onError={unsupported}
        className="aspect-video max-h-[62vh] w-full cursor-pointer bg-black object-contain"
      />}
      {ready && !playing && <button type="button" onClick={() => void togglePlayback()} className="absolute left-1/2 top-1/2 grid h-14 w-14 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-white/20 bg-[#07131f]/70 text-white opacity-80 shadow-2xl backdrop-blur-md transition hover:scale-105 hover:bg-blue-400/30 hover:opacity-100" aria-label="Воспроизвести видео"><Play className="ml-1 h-6 w-6 fill-current" /></button>}
      {!ready && <div className="pointer-events-none absolute inset-0 grid place-items-center bg-black/55"><div className="flex items-center gap-2 rounded-full border border-white/10 bg-[#07131f]/90 px-4 py-2 text-xs text-slate-200"><Loader2 className="h-4 w-4 animate-spin text-blue-200" />Открываем видео</div></div>}
    </div>}
    timeline={<Timeline frame={frame} lastFrame={lastFrame} fps={fps} thumbnails={thumbnails} disabled={controlsDisabled} onChange={seekFrame} />}
    controls={<StepControls frame={frame} lastFrame={lastFrame} fps={fps} disabled={controlsDisabled} playing={playing} onTogglePlay={() => void togglePlayback()} onStep={(delta) => seekFrame(frame + delta)} />}
    footer={<CaptureFooter
      message={error ?? (ready ? "Остановите видео на нужном моменте — в поиск уйдёт именно тот кадр, который вы видите." : "Открываем видео…")}
      error={Boolean(error)}
      hint={`Шкала и стрелки ← → двигают видео покадрово${fpsKnown ? ` · ${fps.toFixed(fps % 1 ? 2 : 0)} к/с` : ""}.`}
      disabled={controlsDisabled}
      capturing={capturing}
      onCapture={() => void capture()}
    />}
  />;
}

/* ─────────────────── Декодирование через ffmpeg.wasm ─────────────────── */

type Shown = { url: string; frame: number; exact: boolean };
type Waiter = { resolve: (blob: Blob) => void; reject: (error: Error) => void };

function DecoderVideoPicker({ file, onCancel, onCapture }: PickerProps) {
  const decoderRef = useRef<VideoFrameDecoder | null>(null);
  const infoRef = useRef<VideoInfo | null>(null);
  const currentFrameRef = useRef(0);
  const wantedRef = useRef<{ frame: number; exact: boolean } | null>(null);
  const runningRef = useRef(false);
  const exactCacheRef = useRef(new Map<number, Blob>());
  const waitersRef = useRef(new Map<number, Waiter[]>());
  const thumbQueueRef = useRef<number[]>([]);
  const shownRef = useRef<Shown | null>(null);
  const settleTimerRef = useRef(0);
  const draggingRef = useRef(false);

  const [phase, setPhase] = useState<"loading" | "probing" | "ready" | "error">("loading");
  const [loadProgress, setLoadProgress] = useState<number | null>(null);
  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [frame, setFrame] = useState(0);
  const [shown, setShown] = useState<Shown | null>(null);
  const [decoding, setDecoding] = useState(false);
  const [thumbnails, setThumbnails] = useState<(string | null)[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);

  const fps = info?.fps ?? DEFAULT_FPS;
  const lastFrame = info ? lastFrameOf(info.duration, info.fps) : 0;

  const show = useCallback((blob: Blob, frameIndex: number, exact: boolean) => {
    const previous = shownRef.current;
    const next = { url: URL.createObjectURL(blob), frame: frameIndex, exact };
    shownRef.current = next;
    setShown(next);
    if (previous) URL.revokeObjectURL(previous.url);
  }, []);

  const pump = useCallback(async () => {
    const decoder = decoderRef.current;
    const meta = infoRef.current;
    if (runningRef.current || !decoder || !meta) return;
    runningRef.current = true;
    try {
      while (decoderRef.current === decoder) {
        const wanted = wantedRef.current;
        if (wanted) {
          wantedRef.current = null;
          const cached = wanted.exact ? exactCacheRef.current.get(wanted.frame) : undefined;
          if (cached) {
            if (wanted.frame === currentFrameRef.current) show(cached, wanted.frame, true);
            continue;
          }
          setDecoding(true);
          try {
            // Точный кадр: seek чуть раньше его метки времени, ffmpeg отбросит всё, что раньше.
            const blob = wanted.exact
              ? await decoder.frame(Math.max(0, (wanted.frame - 0.25) / meta.fps), { maxSide: 1920 })
              : await decoder.frame((wanted.frame + 0.5) / meta.fps, { maxSide: 960, keyframe: true, timeoutMs: 30_000 });
            if (decoderRef.current !== decoder) return;
            if (wanted.exact) {
              exactCacheRef.current.set(wanted.frame, blob);
              if (exactCacheRef.current.size > 24) exactCacheRef.current.delete(exactCacheRef.current.keys().next().value!);
              waitersRef.current.get(wanted.frame)?.forEach((waiter) => waiter.resolve(blob));
              waitersRef.current.delete(wanted.frame);
              if (wanted.frame === currentFrameRef.current) show(blob, wanted.frame, true);
              setMessage(null);
            } else {
              const current = shownRef.current;
              if (!(current?.exact && current.frame === currentFrameRef.current)) show(blob, wanted.frame, false);
            }
          } catch (cause) {
            if (decoderRef.current !== decoder) return;
            const error = cause instanceof Error ? cause : new Error("Не удалось декодировать кадр.");
            if (wanted.exact) {
              waitersRef.current.get(wanted.frame)?.forEach((waiter) => waiter.reject(error));
              waitersRef.current.delete(wanted.frame);
              if (wanted.frame === currentFrameRef.current) setMessage(`Кадр не декодировался: ${error.message}`);
            }
          }
          continue;
        }
        const thumbIndex = thumbQueueRef.current.shift();
        if (thumbIndex !== undefined) {
          setDecoding(false);
          try {
            const blob = await decoder.frame((meta.duration * (thumbIndex + 0.5)) / THUMB_COUNT, { maxSide: 192, keyframe: true, timeoutMs: 30_000 });
            if (decoderRef.current !== decoder) return;
            const thumbUrl = URL.createObjectURL(blob);
            setThumbnails((current) => {
              const next = [...current];
              next[thumbIndex] = thumbUrl;
              return next;
            });
          } catch {
            // Миниатюры необязательны.
          }
          continue;
        }
        break;
      }
    } finally {
      runningRef.current = false;
      setDecoding(false);
    }
  }, [show]);

  useEffect(() => {
    const decoder = new VideoFrameDecoder(file);
    decoderRef.current = decoder;
    const cache = exactCacheRef.current;
    const waiters = waitersRef.current;
    let cancelled = false;
    void (async () => {
      try {
        await decoder.load((ratio) => { if (!cancelled) setLoadProgress(ratio); });
        if (cancelled) return;
        setPhase("probing");
        let meta = await decoder.probe();
        if (!Number.isFinite(meta.duration) || meta.duration <= 0) {
          const probe = await probeIsoVideo(file);
          if (!probe.duration) throw new Error("Не удалось определить длительность видео.");
          meta = { ...meta, duration: probe.duration };
        }
        if (cancelled) return;
        infoRef.current = meta;
        setInfo(meta);
        setPhase("ready");
        currentFrameRef.current = 0;
        wantedRef.current = { frame: 0, exact: true };
        thumbQueueRef.current = Array.from({ length: THUMB_COUNT }, (_, index) => index);
        void pump();
      } catch (cause) {
        if (cancelled) return;
        setPhase("error");
        setFatal(cause instanceof Error ? `Не удалось открыть видео: ${cause.message}` : "Не удалось открыть видео.");
      }
    })();
    return () => {
      cancelled = true;
      decoder.dispose();
      if (decoderRef.current === decoder) decoderRef.current = null;
      infoRef.current = null;
      wantedRef.current = null;
      thumbQueueRef.current = [];
      cache.clear();
      waiters.forEach((list) => list.forEach((waiter) => waiter.reject(new Error("Видео закрыто."))));
      waiters.clear();
      window.clearTimeout(settleTimerRef.current);
    };
  }, [file, pump]);

  // Освобождаем object URL миниатюр и текущего кадра при размонтировании.
  const thumbUrlsRef = useRef<string[]>([]);
  useEffect(() => { thumbUrlsRef.current = thumbnails.filter((url): url is string => Boolean(url)); }, [thumbnails]);
  useEffect(() => () => {
    if (shownRef.current) URL.revokeObjectURL(shownRef.current.url);
    thumbUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  const selectFrame = (next: number, { preview }: { preview: boolean }) => {
    const meta = infoRef.current;
    if (!meta) return;
    const target = Math.max(0, Math.min(lastFrameOf(meta.duration, meta.fps), Math.round(next)));
    currentFrameRef.current = target;
    setFrame(target);
    setMessage(null);
    window.clearTimeout(settleTimerRef.current);
    const cached = exactCacheRef.current.get(target);
    if (cached) {
      show(cached, target, true);
      return;
    }
    if (preview) {
      // Пока пользователь тянет ползунок — быстрый предпросмотр по ближайшему ключевому кадру.
      wantedRef.current = { frame: target, exact: false };
      void pump();
    }
    settleTimerRef.current = window.setTimeout(() => {
      wantedRef.current = { frame: currentFrameRef.current, exact: true };
      void pump();
    }, preview ? 280 : 160);
  };

  const exactFrame = (frameIndex: number) => {
    const cached = exactCacheRef.current.get(frameIndex);
    if (cached) return Promise.resolve(cached);
    return new Promise<Blob>((resolve, reject) => {
      const list = waitersRef.current.get(frameIndex) ?? [];
      list.push({ resolve, reject });
      waitersRef.current.set(frameIndex, list);
      window.clearTimeout(settleTimerRef.current);
      wantedRef.current = { frame: frameIndex, exact: true };
      void pump();
    });
  };

  const capture = async () => {
    const meta = infoRef.current;
    if (!meta) return;
    const target = currentFrameRef.current;
    setCapturing(true);
    setMessage(null);
    try {
      // PNG без потерь — API принимает его напрямую, лишнее перекодирование не нужно.
      const blob = await exactFrame(target);
      onCapture(new File([blob], frameFileName(file, target, target / meta.fps, "png"), { type: "image/png" }));
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : "Не удалось создать снимок.");
    } finally {
      setCapturing(false);
    }
  };

  const exactShown = Boolean(shown?.exact && shown.frame === frame);
  const ready = phase === "ready";
  const controlsDisabled = !ready || capturing;
  const statusMessage = fatal
    ?? message
    ?? (phase === "loading" ? "Загружаем видеодекодер…"
      : phase === "probing" ? "Читаем параметры видео…"
        : exactShown ? "Это точный кадр — именно он уйдёт в поиск."
          : "Декодируем выбранный кадр…");

  return <PickerShell
    file={file}
    onCancel={onCancel}
    badge={info && <span className="hidden shrink-0 rounded-md border border-amber-300/20 bg-amber-300/[.07] px-2 py-1 text-[10px] text-amber-100/80 sm:inline" title="Браузер не поддерживает этот кодек, кадры декодируются локально">{info.codec.toUpperCase()} · локальный декодер</span>}
    viewport={<div className="relative grid aspect-video max-h-[62vh] w-full place-items-center overflow-hidden bg-black">
      {shown && <img src={shown.url} alt={`Кадр ${shown.frame + 1}`} className={`absolute inset-0 h-full w-full object-contain transition ${exactShown ? "" : "opacity-80"}`} />}
      {phase === "loading" && <div className="flex w-64 flex-col items-center gap-3 text-center">
        <Loader2 className="h-6 w-6 animate-spin text-blue-200" />
        <p className="text-xs text-slate-300">Браузер не умеет показывать этот формат сам — подключаем встроенный декодер</p>
        <div className="h-1 w-full overflow-hidden rounded-full bg-white/10"><div className={`h-full rounded-full bg-blue-300 transition-all ${loadProgress == null ? "w-1/3 animate-pulse" : ""}`} style={loadProgress == null ? undefined : { width: `${Math.round(loadProgress * 100)}%` }} /></div>
        {loadProgress != null && loadProgress < 1 && <p className="font-mono text-[10px] text-slate-500">{Math.round(loadProgress * 100)}%</p>}
      </div>}
      {phase === "probing" && !shown && <div className="flex items-center gap-2 text-xs text-slate-300"><Loader2 className="h-4 w-4 animate-spin text-blue-200" />Читаем видео…</div>}
      {phase === "error" && <div className="flex max-w-sm flex-col items-center gap-3 px-6 text-center"><AlertCircle className="h-7 w-7 text-red-300" /><p className="text-xs leading-5 text-red-100">{fatal}</p></div>}
      {/* Справа: слева сверху на записях камер обычно штамп даты и времени — его не закрываем. */}
      {ready && <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-2 rounded-full border border-white/10 bg-[#07131f]/85 px-3 py-1.5 text-[10px] text-slate-200 backdrop-blur">
        {exactShown
          ? <><Check className="h-3 w-3 text-emerald-300" />Точный кадр {frame + 1}{decoding && <Loader2 className="h-3 w-3 animate-spin text-slate-500" />}</>
          : <><Loader2 className="h-3 w-3 animate-spin text-blue-200" />{shown ? "Предпросмотр · уточняем кадр" : "Декодируем кадр"}</>}
      </div>}
    </div>}
    timeline={<Timeline
      frame={frame}
      lastFrame={lastFrame}
      fps={fps}
      thumbnails={thumbnails}
      disabled={controlsDisabled}
      onChange={(next) => selectFrame(next, { preview: draggingRef.current })}
      onDragStart={() => { draggingRef.current = true; }}
      onDragEnd={() => { draggingRef.current = false; }}
    />}
    controls={<StepControls frame={frame} lastFrame={lastFrame} fps={fps} disabled={controlsDisabled} onStep={(delta) => selectFrame(frame + delta, { preview: Math.abs(delta) > 1 })} />}
    footer={<CaptureFooter
      message={statusMessage}
      error={Boolean(fatal || message)}
      hint="Во время перетаскивания показывается ближайший ключевой кадр, после остановки — точный. Стрелки ← → — покадрово."
      disabled={!ready || Boolean(fatal)}
      capturing={capturing}
      onCapture={() => void capture()}
    />}
  />;
}
