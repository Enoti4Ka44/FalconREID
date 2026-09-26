import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import {
  AlertCircle,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Check,
  Columns2,
  Download,
  CircleHelp,
  Crop,
  Film,
  GalleryHorizontalEnd,
  Grid2X2,
  ImagePlus,
  Loader2,
  LogOut,
  Menu,
  RefreshCw,
  ScanSearch,
  Search,
  Siren,
  SlidersHorizontal,
  UploadCloud,
  UserRound,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { VideoFramePicker } from "@/components/video-frame-picker";
import { apiFetch, apiUrl } from "@/lib/api";
import { useAuth } from "@/lib/auth";

type ApiCandidate = {
  gallery_id: string;
  confidence: number;
  accepted: boolean;
  camera_group: number;
};

type SearchResponse = {
  query_embedding_dim: number;
  inference_ms: number;
  threshold: number;
  refused: boolean;
  message: string;
  n_query_images: number;
  sim_max: number;
  sim_hist: { counts: number[]; edges: number[] };
  candidates: ApiCandidate[];
  watchlist_alerts: Array<{ id: string; name: string; confidence: number }>;
};

const scorePercent = (value: number) => `${(value * 100).toFixed(1)}%`;
const candidateImage = (id: string) => apiUrl(`/api/gallery/${encodeURIComponent(id)}/thumb`);
const candidateFrame = (id: string) => apiUrl(`/api/gallery/${encodeURIComponent(id)}/frame`);

const MAX_FILES = 5;
const VIDEO_EXTENSIONS = /\.(mp4|m4v|mov|webm|mkv|avi|3gp|3g2|ts|mts|m2ts|wmv|flv|ogv|mpg|mpeg)$/i;
const FILE_ACCEPT = "image/*,video/*,.mov,.mkv,.avi,.m4v,.3gp,.ts,.mts,.m2ts,.wmv,.flv";
const isVideoFile = (file: File) => file.type.startsWith("video/") || VIDEO_EXTENSIONS.test(file.name);

/** API принимает JPG и PNG: остальные растровые форматы перекодируем в JPEG прямо в браузере. */
async function normalizeImage(file: File): Promise<File | null> {
  if (file.type === "image/jpeg" || file.type === "image/png") return file;
  try {
    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    canvas.getContext("2d")?.drawImage(bitmap, 0, 0);
    bitmap.close();
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.94));
    return blob ? new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" }) : null;
  } catch {
    return null;
  }
}

// null — раздел ещё не реализован: показываем его неактивным, а не ссылкой на ту же страницу.
const navItems = [
  ["Обзор", Grid2X2, null],
  ["Поиск", Search, "/app"],
  ["Галерея", GalleryHorizontalEnd, "/gallery"],
  ["Сравнение", SlidersHorizontal, null],
  ["Аналитика", BarChart3, "/analytics"],
] as const;

function NavLink({ label, Icon, href, active, mobile }: { label: string; Icon: typeof Search; href: string | null; active: boolean; mobile?: boolean }) {
  const base = mobile ? "flex items-center gap-3 rounded-lg px-3 py-3 text-sm" : "flex items-center gap-2 rounded-full px-4 py-2 text-[13px] transition";
  if (!href) {
    return <span aria-disabled="true" title="Раздел в разработке" className={`${base} cursor-not-allowed text-slate-600`}>
      <Icon className={mobile ? "h-4 w-4" : "h-3.5 w-3.5"} />{label}<span className="rounded bg-white/[.05] px-1.5 py-px text-[9px] uppercase tracking-wider text-slate-600">скоро</span>
    </span>;
  }
  const state = active
    ? mobile ? "bg-blue-300/10 text-blue-100" : "bg-blue-300/15 text-blue-100 shadow-[inset_0_0_0_1px_rgba(125,184,255,.25)]"
    : mobile ? "text-slate-300 hover:bg-white/5" : "text-slate-400 hover:bg-white/[.04] hover:text-slate-100";
  return <a href={href} aria-current={active ? "page" : undefined} className={`${base} ${state}`}><Icon className={mobile ? "h-4 w-4" : "h-3.5 w-3.5"} />{label}</a>;
}

function Brand() {
  return (
    <a href="/" className="flex shrink-0 items-center gap-3" aria-label="FALCON — на главную">
      <img src="/falcon-mark.svg" width="34" height="34" alt="" />
      <span className="text-[17px] font-semibold tracking-[.28em] text-slate-100">FALCON</span>
    </a>
  );
}

export function AppHeader({ apiStatus, active = "Поиск" }: { apiStatus: "checking" | "online" | "offline"; active?: (typeof navItems)[number][0] }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const { user, logout } = useAuth();
  const initials = user?.username.slice(0, 2).toUpperCase() ?? "—";
  const signOut = async () => {
    await logout();
    window.location.assign("/auth");
  };
  return (
    <header className="sticky top-0 z-50 border-b border-white/[.07] bg-[#050d16]/85 backdrop-blur-xl">
      <div className="mx-auto flex h-[72px] max-w-[1580px] items-center gap-7 px-5 lg:px-8">
        <Brand />
        <nav className="mx-auto hidden items-center gap-1 rounded-full border border-blue-200/[.12] bg-white/[.025] p-1 lg:flex" aria-label="Навигация приложения">
          {navItems.map(([label, Icon, href]) => <NavLink key={label} label={label} Icon={Icon} href={href} active={label === active} />)}
        </nav>
        <div className="ml-auto hidden items-center gap-3 md:flex lg:ml-0">
          <div className="flex h-10 items-center gap-2 rounded-full border border-white/[.08] bg-white/[.025] px-4 text-xs text-slate-500">
            <span className={`h-2 w-2 rounded-full ${apiStatus === "online" ? "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,.6)]" : apiStatus === "offline" ? "bg-red-400" : "animate-pulse bg-amber-300"}`} />
            {apiStatus === "online" ? "API подключён" : apiStatus === "offline" ? "API недоступен" : "Проверяем API"}
          </div>
          <div className="relative">
            <button type="button" onClick={() => setAccountOpen((value) => !value)} aria-expanded={accountOpen} aria-label="Открыть меню аккаунта" className="grid h-10 w-10 place-items-center rounded-full border border-blue-200/[.16] bg-blue-300/10 text-xs font-semibold text-blue-100 transition hover:bg-blue-300/15">{initials}</button>
            {accountOpen && <div className="absolute right-0 top-12 w-60 rounded-xl border border-white/[.09] bg-[#07131f] p-2 shadow-2xl">
              <div className="flex items-center gap-3 border-b border-white/[.06] px-3 py-3"><span className="grid h-9 w-9 place-items-center rounded-lg bg-blue-300/10"><UserRound className="h-4 w-4 text-blue-200" /></span><span className="min-w-0"><span className="block text-[10px] uppercase tracking-wider text-slate-600">Аккаунт</span><span className="block truncate text-sm text-slate-200">{user?.username}</span></span></div>
              <button type="button" onClick={() => void signOut()} className="mt-1 flex w-full items-center gap-2 rounded-lg px-3 py-2.5 text-left text-xs text-slate-400 transition hover:bg-white/[.04] hover:text-slate-100"><LogOut className="h-4 w-4" />Выйти из аккаунта</button>
            </div>}
          </div>
        </div>
        <button type="button" onClick={() => setMobileOpen((value) => !value)} aria-expanded={mobileOpen} className="ml-auto grid h-10 w-10 place-items-center rounded-full border border-white/10 md:ml-0 lg:hidden" aria-label={mobileOpen ? "Закрыть меню" : "Открыть меню"}>
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>
      {mobileOpen && <nav className="border-t border-white/[.06] bg-[#07121e] px-5 py-3 lg:hidden" aria-label="Мобильная навигация">{navItems.map(([label, Icon, href]) => <NavLink key={label} label={label} Icon={Icon} href={href} active={label === active} mobile />)}<p className="mt-2 flex items-center gap-2 border-t border-white/[.06] px-3 pt-3 text-xs text-slate-500 md:hidden"><span className={`h-2 w-2 rounded-full ${apiStatus === "online" ? "bg-emerald-400" : apiStatus === "offline" ? "bg-red-400" : "animate-pulse bg-amber-300"}`} />{apiStatus === "online" ? "API подключён" : apiStatus === "offline" ? "API недоступен" : "Проверяем API"}</p><div className="mt-3 flex items-center justify-between border-t border-white/[.06] px-3 pt-3"><span className="flex min-w-0 items-center gap-2 text-xs text-slate-400"><UserRound className="h-4 w-4 shrink-0" /><span className="truncate">{user?.username}</span></span><button type="button" onClick={() => void signOut()} className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-slate-400 hover:bg-white/5 hover:text-slate-100"><LogOut className="h-4 w-4" />Выйти</button></div></nav>}
    </header>
  );
}

type CropRect = { x: number; y: number; w: number; h: number };

function ImageCropEditor({ file, canUndo, onApply, onUndo }: { file: File; canUndo: boolean; onApply: (file: File) => void; onUndo: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const mappingRef = useRef({ scale: 1, x: 0, y: 0, w: 1, h: 1 });
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const [rect, setRect] = useState<CropRect | null>(null);
  const [ratio, setRatio] = useState<number | null>(null);
  const [applying, setApplying] = useState(false);

  const draw = (selection = rect) => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const scale = Math.min(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
    const width = image.naturalWidth * scale;
    const height = image.naturalHeight * scale;
    const x = (canvas.width - width) / 2;
    const y = (canvas.height - height) / 2;
    mappingRef.current = { scale, x, y, w: width, h: height };
    context.drawImage(image, x, y, width, height);
    if (!selection) return;
    context.save();
    context.fillStyle = "rgba(3, 10, 18, .66)";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.clearRect(selection.x, selection.y, selection.w, selection.h);
    context.drawImage(
      image,
      (selection.x - x) / scale,
      (selection.y - y) / scale,
      selection.w / scale,
      selection.h / scale,
      selection.x,
      selection.y,
      selection.w,
      selection.h,
    );
    context.strokeStyle = "#93c5fd";
    context.lineWidth = 2;
    context.shadowColor = "rgba(96, 165, 250, .8)";
    context.shadowBlur = 12;
    context.strokeRect(selection.x, selection.y, selection.w, selection.h);
    context.restore();
  };

  useEffect(() => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const scale = Math.min(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
      const width = image.naturalWidth * scale;
      const height = image.naturalHeight * scale;
      const initial = { x: (canvas.width - width) / 2, y: (canvas.height - height) / 2, w: width, h: height };
      setRect(initial);
      requestAnimationFrame(() => draw(initial));
    };
    image.src = url;
    return () => URL.revokeObjectURL(url);
    // draw intentionally uses the newly loaded image
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file]);

  useEffect(() => { draw(); }, [rect]);

  const point = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: (event.clientX - bounds.left) * (event.currentTarget.width / bounds.width),
      y: (event.clientY - bounds.top) * (event.currentTarget.height / bounds.height),
    };
  };

  const pointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const next = point(event);
    const image = mappingRef.current;
    dragStart.current = { x: Math.max(image.x, Math.min(next.x, image.x + image.w)), y: Math.max(image.y, Math.min(next.y, image.y + image.h)) };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const pointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!dragStart.current) return;
    const image = mappingRef.current;
    const current = point(event);
    const endX = Math.max(image.x, Math.min(current.x, image.x + image.w));
    const endY = Math.max(image.y, Math.min(current.y, image.y + image.h));
    let width = Math.abs(endX - dragStart.current.x);
    let height = Math.abs(endY - dragStart.current.y);
    if (ratio && width > 4 && height > 4) {
      if (width / height > ratio) width = height * ratio;
      else height = width / ratio;
    }
    const next = { x: endX >= dragStart.current.x ? dragStart.current.x : dragStart.current.x - width, y: endY >= dragStart.current.y ? dragStart.current.y : dragStart.current.y - height, w: width, h: height };
    setRect(next);
  };

  const pointerUp = () => { dragStart.current = null; };

  const applyCrop = async () => {
    if (!rect || !imageRef.current || rect.w < 10 || rect.h < 10) return;
    setApplying(true);
    const image = imageRef.current;
    const mapping = mappingRef.current;
    const sx = Math.max(0, (rect.x - mapping.x) / mapping.scale);
    const sy = Math.max(0, (rect.y - mapping.y) / mapping.scale);
    const sw = Math.min(image.naturalWidth - sx, rect.w / mapping.scale);
    const sh = Math.min(image.naturalHeight - sy, rect.h / mapping.scale);
    const output = document.createElement("canvas");
    output.width = Math.max(1, Math.round(sw));
    output.height = Math.max(1, Math.round(sh));
    output.getContext("2d")?.drawImage(image, sx, sy, sw, sh, 0, 0, output.width, output.height);
    const blob = await new Promise<Blob | null>((resolve) => output.toBlob(resolve, "image/jpeg", .94));
    if (blob) onApply(new File([blob], file.name.replace(/\.[^.]+$/, "") + "-crop.jpg", { type: "image/jpeg" }));
    setApplying(false);
  };

  const resetSelection = () => {
    const image = mappingRef.current;
    setRect({ x: image.x, y: image.y, w: image.w, h: image.h });
  };

  const selectRatio = (nextRatio: number | null) => {
    setRatio(nextRatio);
    const image = mappingRef.current;
    if (!nextRatio) {
      setRect({ x: image.x, y: image.y, w: image.w, h: image.h });
      return;
    }
    let width = image.w;
    let height = width / nextRatio;
    if (height > image.h) {
      height = image.h;
      width = height * nextRatio;
    }
    setRect({
      x: image.x + (image.w - width) / 2,
      y: image.y + (image.h - height) / 2,
      w: width,
      h: height,
    });
  };

  return <div className="overflow-hidden rounded-xl border border-white/[.08] bg-[#050d16]">
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[.07] px-4 py-3"><div><p className="text-sm font-medium text-slate-200">Кадрирование</p><p className="text-[11px] text-slate-500">Обведите автомобиль рамкой прямо на изображении</p></div><div className="flex items-center gap-1 rounded-lg border border-white/[.07] bg-white/[.025] p-1">{[["Свободно", null], ["4:3", 4 / 3], ["16:9", 16 / 9], ["1:1", 1]] .map(([label, value]) => <button type="button" key={label as string} onClick={() => selectRatio(value as number | null)} className={`rounded-md px-2.5 py-1.5 text-[11px] ${ratio === value ? "bg-blue-300/15 text-blue-100" : "text-slate-500 hover:text-slate-200"}`}>{label as string}</button>)}</div></div>
    <canvas ref={canvasRef} width={960} height={520} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={pointerUp} className="block aspect-[960/520] w-full cursor-crosshair touch-none" />
    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/[.07] px-4 py-3"><p className="text-[11px] text-slate-500">Можно оставить весь кадр или нарисовать новую область</p><div className="flex flex-wrap items-center gap-2"><Button type="button" variant="ghost" onClick={resetSelection} disabled={applying} size="sm" className="rounded-full text-slate-400 hover:text-white">Сбросить область</Button>{canUndo && <Button type="button" variant="outline" onClick={onUndo} disabled={applying} size="sm" className="rounded-full"><RefreshCw className="h-3.5 w-3.5" />Отменить кадрирование</Button>}<Button type="button" onClick={applyCrop} disabled={applying || !rect} size="sm" className="rounded-full bg-slate-50 text-slate-950 hover:bg-white">{applying ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Crop className="h-3.5 w-3.5" />}Применить кадрирование</Button></div></div>
  </div>;
}

function SearchParameters({ topK, threshold, onTopKChange, onThresholdChange, className = "" }: {
  topK: number;
  threshold: number | null;
  onTopKChange: (value: number) => void;
  onThresholdChange: (value: number | null) => void;
  className?: string;
}) {
  return <div className={`${className} rounded-xl border border-white/[.07] bg-[#050d16]/45 p-4`}>
    <div className="flex items-center justify-between gap-3">
      <span className="flex items-center gap-2 text-sm font-medium text-slate-200"><SlidersHorizontal className="h-4 w-4 text-blue-300" />Параметры поиска</span>
      <span className="text-xs text-slate-600">API</span>
    </div>
    <div className="mt-4 grid gap-4 md:grid-cols-2">
      <label className="flex items-center justify-between gap-4 text-xs text-slate-500">
        <span><span className="block text-slate-300">Количество кандидатов</span><span className="mt-1 block">Параметр top_k</span></span>
        <select value={topK} onChange={(event) => onTopKChange(Number(event.target.value))} className="h-9 rounded-lg border border-white/[.09] bg-[#07131f] px-3 text-xs text-slate-200 outline-none focus:border-blue-300/40">
          {[5, 10, 20, 50].map((value) => <option key={value} value={value}>{value}</option>)}
        </select>
      </label>
      <div className="border-t border-white/[.06] pt-4 md:border-l md:border-t-0 md:pl-4 md:pt-0">
        <div className="flex items-center justify-between gap-4">
          <span><span className="block text-xs text-slate-300">Порог совпадения</span><span className="mt-1 block text-xs text-slate-600">Параметр threshold</span></span>
          <button type="button" onClick={() => onThresholdChange(threshold == null ? 0.45 : null)} aria-pressed={threshold != null} className={`rounded-full border px-3 py-1.5 text-xs transition ${threshold == null ? "border-white/[.08] text-slate-500 hover:text-slate-300" : "border-blue-300/25 bg-blue-300/10 text-blue-100"}`}>{threshold == null ? "Авто" : threshold.toFixed(2)}</button>
        </div>
        {threshold != null && <div className="mt-3"><input type="range" min="0.1" max="0.95" step="0.01" value={threshold} onChange={(event) => onThresholdChange(Number(event.target.value))} aria-label="Порог совпадения" className="h-1.5 w-full cursor-pointer accent-blue-300" /><div className="mt-2 flex justify-between text-xs text-slate-700"><span>Мягче · 0.10</span><span>0.95 · Строже</span></div></div>}
        {threshold == null && <p className="mt-3 text-xs leading-5 text-slate-600">Используется порог, настроенный на сервере.</p>}
      </div>
    </div>
  </div>;
}

function UploadWorkspace({ files, originalFiles, activeIndex, loading, error, topK, threshold, onFiles, onRemove, onReplace, onUndoCrop, onActiveIndex, onTopKChange, onThresholdChange, onSearch }: {
  files: File[];
  originalFiles: File[];
  activeIndex: number;
  loading: boolean;
  error: string | null;
  topK: number;
  threshold: number | null;
  onFiles: (files: File[]) => void;
  onRemove: (index: number) => void;
  onReplace: (index: number, file: File) => void;
  onUndoCrop: (index: number) => void;
  onActiveIndex: (index: number) => void;
  onTopKChange: (value: number) => void;
  onThresholdChange: (value: number | null) => void;
  onSearch: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  // Ссылки создаются внутри эффекта: так их не отзовёт повторный монтаж в StrictMode.
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  useEffect(() => {
    const urls = files.map((file) => URL.createObjectURL(file));
    setPreviewUrls(urls);
    return () => urls.forEach((url) => URL.revokeObjectURL(url));
  }, [files]);
  const acceptFiles = (list: FileList | null) => {
    if (!list) return;
    const selected = Array.from(list);
    const video = selected.find(isVideoFile);
    if (video) setVideoFile(video);
    const images = selected.filter((file) => !isVideoFile(file));
    if (images.length) onFiles(images);
  };

  return (
    <section className="relative z-10 overflow-hidden rounded-2xl border border-blue-300/20 bg-[#06111b]/80 p-4 shadow-[0_30px_100px_rgba(0,0,0,.42)] backdrop-blur md:p-6">
      {files.length > 0 && <div className="mb-5 border-b border-white/[.07] pb-5"><div className="mb-3 flex items-center justify-between"><div><p className="text-sm font-medium text-slate-200">Загруженные кадры</p><p className="mt-1 text-[11px] text-slate-500">Выберите миниатюру, чтобы посмотреть или откадрировать</p></div><span className="text-xs text-slate-500">{files.length} / 5</span></div><div className="flex gap-3 overflow-x-auto pb-1">{files.map((file, index) => <button type="button" key={`${file.name}-${file.lastModified}`} onClick={() => onActiveIndex(index)} className={`group relative h-24 min-w-[150px] overflow-hidden rounded-xl border text-left transition ${activeIndex === index ? "border-blue-300 shadow-[0_0_0_2px_rgba(96,165,250,.12)]" : "border-white/[.08] opacity-70 hover:opacity-100"}`}><img src={previewUrls[index]} alt={`Кадр ${index + 1}`} className="h-full w-full object-cover" /><span className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#050d16] to-transparent px-2.5 pb-2 pt-6 text-[10px] text-slate-200">Кадр {index + 1}</span>{index === 0 && <span className="absolute left-2 top-2 rounded bg-blue-300 px-1.5 py-0.5 text-[8px] font-semibold uppercase text-[#06111b]">Основной</span>}<span onClick={(event) => { event.stopPropagation(); onRemove(index); }} className="absolute right-2 top-2 grid h-6 w-6 place-items-center rounded-md bg-[#050d16]/80 text-slate-400 opacity-0 backdrop-blur transition group-hover:opacity-100"><X className="h-3.5 w-3.5" /></span></button>)}{files.length < 5 && <button type="button" onClick={() => inputRef.current?.click()} className="flex h-24 min-w-[130px] flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-white/10 text-[10px] text-slate-500 hover:border-blue-300/30 hover:text-blue-200"><ImagePlus className="h-5 w-5" />Добавить кадр</button>}</div></div>}
      <div className="grid gap-5 xl:grid-cols-[1.55fr_.75fr]">
        <div>
          {videoFile ? <VideoFramePicker key={`${videoFile.name}-${videoFile.lastModified}-${videoFile.size}`} file={videoFile} onCancel={() => setVideoFile(null)} onCapture={(frame) => { onFiles([frame]); setVideoFile(null); }} /> : files.length > 0 ? <ImageCropEditor key={`${files[activeIndex]?.name}-${files[activeIndex]?.lastModified}`} file={files[activeIndex] ?? files[0]} canUndo={Boolean(originalFiles[activeIndex] && files[activeIndex] !== originalFiles[activeIndex])} onApply={(cropped) => onReplace(activeIndex, cropped)} onUndo={() => onUndoCrop(activeIndex)} /> : <button
            type="button"
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFiles(event.dataTransfer.files); }}
            className={`group relative flex min-h-[430px] w-full overflow-hidden rounded-xl border border-dashed transition ${dragging ? "border-blue-300 bg-blue-300/[.08]" : "border-blue-300/25 bg-[#07131f]/80 hover:border-blue-300/55 hover:bg-blue-300/[.035]"}`}
          >
            <div className="m-auto flex max-w-md flex-col items-center px-6 text-center">
                <div className="grid h-20 w-20 place-items-center rounded-2xl border border-blue-300/20 bg-blue-300/[.07] text-blue-200 shadow-[0_0_50px_rgba(73,139,199,.12)]"><UploadCloud className="h-9 w-9" strokeWidth={1.5} /></div>
                <h2 className="mt-7 text-2xl font-medium tracking-[-.025em] text-slate-50">Загрузите фото или видео</h2>
                <p className="mt-3 text-sm leading-6 text-slate-400">Фото появятся в ленте предпросмотра. Из видео можно выбрать точный кадр и затем откадрировать его.</p>
                <div className="mt-6 flex flex-wrap justify-center gap-3"><span className="rounded-full bg-slate-50 px-5 py-2.5 text-sm font-medium text-slate-950 transition group-hover:bg-white">Выбрать файл</span><span className="flex items-center gap-2 rounded-full border border-white/10 px-4 py-2.5 text-xs text-slate-300"><Film className="h-4 w-4" />MP4 · MOV · WebM · AVI</span></div>
              </div>
          </button>}
          <input ref={inputRef} type="file" accept={FILE_ACCEPT} multiple className="hidden" onChange={(event) => { acceptFiles(event.target.files); event.currentTarget.value = ""; }} />
          {error && <div className="mt-4 flex items-start gap-2 rounded-lg border border-red-400/20 bg-red-400/[.07] p-3 text-sm text-red-200"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{error}</div>}
          <SearchParameters className="mt-4 hidden xl:block" topK={topK} threshold={threshold} onTopKChange={onTopKChange} onThresholdChange={onThresholdChange} />
        </div>

        <aside className="flex flex-col rounded-xl border border-white/[.08] bg-[#07131f]/85 p-6">
          <p className="text-[10px] font-semibold uppercase tracking-[.2em] text-blue-200/60">Новый поиск</p>
          <h3 className="mt-3 text-2xl font-medium tracking-[-.03em]">Поиск по визуальным признакам</h3>
          <p className="mt-3 text-sm leading-6 text-slate-400">Номер автомобиля не используется. Модель сравнит форму кузова, цвет, оптику, колёса и другие устойчивые признаки.</p>
          <ol className="mt-8 space-y-5">
            {["Добавьте фото или выберите кадр из видео", "Выделите область с нужным автомобилем", "Запустите поиск по подготовленным кадрам"].map((text, index) => <li key={text} className="flex gap-3 text-sm text-slate-300"><span className="grid h-6 w-6 shrink-0 place-items-center rounded-full border border-blue-300/20 bg-blue-300/[.06] text-[10px] text-blue-200">{files.length > index ? <Check className="h-3 w-3" /> : index + 1}</span><span className="pt-0.5">{text}</span></li>)}
          </ol>
          <SearchParameters className="mt-7 xl:hidden" topK={topK} threshold={threshold} onTopKChange={onTopKChange} onThresholdChange={onThresholdChange} />
          <div className="mt-auto pt-8">
            <Button disabled={!files.length || loading} onClick={onSearch} className="h-12 w-full rounded-full bg-slate-50 text-slate-950 hover:bg-white">
              {loading ? <><Loader2 className="h-4 w-4 animate-spin" />Ищем совпадения…</> : <>Найти похожие <ArrowRight className="h-4 w-4" /></>}
            </Button>
            <p className="mt-3 text-center text-[11px] text-slate-600">POST /api/search · top {topK} · {threshold == null ? "порог сервера" : `порог ${threshold.toFixed(2)}`}</p>
          </div>
        </aside>
      </div>
    </section>
  );
}

type ViewMode = "attention" | "candidate" | "original";

function CompareViewer({ original, attentionImage, candidate, mode }: { original: string; attentionImage: string | null; candidate?: ApiCandidate; mode: ViewMode }) {
  const [split, setSplit] = useState(50);
  const overlay = mode === "attention" ? attentionImage : mode === "candidate" && candidate ? candidateFrame(candidate.gallery_id) : null;
  const overlayLabel = mode === "attention" ? "Внимание модели" : candidate ? `Кандидат #${candidate.gallery_id}` : "Кандидат";
  const unavailable = mode === "attention" && !attentionImage
    ? "Карта внимания недоступна: сервер не вернул ответ /api/explain"
    : mode === "candidate" && !candidate ? "Нет кандидата для сравнения" : null;
  return <div className="compare-viewer relative min-h-[420px] overflow-hidden rounded-xl border border-white/[.08] bg-[#07131f]">
    <img src={original} alt="Загруженный автомобиль" className="absolute inset-0 h-full w-full bg-[#050d16] object-contain" />
    {overlay && <div className="absolute inset-0 overflow-hidden bg-[#050d16]" style={{ clipPath: `inset(0 0 0 ${split}%)` }} aria-hidden="true">
      <img src={overlay} alt="" className="absolute inset-0 h-full w-full object-contain" />
    </div>}
    <span className="pointer-events-none absolute left-4 top-4 rounded-md border border-white/10 bg-[#07121e]/85 px-3 py-1.5 text-[11px] text-slate-200 backdrop-blur">{mode === "original" ? "Загруженный кадр" : "Оригинал"}</span>
    {overlay && <span className="pointer-events-none absolute right-4 top-4 max-w-[45%] truncate rounded-md border border-white/10 bg-[#07121e]/85 px-3 py-1.5 text-[11px] text-slate-200 backdrop-blur">{overlayLabel}</span>}
    {overlay && <><div className="pointer-events-none absolute inset-y-0 z-10 w-px bg-white/80 shadow-[0_0_18px_rgba(255,255,255,.45)]" style={{ left: `${split}%` }} /><div className="pointer-events-none absolute top-1/2 z-20 grid h-10 w-10 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-white/20 bg-slate-50 text-[#07121e] shadow-xl" style={{ left: `${split}%` }}><span className="font-mono text-xs">‹›</span></div><input type="range" min="0" max="100" value={split} onChange={(event) => setSplit(Number(event.target.value))} aria-label={`Сравнить оригинал и ${mode === "attention" ? "карту внимания модели" : "кандидата"}`} className="comparison-range absolute inset-0 z-30 h-full w-full cursor-ew-resize opacity-0" /></>}
    {unavailable && <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center px-4"><p className="rounded-full border border-amber-300/20 bg-[#07121e]/90 px-4 py-2 text-center text-xs text-amber-100/90">{unavailable}</p></div>}
  </div>;
}

function CandidateCard({ item, selected, onSelect }: { item: ApiCandidate; selected: boolean; onSelect: () => void }) {
  return <button type="button" onClick={onSelect} className={`group min-w-[210px] flex-1 overflow-hidden rounded-xl border bg-[#091624]/85 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 ${selected ? "border-blue-300/75 shadow-[0_0_0_2px_rgba(91,166,248,.15)]" : "border-white/[.08] hover:-translate-y-1 hover:border-blue-300/30"}`}>
    <div className="relative h-[106px] overflow-hidden bg-[#0a1723]"><img src={candidateImage(item.gallery_id)} alt="Автомобиль-кандидат" className="h-full w-full object-cover opacity-80 transition duration-500 group-hover:scale-105 group-hover:opacity-100" />{selected && <span className="absolute right-2 top-2 rounded-full bg-blue-300 px-2 py-1 text-[9px] font-semibold uppercase tracking-wider text-[#06111b]">Выбрано</span>}</div>
    <div className="p-3.5"><div className="flex items-center justify-between gap-3 text-sm font-medium text-slate-100"><span className="truncate">#{item.gallery_id}</span><span>{scorePercent(item.confidence)}</span></div><p className="mt-1.5 text-[12px] text-slate-400">Группа камер {item.camera_group}</p><p className={`mt-1 text-[11px] ${item.accepted ? "text-emerald-400" : "text-slate-600"}`}>{item.accepted ? "Выше порога" : "Ниже порога"}</p></div>
  </button>;
}

function Distribution({ result }: { result: SearchResponse }) {
  const counts = result.sim_hist?.counts ?? [];
  const max = Math.max(...counts, 1);
  const edges = result.sim_hist?.edges ?? [];
  const low = edges.length ? edges[0] : -0.2;
  const high = edges.length ? edges[edges.length - 1] : 1;
  const thresholdLeft = useMemo(() => {
    if (high <= low) return 50;
    return Math.max(0, Math.min(100, ((result.threshold - low) / (high - low)) * 100));
  }, [high, low, result.threshold]);
  if (!counts.length) return null;
  return <div className="border-t border-white/[.07] pt-5"><div className="mb-3 flex items-center justify-between text-[11px] uppercase tracking-[.08em] text-slate-400"><span>Распределение совпадений</span><span className="normal-case tracking-normal text-slate-500">Порог <strong className="text-slate-200">{result.threshold.toFixed(3)}</strong></span></div><div className="relative flex h-16 items-end gap-[2px] border-b border-white/10">{counts.map((count, index) => <span key={index} className="flex-1 rounded-t-[2px] bg-blue-500/45" style={{ height: `${Math.max(3, (Math.log1p(count) / Math.log1p(max)) * 58)}px` }} />)}<span className="absolute bottom-0 h-14 border-l border-dashed border-slate-300/70" style={{ left: `${thresholdLeft}%` }} /><span className="absolute -top-1 right-0 text-[11px] font-semibold text-slate-100">{scorePercent(result.sim_max)}</span></div><div className="mt-1 flex justify-between text-[10px] text-slate-600"><span>{low.toFixed(1)}</span><span>{high.toFixed(1)}</span></div></div>;
}

function exportResult(result: SearchResponse) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `falcon-search-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function MatchPanel({ candidate, result, comparing, onSelect, onCompare }: { candidate?: ApiCandidate; result: SearchResponse; comparing: boolean; onSelect: (id: string) => void; onCompare: () => void }) {
  const others = result.candidates
    .map((item, index) => ({ item, rank: index + 1 }))
    .filter(({ item }) => item.gallery_id !== candidate?.gallery_id)
    .slice(0, 2);
  const accepted = Boolean(candidate?.accepted && !result.refused);
  return <aside className="flex min-w-0 flex-col rounded-xl border border-white/[.08] bg-[#07131f]/90 p-5 lg:p-6">
    <div className="flex items-center justify-between gap-3"><div className={`flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[.12em] ${accepted ? "text-blue-200/75" : "text-amber-200/80"}`}><span className={`h-2 w-2 rounded-full ${accepted ? "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,.8)]" : "bg-amber-400"}`} />{accepted ? "Совпадение найдено" : "Уверенного совпадения нет"}</div><span className={`rounded-md border px-2.5 py-1 text-[10px] ${accepted ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-300" : "border-amber-400/20 bg-amber-400/10 text-amber-200"}`}>{accepted ? "Выше порога" : "Отказ"}</span></div>
    <p className="mt-3 text-5xl font-medium tracking-[-.05em] text-slate-50">{scorePercent(candidate?.confidence ?? result.sim_max)}</p>
    <dl className="mt-5 grid grid-cols-[120px_1fr] gap-x-4 gap-y-2 border-t border-white/[.07] pt-4 text-[13px]"><dt className="text-slate-500">ID галереи</dt><dd className="truncate font-medium text-slate-200">{candidate ? `#${candidate.gallery_id}` : "—"}</dd><dt className="text-slate-500">Камера-группа</dt><dd className="text-slate-200">{candidate?.camera_group ?? "—"}</dd><dt className="text-slate-500">Инференс</dt><dd className="text-slate-200">{result.inference_ms.toFixed(1)} мс</dd><dt className="text-slate-500">Размерность</dt><dd className="text-slate-200">{result.query_embedding_dim}</dd><dt className="text-slate-500">Порог</dt><dd className="text-slate-200">{result.threshold.toFixed(3)}</dd></dl>
    <div className="mt-5 border-t border-white/[.07] pt-4"><p className="mb-3 text-[11px] uppercase tracking-[.08em] text-slate-400">Другие кандидаты</p>{others.length ? others.map(({ item, rank }) => <button type="button" key={item.gallery_id} onClick={() => onSelect(item.gallery_id)} className="flex w-full items-center gap-3 rounded-md border-b border-white/[.06] py-2.5 text-left transition last:border-b-0 hover:bg-white/[.03] focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300"><img src={candidateImage(item.gallery_id)} alt="" className="h-12 w-16 rounded-md object-cover opacity-80" /><div className="min-w-0 flex-1"><p className="truncate text-xs font-medium text-slate-200">#{rank} · {item.gallery_id}</p><p className="mt-1 text-[10px] text-slate-500">Группа камер {item.camera_group}</p></div><span className="text-sm text-slate-200">{scorePercent(item.confidence)}</span></button>) : <p className="text-xs text-slate-500">Кандидатов нет</p>}</div>
    <div className="mt-auto pt-4"><Distribution result={result} /></div>
    <div className="mt-5 grid grid-cols-3 gap-2">
      {candidate
        ? <Button asChild className="h-11 bg-slate-50 px-3 text-xs text-slate-950 hover:bg-white"><a href={candidateFrame(candidate.gallery_id)} target="_blank" rel="noreferrer" title="Открыть кадр кандидата в новой вкладке"><ArrowUpRight className="h-4 w-4" /><span className="hidden sm:inline">Открыть</span></a></Button>
        : <Button disabled className="h-11 bg-slate-50 px-3 text-xs text-slate-950"><ArrowUpRight className="h-4 w-4" /><span className="hidden sm:inline">Открыть</span></Button>}
      <Button variant="outline" onClick={onCompare} disabled={!candidate} aria-pressed={comparing} title="Сравнить загруженный кадр с кандидатом шторкой" className={`h-11 px-3 text-xs ${comparing ? "border-blue-300/50 bg-blue-300/10 text-blue-100" : ""}`}><Columns2 className="h-4 w-4" /><span className="hidden sm:inline">Сравнить</span></Button>
      <Button variant="outline" onClick={() => exportResult(result)} title="Скачать ответ API в формате JSON" className="h-11 px-3 text-xs"><Download className="h-4 w-4" /><span className="hidden sm:inline">JSON</span></Button>
    </div>
  </aside>;
}

export default function SearchWorkspaceLive() {
  const [files, setFiles] = useState<File[]>([]);
  const [originalFiles, setOriginalFiles] = useState<File[]>([]);
  const [activeUploadIndex, setActiveUploadIndex] = useState(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [attentionImage, setAttentionImage] = useState<string | null>(null);
  const [view, setView] = useState<ViewMode>("attention");
  const [searchTopK, setSearchTopK] = useState(10);
  const [searchThreshold, setSearchThreshold] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");

  useEffect(() => {
    apiFetch("/api/health")
      .then(async (response) => {
        if (!response.ok) throw new Error(String(response.status));
        const health = await response.json() as { status?: string };
        setApiStatus(health.status === "ok" ? "online" : "offline");
      })
      .catch(() => setApiStatus("offline"));
  }, []);

  useEffect(() => {
    if (!files[0]) { setPreviewUrl(null); return; }
    const url = URL.createObjectURL(files[0]);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [files]);

  const addFiles = async (incoming: File[]) => {
    if (files.length >= MAX_FILES) { setError(`Можно загрузить не больше ${MAX_FILES} кадров — удалите лишний, чтобы добавить новый.`); return; }
    const converted = await Promise.all(incoming.map(normalizeImage));
    const images = converted.filter((file): file is File => Boolean(file));
    if (!images.length) { setError("Не удалось прочитать изображение. Поддерживаются JPG, PNG, WebP, BMP, GIF и AVIF."); return; }
    const next = [...files, ...images].slice(0, MAX_FILES);
    const nextOriginals = [...originalFiles, ...images].slice(0, MAX_FILES);
    setFiles(next);
    setOriginalFiles(nextOriginals);
    setActiveUploadIndex(Math.max(0, next.length - 1));
    setResult(null); setAttentionImage(null); setError(null);
  };

  const removeFile = (index: number) => {
    const next = files.filter((_, itemIndex) => itemIndex !== index);
    setFiles(next);
    setOriginalFiles((current) => current.filter((_, itemIndex) => itemIndex !== index));
    setActiveUploadIndex((current) => Math.max(0, Math.min(current > index ? current - 1 : current, next.length - 1)));
    setResult(null); setAttentionImage(null);
  };

  const replaceFile = (index: number, file: File) => {
    setFiles((current) => current.map((item, itemIndex) => itemIndex === index ? file : item));
    setError(null);
  };

  const undoCrop = (index: number) => {
    const original = originalFiles[index];
    if (!original) return;
    setFiles((current) => current.map((item, itemIndex) => itemIndex === index ? original : item));
    setError(null);
  };

  const runSearch = async () => {
    if (!files.length) return;
    setLoading(true); setError(null);
    try {
      const searchForm = new FormData();
      files.forEach((file) => searchForm.append("files", file));
      searchForm.append("top_k", String(searchTopK));
      if (searchThreshold != null) searchForm.append("threshold", String(searchThreshold));
      const explainForm = new FormData(); explainForm.append("file", files[0]);
      // Карта внимания необязательна: её сбой не должен ломать сам поиск.
      const explanation = apiFetch("/api/explain", { method: "POST", body: explainForm })
        .then((response) => (response.ok ? response.json() as Promise<{ image_base64?: string }> : null))
        .catch(() => null);
      const searchResponse = await apiFetch("/api/search", { method: "POST", body: searchForm });
      if (!searchResponse.ok) {
        const detail = await searchResponse.json().catch(() => null);
        const message = typeof detail?.detail === "string" ? detail.detail : Array.isArray(detail?.detail) ? detail.detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join("; ") : "";
        throw new Error(message || `Ошибка API ${searchResponse.status}`);
      }
      const data = await searchResponse.json() as SearchResponse;
      const attention = await explanation;
      setResult(data);
      setSelectedId(data.candidates[0]?.gallery_id ?? null);
      setAttentionImage(attention?.image_base64 ? `data:image/png;base64,${attention.image_base64}` : null);
      setView(attention?.image_base64 ? "attention" : "candidate");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Не удалось выполнить поиск. Проверьте соединение с API.");
    } finally { setLoading(false); }
  };

  const newSearch = () => { setResult(null); setSelectedId(null); setAttentionImage(null); setError(null); setView("attention"); };
  const candidate = result?.candidates.find((item) => item.gallery_id === selectedId) ?? result?.candidates[0];

  return <div className="app-shell min-h-screen bg-[#050d16] text-slate-100"><AppHeader apiStatus={apiStatus} /><main className="relative mx-auto max-w-[1580px] px-5 py-8 lg:px-8 lg:py-10"><div className="app-orb app-orb-one" aria-hidden="true" /><div className="app-orb app-orb-two" aria-hidden="true" />
    <section className="relative z-10 mb-7 flex flex-col justify-between gap-5 md:flex-row md:items-end"><div><div className="mb-3 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[.24em] text-blue-200/60"><ScanSearch className="h-3.5 w-3.5" />Компьютерное зрение · ReID</div><h1 className="text-3xl font-medium tracking-[-.035em] text-slate-50 md:text-[38px]">{result ? "Результаты поиска" : "Поиск автомобиля"}</h1><p className="mt-2 max-w-xl text-sm text-slate-400">{result ? `${result.message} · обработано изображений: ${result.n_query_images}` : "Загрузите фотографию — модель найдёт максимально похожие автомобили в галерее."}</p></div>
      {result && <div className="flex flex-wrap items-center gap-2"><div className="flex items-center gap-1 rounded-full border border-blue-200/[.12] bg-[#07131f]/80 p-1" role="group" aria-label="Что показать справа от шторки">{([["attention", "Внимание модели"], ["candidate", "С кандидатом"], ["original", "Скрыть"]] as const).map(([value, label]) => <button type="button" key={value} onClick={() => setView(value)} aria-pressed={view === value} className={`flex items-center gap-2 rounded-full px-3.5 py-2 text-xs transition ${view === value ? "bg-blue-300/10 text-blue-100" : "text-slate-500 hover:text-slate-300"}`}>{value === "attention" && <span className={`h-2 w-2 rounded-full ${attentionImage ? "bg-emerald-400" : "bg-amber-400"}`} />}{label}</button>)}</div><Button variant="outline" onClick={newSearch} className="rounded-full"><ImagePlus className="h-4 w-4" />Новый поиск</Button></div>}
    </section>
    {!result ? <UploadWorkspace files={files} originalFiles={originalFiles} activeIndex={activeUploadIndex} loading={loading} error={error} topK={searchTopK} threshold={searchThreshold} onFiles={addFiles} onRemove={removeFile} onReplace={replaceFile} onUndoCrop={undoCrop} onActiveIndex={setActiveUploadIndex} onTopKChange={setSearchTopK} onThresholdChange={setSearchThreshold} onSearch={runSearch} /> : <section className="relative z-10 rounded-2xl border border-blue-300/20 bg-[#06111b]/75 p-3 shadow-[0_30px_100px_rgba(0,0,0,.42)] backdrop-blur md:p-4"><div className="grid gap-4 xl:grid-cols-[1.55fr_1fr]"><div className="min-w-0">{result.watchlist_alerts?.length > 0 && <div className="mb-3 flex flex-wrap items-center gap-2 rounded-xl border border-red-400/25 bg-red-500/[.08] px-4 py-3 text-sm text-red-100"><Siren className="h-4 w-4 shrink-0 text-red-300" /><span className="font-medium">Совпадение со списком наблюдения:</span>{result.watchlist_alerts.map((alert) => <span key={alert.id} className="rounded-md border border-red-300/20 bg-red-400/10 px-2 py-0.5 text-xs">{alert.name} · {scorePercent(alert.confidence)}</span>)}</div>}{previewUrl && <CompareViewer original={previewUrl} attentionImage={attentionImage} candidate={candidate} mode={view} />}<div className="mb-3 mt-4 flex items-center justify-between gap-3"><h2 className="text-sm font-medium text-slate-200">Кандидаты на совпадение <span className="text-slate-600">({result.candidates.length})</span></h2><div className="hidden items-center gap-2 text-[11px] text-slate-500 sm:flex">Сортировка: <span className="rounded-md border border-white/[.08] bg-white/[.025] px-3 py-2 text-slate-300">Сходство</span></div></div>{result.candidates.length ? <div className="flex gap-3 overflow-x-auto pb-1">{result.candidates.map((item) => <CandidateCard key={item.gallery_id} item={item} selected={item.gallery_id === candidate?.gallery_id} onSelect={() => setSelectedId(item.gallery_id)} />)}</div> : <p className="rounded-xl border border-dashed border-white/10 px-4 py-8 text-center text-sm text-slate-500">В галерее не нашлось ни одного кандидата.</p>}</div><MatchPanel candidate={candidate} result={result} comparing={view === "candidate"} onSelect={setSelectedId} onCompare={() => setView((current) => current === "candidate" ? "attention" : "candidate")} /></div></section>}
    <div className="relative z-10 mt-4 flex items-center justify-between text-[11px] text-slate-600"><span>FALCON ReID · реальные данные из /api/search</span><a href="/docs" className="flex items-center gap-2 hover:text-slate-300"><CircleHelp className="h-3.5 w-3.5" />Документация API</a></div>
  </main></div>;
}
