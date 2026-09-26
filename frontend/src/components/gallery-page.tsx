import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  ArrowDown,
  ArrowUpRight,
  Check,
  ChevronDown,
  CircleHelp,
  Database,
  Copy,
  GalleryHorizontalEnd,
  HardDrive,
  Images,
  LayoutGrid,
  Loader2,
  MapPin,
  RotateCcw,
  Search,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { AppHeader } from "@/components/search-workspace-live";
import { Button } from "@/components/ui/button";
import { apiFetch, apiUrl } from "@/lib/api";

type ApiStatus = "checking" | "online" | "offline";

type GalleryItem = {
  gallery_id: string;
  camera_group: number;
};

type GalleryResponse = {
  total: number;
  offset: number;
  items: GalleryItem[];
};

type LocationStat = {
  camera_group: number;
  count: number;
};

type AssetMeta = {
  bytes: number;
  width: number;
  height: number;
};

const thumbUrl = (id: string) =>
  apiUrl(`/api/gallery/${encodeURIComponent(id)}/thumb`);
const frameUrl = (id: string) =>
  apiUrl(`/api/gallery/${encodeURIComponent(id)}/frame`);

const pageSizes = [10, 20, 50, 100];

function formatBytes(bytes?: number) {
  if (bytes == null) return "считаем…";
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

function matchSize(bytes: number | undefined, filter: string) {
  if (filter === "all") return true;
  if (bytes == null) return false;
  if (filter === "small") return bytes < 100 * 1024;
  if (filter === "medium") return bytes >= 100 * 1024 && bytes < 300 * 1024;
  return bytes >= 300 * 1024;
}

function matchOrientation(meta: AssetMeta | undefined, filter: string) {
  if (filter === "all") return true;
  if (!meta?.width || !meta.height) return false;
  const ratio = meta.width / meta.height;
  if (filter === "landscape") return ratio > 1.08;
  if (filter === "portrait") return ratio < 0.92;
  return ratio >= 0.92 && ratio <= 1.08;
}

function SelectField({
  label,
  value,
  onChange,
  children,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
}) {
  return (
    <label className="group relative flex min-w-[150px] flex-1 flex-col gap-1.5">
      <span className="text-[10px] font-bold uppercase tracking-[.1em] text-slate-600">
        {label}
      </span>
      <span className="relative">
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="h-10 w-full appearance-none rounded-lg border border-white/[.08] bg-[#07131f] px-3 pr-9 text-xs text-slate-300 outline-none transition focus:border-blue-300/40 focus:ring-2 focus:ring-blue-300/10"
        >
          {children}
        </select>
        <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-600 transition group-focus-within:text-blue-200" />
      </span>
    </label>
  );
}

function GallerySkeleton() {
  return (
    <div className="overflow-hidden rounded-xl border border-white/[.06] bg-[#07131f]/70">
      <div className="aspect-[4/3] animate-pulse bg-gradient-to-br from-white/[.04] via-blue-300/[.06] to-white/[.025]" />
      <div className="space-y-3 p-4">
        <div className="h-3 w-3/5 animate-pulse rounded bg-white/[.07]" />
        <div className="h-2.5 w-2/5 animate-pulse rounded bg-white/[.04]" />
      </div>
    </div>
  );
}

function GalleryCard({
  item,
  meta,
  index,
}: {
  item: GalleryItem;
  meta?: AssetMeta;
  index: number;
}) {
  const [copied, setCopied] = useState(false);
  const copyName = async () => {
    try {
      await navigator.clipboard.writeText(item.gallery_id);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <article className="relative overflow-hidden rounded-xl border border-white/[.075] bg-[#07131f]/85">
      <a
        href={frameUrl(item.gallery_id)}
        target="_blank"
        rel="noreferrer"
        className="group block focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-300"
        aria-label={`Открыть фотографию ${item.gallery_id}`}
      >
        <div className="relative aspect-[4/3] overflow-hidden bg-[#040b12]">
          <img
            src={thumbUrl(item.gallery_id)}
            alt={`Автомобиль ${item.gallery_id}`}
            loading="lazy"
            className="h-full w-full object-cover opacity-90 transition duration-500 group-hover:scale-[1.035] group-hover:opacity-100"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#050d16]/90 via-transparent to-[#050d16]/10" />
          <span className="absolute left-3 top-3 rounded-md border border-white/[.09] bg-[#050d16]/75 px-2 py-1 font-mono text-[9px] text-slate-400 backdrop-blur">
            {String(index + 1).padStart(3, "0")}
          </span>
          <span className="absolute bottom-3 right-3 grid h-9 w-9 translate-y-2 place-items-center rounded-full border border-white/10 bg-[#07131f]/85 text-blue-100 opacity-0 backdrop-blur transition group-hover:translate-y-0 group-hover:opacity-100">
            <ArrowUpRight className="h-4 w-4" />
          </span>
        </div>
      </a>
      <div className="p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <button type="button" onClick={() => void copyName()} title="Скопировать название" className="group/copy flex max-w-full items-center gap-2 text-left font-mono text-[12px] font-medium text-slate-200 transition hover:text-blue-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300">
                <span className="truncate">{item.gallery_id}</span>
                {copied ? <Check className="h-3.5 w-3.5 shrink-0 text-emerald-300" /> : <Copy className="h-3.5 w-3.5 shrink-0 text-slate-600 transition group-hover/copy:text-blue-300" />}
              </button>
              <p className="mt-1.5 flex items-center gap-1.5 text-[10px] text-slate-500">
                <MapPin className="h-3 w-3" />
                Камера-группа {item.camera_group}
              </p>
            </div>
            {copied && <span className="shrink-0 text-xs text-emerald-300">Скопировано</span>}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2 border-t border-white/[.06] pt-3 text-[10px]">
            <span className="flex items-center gap-1.5 text-slate-600">
              <HardDrive className="h-3 w-3" />
              {formatBytes(meta?.bytes)}
            </span>
            <span className="text-right font-mono text-slate-500">
              {meta?.width ? `${meta.width}×${meta.height}` : "—×—"}
            </span>
          </div>
      </div>
    </article>
  );
}

export default function GalleryPage() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [items, setItems] = useState<GalleryItem[]>([]);
  const [locations, setLocations] = useState<LocationStat[]>([]);
  const [total, setTotal] = useState(0);
  const [pageSize, setPageSize] = useState(20);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nameQuery, setNameQuery] = useState("");
  const [cameraGroup, setCameraGroup] = useState("all");
  const [sizeFilter, setSizeFilter] = useState("all");
  const [orientation, setOrientation] = useState("all");
  const [sortBy, setSortBy] = useState("index");
  const [metadata, setMetadata] = useState<Record<string, AssetMeta>>({});
  const requestVersion = useRef(0);

  useEffect(() => {
    apiFetch("/api/health")
      .then((response) => {
        if (!response.ok) throw new Error(String(response.status));
        setApiStatus("online");
      })
      .catch(() => setApiStatus("offline"));
    apiFetch("/api/stats/locations")
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then((data: { locations: LocationStat[] }) =>
        setLocations(data.locations ?? []),
      )
      .catch(() => undefined);
  }, []);

  const loadPage = useCallback(
    async (offset: number, replace: boolean) => {
      const version = replace
        ? ++requestVersion.current
        : requestVersion.current;
      setLoading(true);
      setError(null);
      const params = new URLSearchParams({
        offset: String(offset),
        limit: String(pageSize),
      });
      if (cameraGroup !== "all") params.set("group", cameraGroup);
      try {
        const response = await apiFetch(`/api/gallery/list?${params}`);
        if (!response.ok)
          throw new Error(
            response.status >= 500
              ? `Сервер галереи недоступен (ошибка ${response.status}). Проверьте, что API запущен.`
              : `Не удалось загрузить галерею (ошибка ${response.status}).`,
          );
        const data = (await response.json()) as GalleryResponse;
        if (version !== requestVersion.current) return;
        setItems((current) =>
          replace
            ? data.items
            : [
                ...current,
                ...data.items.filter(
                  (next) =>
                    !current.some(
                      (item) => item.gallery_id === next.gallery_id,
                    ),
                ),
              ],
        );
        setTotal(data.total);
      } catch (cause) {
        if (version !== requestVersion.current) return;
        setError(
          cause instanceof TypeError
            ? "Нет соединения с API галереи."
            : cause instanceof Error
              ? cause.message
              : "Не удалось загрузить галерею",
        );
      } finally {
        if (version === requestVersion.current) setLoading(false);
      }
    },
    [cameraGroup, pageSize],
  );

  useEffect(() => {
    void loadPage(0, true);
  }, [loadPage]);

  const sourceKey = items.map((item) => item.gallery_id).join("|");

  useEffect(() => {
    let cancelled = false;
    const missing = items.filter((item) => !metadata[item.gallery_id]);
    if (!missing.length) return;
    const inspect = async (item: GalleryItem) => {
      try {
        const response = await apiFetch(`/api/gallery/${encodeURIComponent(item.gallery_id)}/thumb`);
        if (!response.ok) return;
        const blob = await response.blob();
        const bitmap = await createImageBitmap(blob);
        const next = {
          bytes: blob.size,
          width: bitmap.width,
          height: bitmap.height,
        };
        bitmap.close();
        if (!cancelled)
          setMetadata((current) => ({ ...current, [item.gallery_id]: next }));
      } catch {
        // Метаданные необязательны: карточка остаётся доступной даже без них.
      }
    };
    const batches = Array.from(
      { length: Math.ceil(missing.length / 6) },
      (_, index) => missing.slice(index * 6, index * 6 + 6),
    );
    void (async () => {
      for (const batch of batches) {
        if (cancelled) break;
        await Promise.all(batch.map(inspect));
      }
    })();
    return () => {
      cancelled = true;
    };
    // sourceKey intentionally tracks only the ordered set of gallery ids.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey]);

  const filteredItems = useMemo(() => {
    const query = nameQuery.trim().toLocaleLowerCase("ru");
    const filtered = items.filter((item) => {
      const meta = metadata[item.gallery_id];
      return (
        (!query || item.gallery_id.toLocaleLowerCase("ru").includes(query)) &&
        (cameraGroup === "all" || item.camera_group === Number(cameraGroup)) &&
        matchSize(meta?.bytes, sizeFilter) &&
        matchOrientation(meta, orientation)
      );
    });
    return [...filtered].sort((a, b) => {
      if (sortBy === "name")
        return a.gallery_id.localeCompare(b.gallery_id, "ru");
      if (sortBy === "size")
        return (
          (metadata[b.gallery_id]?.bytes ?? -1) -
          (metadata[a.gallery_id]?.bytes ?? -1)
        );
      return 0;
    });
  }, [
    cameraGroup,
    items,
    metadata,
    nameQuery,
    orientation,
    sizeFilter,
    sortBy,
  ]);

  const resetFilters = () => {
    setNameQuery("");
    setCameraGroup("all");
    setSizeFilter("all");
    setOrientation("all");
    setSortBy("index");
  };

  const hasMore = items.length < total;

  return (
    <div className="app-shell min-h-screen bg-[#050d16] text-slate-100">
      <AppHeader apiStatus={apiStatus} active="Галерея" />
      <main className="relative mx-auto max-w-[1580px] px-5 py-8 lg:px-8 lg:py-10">
        <div className="app-orb app-orb-one" aria-hidden="true" />
        <div className="app-orb app-orb-two" aria-hidden="true" />

        <section className="relative z-10 mb-7 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div>
            <div className="mb-3 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[.24em] text-blue-200/60">
              <GalleryHorizontalEnd className="h-3.5 w-3.5" />
              Объекты наблюдения · Галерея
            </div>
            <h1 className="text-3xl font-medium tracking-[-.035em] text-slate-50 md:text-[38px]">
              Галерея автомобилей
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
              Просматривайте фотографии из индекса последовательно и уточняйте
              выборку по параметрам объектов.
            </p>
          </div>
          <div className="grid grid-cols-3 gap-px overflow-hidden rounded-xl border border-white/[.08] bg-white/[.08] lg:min-w-[420px]">
            <div className="bg-[#07131f]/95 px-4 py-3">
              <p className="text-[9px] uppercase font-bold tracking-[.14em] text-slate-600">
                В индексе
              </p>
              <p className="mt-1 font-mono text-lg text-slate-100">
                {total || "—"}
              </p>
            </div>
            <div className="bg-[#07131f]/95 px-4 py-3">
              <p className="text-[9px] uppercase font-bold tracking-[.14em] text-slate-600">
                Загружено
              </p>
              <p className="mt-1 font-mono text-lg text-slate-100">
                {items.length}
              </p>
            </div>
            <div className="bg-[#07131f]/95 px-4 py-3">
              <p className="text-[9px] uppercase font-bold tracking-[.14em] text-slate-600">
                В выборке
              </p>
              <p className="mt-1 font-mono text-lg text-blue-200">
                {filteredItems.length}
              </p>
            </div>
          </div>
        </section>

        <section className="relative z-10 mb-5 overflow-hidden rounded-2xl border border-blue-300/20 bg-[#06111b]/80 p-4 shadow-[0_30px_100px_rgba(0,0,0,.32)] backdrop-blur md:p-5">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <span className="grid h-9 w-9 place-items-center rounded-lg bg-blue-300/10 text-blue-200">
                <SlidersHorizontal className="h-4 w-4" />
              </span>
              <div>
                <h2 className="text-sm font-medium text-slate-100">
                  Фильтры каталога
                </h2>
                <p className="mt-0.5 text-[12px] text-slate-600">
                  Применяются сразу к загруженной выборке
                </p>
              </div>
            </div>
            <button
              onClick={resetFilters}
              className="flex items-center gap-2 rounded-lg px-3 py-2 text-[11px] text-slate-500 transition hover:bg-white/[.04] hover:text-slate-200"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Сбросить
            </button>
          </div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-[1.5fr_1fr_1fr_1fr]">
            <label className="relative flex flex-col gap-1.5">
              <span className="text-[10px] font-bold uppercase tracking-[.1em] text-slate-600">
                Имя или ID
              </span>
              <span className="relative">
                <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-600" />
                <input
                  value={nameQuery}
                  onChange={(event) => setNameQuery(event.target.value)}
                  placeholder="Например, 0ad7…"
                  className="h-10 w-full rounded-lg border border-white/[.08] bg-[#07131f] pl-9 pr-8 text-xs text-slate-200 outline-none placeholder:text-slate-700 focus:border-blue-300/40 focus:ring-2 focus:ring-blue-300/10"
                />
                {nameQuery && (
                  <button
                    onClick={() => setNameQuery("")}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-600 hover:text-slate-200"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </span>
            </label>
            <SelectField
              label="Камера"
              value={cameraGroup}
              onChange={setCameraGroup}
            >
              <option value="all">Все группы</option>
              {locations.map((location) => (
                <option
                  key={location.camera_group}
                  value={location.camera_group}
                >
                  Группа {location.camera_group} · {location.count}
                </option>
              ))}
            </SelectField>
            <SelectField
              label="Размер превью"
              value={sizeFilter}
              onChange={setSizeFilter}
            >
              <option value="all">Любой</option>
              <option value="small">До 100 КБ</option>
              <option value="medium">100–300 КБ</option>
              <option value="large">От 300 КБ</option>
            </SelectField>
            <SelectField
              label="Ориентация"
              value={orientation}
              onChange={setOrientation}
            >
              <option value="all">Любая</option>
              <option value="landscape">Альбомная</option>
              <option value="portrait">Портретная</option>
              <option value="square">Квадратная</option>
            </SelectField>
          </div>
        </section>

        <section className="relative z-10">
          <div className="mb-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
            <div className="flex items-center gap-3">
              <span className="grid h-9 w-9 place-items-center rounded-lg border border-white/[.07] bg-white/[.025] text-slate-400">
                <LayoutGrid className="h-4 w-4" />
              </span>
              <div>
                <h2 className="text-sm font-medium text-slate-200">
                  Объекты галереи
                </h2>
                <p className="mt-0.5 text-[10px] text-slate-600">
                  Показано {filteredItems.length} из {total || items.length}
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-end gap-3">
              <SelectField
                label="Сортировка"
                value={sortBy}
                onChange={setSortBy}
              >
                <option value="index">По порядку</option>
                <option value="name">По имени</option>
                <option value="size">По размеру</option>
              </SelectField>
              <SelectField
                label="Размер порции"
                value={String(pageSize)}
                onChange={(value) => setPageSize(Number(value))}
              >
                {pageSizes.map((size) => (
                  <option key={size} value={size}>
                    {size} фотографий
                  </option>
                ))}
              </SelectField>
            </div>
          </div>

          {error && (
            <div className="mb-4 flex items-center justify-between gap-4 rounded-xl border border-red-300/15 bg-red-400/[.06] px-4 py-3 text-xs text-red-100">
              <span>{error}</span>
              <span className="flex shrink-0 items-center gap-2">
                <button
                  type="button"
                  onClick={() => void loadPage(items.length, items.length === 0)}
                  className="rounded-md border border-red-300/20 px-2.5 py-1 text-[11px] hover:bg-red-400/10"
                >
                  Повторить
                </button>
                <button
                  type="button"
                  onClick={() => setError(null)}
                  aria-label="Скрыть ошибку"
                >
                  <X className="h-4 w-4" />
                </button>
              </span>
            </div>
          )}

          {filteredItems.length > 0 ? (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
              {filteredItems.map((item, index) => (
                <GalleryCard
                  key={item.gallery_id}
                  item={item}
                  meta={metadata[item.gallery_id]}
                  index={index}
                />
              ))}
              {loading &&
                Array.from({ length: Math.min(pageSize, 8) }, (_, index) => (
                  <GallerySkeleton key={`skeleton-${index}`} />
                ))}
            </div>
          ) : loading ? (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
              {Array.from({ length: Math.min(pageSize, 10) }, (_, index) => (
                <GallerySkeleton key={index} />
              ))}
            </div>
          ) : (
            <div className="grid min-h-[340px] place-items-center rounded-2xl border border-dashed border-white/[.09] bg-[#06111b]/45 px-6 text-center">
              <div className="max-w-sm">
                <span className="mx-auto grid h-16 w-16 place-items-center rounded-2xl border border-blue-300/15 bg-blue-300/[.06] text-blue-200">
                  <Images className="h-7 w-7" />
                </span>
                <h3 className="mt-5 text-lg font-medium text-slate-100">
                  Ничего не найдено
                </h3>
                <p className="mt-2 text-sm leading-6 text-slate-500">
                  Измените параметры фильтрации, чтобы увидеть другие
                  фотографии.
                </p>
                <Button
                  variant="outline"
                  onClick={resetFilters}
                  className="mt-5 rounded-full"
                >
                  <RotateCcw className="h-4 w-4" />
                  Сбросить фильтры
                </Button>
              </div>
            </div>
          )}

          {(items.length > 0 || total > 0) && (
            <div className="mt-8 flex flex-col items-center justify-between gap-4 rounded-xl border border-white/[.07] bg-[#06111b]/65 p-4 sm:flex-row">
              <div>
                <p className="text-xs font-medium text-slate-300">
                  Загружено {items.length} из {total}
                </p>
                <p className="mt-1 text-[10px] text-slate-600">
                  Следующая порция продолжит текущую последовательность
                </p>
              </div>
              {hasMore ? (
                <Button
                  variant="outline"
                  onClick={() => void loadPage(items.length, false)}
                  disabled={loading}
                  className="h-12 rounded-full border-blue-300/20 bg-[#07131f]/80 px-7 text-blue-100 hover:bg-blue-300/10"
                >
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <ArrowDown className="h-4 w-4" />
                  )}
                  Загрузить ещё {Math.min(pageSize, total - items.length)}
                </Button>
              ) : (
                <span className="flex h-12 items-center rounded-full border border-emerald-300/15 bg-emerald-400/[.06] px-5 text-xs text-emerald-200">
                  Все фотографии загружены
                </span>
              )}
            </div>
          )}
        </section>

        <footer className="relative z-10 mt-8 flex flex-col justify-between gap-3 border-t border-white/[.06] pt-4 text-[10px] text-slate-600 sm:flex-row sm:items-center">
          <span className="flex items-center gap-2">
            <Database className="h-3.5 w-3.5" />
            FALCON ReID · индекс галереи
          </span>
          <a
            href="/docs"
            className="flex items-center gap-2 transition hover:text-slate-300"
          >
            <CircleHelp className="h-3.5 w-3.5" />
            Документация API
          </a>
        </footer>
      </main>
    </div>
  );
}
