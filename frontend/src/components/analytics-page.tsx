import { useCallback, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  BarChart3,
  Camera,
  CheckCircle2,
  CircleHelp,
  Clock3,
  Database,
  Gauge,
  History,
  Images,
  Loader2,
  MapPinned,
  Network,
  RefreshCw,
  Route,
  Search,
  ShieldAlert,
  TrendingUp,
} from "lucide-react";
import { AppHeader } from "@/components/search-workspace-live";
import { Button } from "@/components/ui/button";
import { CountUp } from "@/components/count-up";
import { apiFetch, apiUrl, readApiError } from "@/lib/api";

type ApiStatus = "checking" | "online" | "offline";
type LocationStat = { camera_group: number; count: number };
type LocationsResponse = { locations: LocationStat[]; total: number };
type UsageRecord = { t: number; ms: number; refused: boolean; accepted: number; n_images: number; alerts: number };
type UsageStats = { requests: number; avg_ms: number | null; p95_ms: number | null; refusal_rate: number | null; alerts_total: number; recent: UsageRecord[] };
type UserStats = Omit<UsageStats, "recent"> & { top_cameras: LocationStat[] };
type TrackItem = { gallery_id: string; camera_group: number };
type VehicleTrack = { size: number; locations: number[]; items: TrackItem[] };
type TracksResponse = { threshold: number; n_tracks: number; tracks: VehicleTrack[] };
type Candidate = { gallery_id: string; confidence: number; camera_group: number };
type HistoryItem = { id: string; refused: boolean; n_accepted: number; inference_ms: number; n_images: number; alerts: number; top3: Candidate[]; created_at: number };

type DashboardData = {
  locations: LocationsResponse;
  tracks: TracksResponse;
  usage: UsageStats;
  my: UserStats;
  history: HistoryItem[];
};

const emptyData: DashboardData = {
  locations: { locations: [], total: 0 },
  tracks: { threshold: 0, n_tracks: 0, tracks: [] },
  usage: { requests: 0, avg_ms: null, p95_ms: null, refusal_rate: null, alerts_total: 0, recent: [] },
  my: { requests: 0, avg_ms: null, p95_ms: null, refusal_rate: null, alerts_total: 0, top_cameras: [] },
  history: [],
};

const number = new Intl.NumberFormat("ru-RU");
const formatNumber = (value: number | null | undefined) => value == null ? "—" : number.format(value);
const formatMs = (value: number | null | undefined) => value == null ? "—" : value >= 1000 ? `${(value / 1000).toFixed(2)} с` : `${Math.round(value)} мс`;
const formatPercent = (value: number | null | undefined) => value == null ? "—" : `${Math.round(value * 100)}%`;
const formatDate = (value: number) => new Intl.DateTimeFormat("ru-RU", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }).format(new Date(value * 1000));
const thumbUrl = (id: string) => apiUrl(`/api/gallery/${encodeURIComponent(id)}/thumb`);

function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`relative overflow-hidden rounded-2xl border border-white/[.075] bg-[#06111b]/78 shadow-[0_22px_70px_rgba(0,0,0,.22)] backdrop-blur ${className}`}>{children}</section>;
}

function MetricCard({ label, value, detail, icon: Icon, accent = false }: { label: string; value: string; detail: string; icon: typeof Activity; accent?: boolean }) {
  return (
    <div className={`group relative overflow-hidden rounded-2xl border p-5 transition hover:-translate-y-0.5 ${accent ? "border-blue-300/20 bg-blue-300/[.07]" : "border-white/[.075] bg-[#07131f]/85"}`}>
      <div className={`absolute -right-7 -top-7 h-24 w-24 rounded-full blur-2xl ${accent ? "bg-blue-300/15" : "bg-slate-300/[.04]"}`} />
      <div className="relative flex items-start justify-between gap-3">
        <div>
          <p className="text-[9px] font-bold uppercase tracking-[.16em] text-slate-600">{label}</p>
          <p className={`mt-3 font-mono text-[28px] leading-none tracking-[-.04em] ${accent ? "text-blue-100" : "text-slate-100"}`}><CountUp value={value} /></p>
          <p className="mt-3 text-[11px] text-slate-500">{detail}</p>
        </div>
        <span className={`grid h-10 w-10 shrink-0 place-items-center rounded-xl border ${accent ? "border-blue-200/15 bg-blue-300/10 text-blue-200" : "border-white/[.07] bg-white/[.025] text-slate-500"}`}><Icon className="h-4 w-4" /></span>
      </div>
    </div>
  );
}

function ActivityChart({ records }: { records: UsageRecord[] }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [pinnedIndex, setPinnedIndex] = useState<number | null>(null);
  const chart = useMemo(() => {
    const ordered = [...records].sort((a, b) => a.t - b.t);
    if (!ordered.length) return { points: [], ticks: [] };
    const values = ordered.map((item) => item.ms);
    const rawMin = Math.min(...values);
    const rawMax = Math.max(...values);
    const padding = Math.max((rawMax - rawMin) * 0.15, rawMax * 0.04, 1);
    const min = Math.max(0, rawMin - padding);
    const max = rawMax + padding;
    const span = Math.max(max - min, 1);
    const points = ordered.map((item, index) => ({
      ...item,
      x: ordered.length === 1 ? 356 : 72 + (index / (ordered.length - 1)) * 568,
      y: 150 - ((item.ms - min) / span) * 116,
    }));
    const ticks = [max, min + span / 2, min].map((value) => ({ value, y: 150 - ((value - min) / span) * 116 }));
    return { points, ticks };
  }, [records]);

  const points = chart.points;

  if (!points.length) {
    return <div className="grid h-[220px] place-items-center rounded-xl border border-dashed border-white/[.08] bg-white/[.015] text-center"><div><Activity className="mx-auto h-6 w-6 text-slate-700" /><p className="mt-3 text-xs text-slate-600">График появится после первого поиска</p></div></div>;
  }

  const line = points.map((point) => `${point.x},${point.y}`).join(" ");
  const area = `72,150 ${line} 640,150`;
  const activeIndex = pinnedIndex ?? hoveredIndex ?? points.length - 1;
  const active = points[activeIndex];
  return (
    <div className="relative overflow-hidden rounded-xl border border-white/[.06] bg-[#050d16]/55 px-2 pb-3 pt-3">
      <svg viewBox="0 0 680 215" className="h-[215px] w-full" role="img" aria-label="График времени инференса по времени выполнения запроса">
        <defs>
          <linearGradient id="activity-area" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#7db8ff" stopOpacity=".26" /><stop offset="100%" stopColor="#7db8ff" stopOpacity="0" /></linearGradient>
          <filter id="activity-glow"><feGaussianBlur stdDeviation="3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        {chart.ticks.map((tick) => <g key={tick.y}><line x1="72" y1={tick.y} x2="640" y2={tick.y} stroke="rgba(148,163,184,.10)" strokeDasharray="3 7" /><text x="62" y={tick.y + 4} textAnchor="end" fill="rgba(148,163,184,.55)" fontSize="12">{formatMs(tick.value)}</text></g>)}
        <line x1="72" y1="26" x2="72" y2="150" stroke="rgba(148,163,184,.32)" />
        <line x1="72" y1="150" x2="640" y2="150" stroke="rgba(148,163,184,.32)" />
        {[0, Math.floor((points.length - 1) / 2), points.length - 1].filter((value, index, items) => items.indexOf(value) === index).map((index) => <g key={`x-${index}`}><line x1={points[index].x} y1="150" x2={points[index].x} y2="156" stroke="rgba(148,163,184,.35)" /><text x={points[index].x} y="172" textAnchor="middle" fill="rgba(148,163,184,.55)" fontSize="12">{new Intl.DateTimeFormat("ru-RU", { hour: "2-digit", minute: "2-digit" }).format(new Date(points[index].t * 1000))}</text></g>)}
        <text x="356" y="207" textAnchor="middle" fill="rgba(148,163,184,.65)" fontSize="12">Время запроса</text>
        <text x="16" y="92" textAnchor="middle" fill="rgba(148,163,184,.65)" fontSize="12" transform="rotate(-90 16 92)">Длительность инференса</text>
        <polygon points={area} fill="url(#activity-area)" className="am-area" />
        {points.length > 1 && <polyline points={line} pathLength={1} className="am-draw" fill="none" stroke="#7db8ff" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" filter="url(#activity-glow)" />}
        {active && <line x1={active.x} y1="26" x2={active.x} y2="150" stroke="rgba(147,197,253,.28)" strokeDasharray="4 5" />}
        {points.map((point, index) => {
          const selected = index === activeIndex;
          return <g
            key={`${point.t}-${index}`}
            role="button"
            tabIndex={0}
            aria-label={`${formatDate(point.t)}: ${formatMs(point.ms)}, ${point.refused ? "отказ" : "совпадение"}`}
            className="am-point cursor-pointer outline-none"
            style={{ "--i": index } as CSSProperties}
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
            onFocus={() => setHoveredIndex(index)}
            onBlur={() => setHoveredIndex(null)}
            onClick={() => setPinnedIndex((current) => current === index ? null : index)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setPinnedIndex((current) => current === index ? null : index);
              }
            }}
          >
            <circle cx={point.x} cy={point.y} r={selected ? 12 : 9} fill={selected ? "rgba(125,184,255,.18)" : "rgba(125,184,255,.08)"} className="transition-all" />
            <circle cx={point.x} cy={point.y} r={selected ? 5 : 3} fill={point.refused ? "#fb7185" : "#93c5fd"} />
            <circle cx={point.x} cy={point.y} r="18" fill="transparent" />
          </g>;
        })}
      </svg>
      <p className="-mt-1 text-center text-[9px] text-slate-700">Наведите или выберите точку, чтобы посмотреть детали</p>
      {active && <div className="mx-3 mt-3 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-blue-300/10 bg-white/[.06] sm:grid-cols-5">
        <div className="bg-[#07131f] px-3 py-2"><p className="text-[9px] text-slate-700">Время</p><p className="mt-1 text-xs text-slate-300">{formatDate(active.t)}</p></div>
        <div className="bg-[#07131f] px-3 py-2"><p className="text-[9px] text-slate-700">Инференс</p><p className="mt-1 font-mono text-xs text-blue-200">{formatMs(active.ms)}</p></div>
        <div className="bg-[#07131f] px-3 py-2"><p className="text-[9px] text-slate-700">Результат</p><p className={`mt-1 text-xs ${active.refused ? "text-rose-300" : "text-emerald-300"}`}>{active.refused ? "Отказ" : "Совпадение"}</p></div>
        <div className="bg-[#07131f] px-3 py-2"><p className="text-[9px] text-slate-700">Кандидатов</p><p className="mt-1 font-mono text-xs text-slate-300">{active.accepted}</p></div>
        <div className="bg-[#07131f] px-3 py-2"><p className="text-[9px] text-slate-700">Кадров / алертов</p><p className="mt-1 font-mono text-xs text-slate-300">{active.n_images} / {active.alerts}</p></div>
      </div>}
    </div>
  );
}

function SuccessRing({ refusalRate, requests }: { refusalRate: number | null; requests: number }) {
  const success = refusalRate == null ? 0 : Math.max(0, Math.min(100, (1 - refusalRate) * 100));
  const radius = 52;
  const length = 2 * Math.PI * radius;
  return (
    <div className="relative mx-auto h-36 w-36">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={radius} fill="none" stroke="rgba(148,163,184,.09)" strokeWidth="8" />
        <circle cx="64" cy="64" r={radius} fill="none" stroke="#7db8ff" strokeWidth="8" strokeLinecap="round" strokeDasharray={length} strokeDashoffset={length * (1 - success / 100)} className="am-ring transition-all duration-700" style={{ "--len": length } as CSSProperties} />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center"><div><p className="font-mono text-2xl text-slate-100">{requests ? `${Math.round(success)}%` : "—"}</p><p className="mt-1 text-[9px] uppercase tracking-[.14em] text-slate-600">успешных</p></div></div>
    </div>
  );
}

function LocationBars({ locations }: { locations: LocationStat[] }) {
  const top = [...locations].sort((a, b) => b.count - a.count).slice(0, 10);
  const max = Math.max(...top.map((item) => item.count), 1);
  return <div className="space-y-3">{top.map((location, index) => <div key={location.camera_group} className="grid grid-cols-[26px_82px_1fr_24px] items-center gap-2 text-[10px]"><span className="font-mono text-slate-700">{String(index + 1).padStart(2, "0")}</span><span className="truncate text-slate-400">Камера {location.camera_group}</span><span className="h-1.5 overflow-hidden rounded-full bg-white/[.055]"><span className="am-bar block h-full rounded-full bg-gradient-to-r from-blue-500/55 to-blue-200" style={{ "--i": index, width: `${Math.max(8, location.count / max * 100)}%` } as CSSProperties} /></span><span className="text-right font-mono text-slate-500">{location.count}</span></div>)}</div>;
}

function TrackCard({ track, index }: { track: VehicleTrack; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const visibleItems = track.items.slice(0, 4);
  const visibleLocations = expanded ? track.locations : track.locations.slice(0, 5);
  const hiddenLocations = Math.max(0, track.locations.length - 5);
  return (
    <article className="am-card rounded-xl border border-white/[.065] bg-white/[.018] p-4 transition hover:border-blue-300/20 hover:bg-blue-300/[.025]">
      <div className="flex items-start justify-between gap-4">
        <div><p className="font-mono text-[10px] text-blue-200/65">TRACK-{String(index + 1).padStart(3, "0")}</p><p className="mt-1.5 text-sm font-medium text-slate-200">{track.size} наблюдений</p></div>
        <span className="rounded-md border border-emerald-300/15 bg-emerald-400/[.06] px-2 py-1 text-[9px] text-emerald-200">{track.locations.length} камер</span>
      </div>
      <div className="mt-4 flex items-center">
        {visibleItems.map((item, itemIndex) => <img key={item.gallery_id} src={thumbUrl(item.gallery_id)} alt="" loading="lazy" className="h-11 w-14 rounded-lg border-2 border-[#07131f] object-cover" style={{ marginLeft: itemIndex ? -10 : 0, zIndex: visibleItems.length - itemIndex }} />)}
        {track.items.length > visibleItems.length && <span className="ml-2 text-[10px] text-slate-600">+{track.items.length - visibleItems.length}</span>}
      </div>
      <div className="mt-4 flex flex-wrap gap-1.5">
        {visibleLocations.map((location) => <span key={location} className="rounded bg-white/[.035] px-2 py-1 font-mono text-[9px] text-slate-500">CAM {location}</span>)}
        {hiddenLocations > 0 && <button type="button" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded} className="rounded px-2 py-1 text-[9px] text-blue-300/70 transition hover:bg-blue-300/[.06] hover:text-blue-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300">{expanded ? "Свернуть" : `ещё ${hiddenLocations}`}</button>}
      </div>
    </article>
  );
}

function LoadingDashboard() {
  return <div className="grid min-h-[520px] place-items-center"><div className="text-center"><Loader2 className="mx-auto h-6 w-6 animate-spin text-blue-300" /><p className="mt-4 text-xs text-slate-500">Собираем аналитику…</p></div></div>;
}

export default function AnalyticsPage() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [data, setData] = useState<DashboardData>(emptyData);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    const endpoints = ["/api/stats/locations", "/api/stats/tracks", "/api/stats/usage", "/api/stats/my", "/api/stats/my/history?limit=20"];
    try {
      const responses = await Promise.all(endpoints.map((endpoint) => apiFetch(endpoint)));
      const failed = responses.find((response) => !response.ok);
      if (failed) throw new Error(await readApiError(failed, `Не удалось загрузить аналитику (${failed.status}).`));
      const [locations, tracks, usage, my, history] = await Promise.all(responses.map((response) => response.json()));
      setData({ locations, tracks, usage, my, history } as DashboardData);
      setApiStatus("online");
      setUpdatedAt(new Date());
    } catch (cause) {
      setApiStatus("offline");
      setError(cause instanceof Error ? cause.message : "Не удалось загрузить аналитику.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const topTracks = useMemo(() => [...data.tracks.tracks].sort((a, b) => b.size - a.size).slice(0, 6), [data.tracks.tracks]);
  const uniqueLocations = data.locations.locations.length;
  const singletonLocations = data.locations.locations.filter((item) => item.count === 1).length;

  return (
    <div className="app-shell min-h-screen bg-[#050d16] text-slate-100">
      <AppHeader apiStatus={apiStatus} active="Аналитика" />
      <main className="relative mx-auto max-w-[1580px] px-5 py-8 lg:px-8 lg:py-10">
        <div className="app-orb app-orb-one" aria-hidden="true" /><div className="app-orb app-orb-two" aria-hidden="true" />
        <section className="relative z-10 mb-7 flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div>
            <div className="mb-3 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[.24em] text-blue-200/60"><BarChart3 className="h-3.5 w-3.5" />Операционный центр · Stats API</div>
            <h1 className="text-3xl font-medium tracking-[-.035em] text-slate-50 md:text-[38px]">Аналитика платформы</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">Состояние галереи, активность поиска и перемещения автомобилей между камерами — в одном пространстве.</p>
          </div>
          <div className="flex items-center gap-3">
            {updatedAt && <span className="hidden text-[10px] text-slate-600 sm:block">Обновлено {updatedAt.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })}</span>}
            <Button variant="outline" onClick={() => void load()} disabled={loading} className="h-10 rounded-full border-blue-300/15 bg-[#07131f]/80 px-4 text-xs"><RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />Обновить</Button>
          </div>
        </section>

        {loading && !updatedAt ? <LoadingDashboard /> : error ? <Panel className="relative z-10 grid min-h-[420px] place-items-center p-8 text-center"><div className="max-w-md"><span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl border border-red-300/15 bg-red-400/[.06]"><AlertCircle className="h-6 w-6 text-red-300" /></span><h2 className="mt-5 text-lg font-medium">Аналитика недоступна</h2><p className="mt-2 text-sm leading-6 text-slate-500">{error}</p><Button variant="outline" onClick={() => void load()} className="mt-5 rounded-full"><RefreshCw className="h-4 w-4" />Повторить</Button></div></Panel> : <div className="relative z-10 space-y-4">
          <section className="am-stagger grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard label="Объектов в индексе" value={formatNumber(data.locations.total)} detail="Доступны для сопоставления" icon={Images} accent />
            <MetricCard label="Камерных групп" value={formatNumber(uniqueLocations)} detail={`${singletonLocations} с одиночным наблюдением`} icon={Camera} />
            <MetricCard label="Межкамерных треков" value={formatNumber(data.tracks.n_tracks)} detail={`Порог сходства ${Math.round(data.tracks.threshold * 100)}%`} icon={Route} />
            <MetricCard label="Всего поисков" value={formatNumber(data.usage.requests)} detail={`Среднее время ${formatMs(data.usage.avg_ms)}`} icon={Search} />
          </section>

          <section className="grid gap-4 xl:grid-cols-[1.55fr_.85fr]">
            <Panel className="p-5 md:p-6">
              <div className="mb-5 flex flex-wrap items-start justify-between gap-4"><div><div className="flex items-center gap-2"><TrendingUp className="h-4 w-4 text-blue-300" /><h2 className="text-sm font-medium text-slate-100">Активность поиска</h2></div><p className="mt-1.5 text-[11px] text-slate-600">Время инференса последних {data.usage.recent.length} запросов</p></div><div className="flex gap-4 text-right"><div><p className="text-[9px] uppercase tracking-wider text-slate-700">P95</p><p className="mt-1 font-mono text-xs text-slate-300">{formatMs(data.usage.p95_ms)}</p></div><div><p className="text-[9px] uppercase tracking-wider text-slate-700">Отказы</p><p className="mt-1 font-mono text-xs text-slate-300">{formatPercent(data.usage.refusal_rate)}</p></div></div></div>
              <ActivityChart records={data.usage.recent} />
              <div className="mt-4 grid grid-cols-3 gap-px overflow-hidden rounded-xl border border-white/[.06] bg-white/[.06]"><div className="bg-[#07131f] p-3"><p className="text-[9px] text-slate-700">Среднее</p><p className="mt-1 font-mono text-sm text-slate-300">{formatMs(data.usage.avg_ms)}</p></div><div className="bg-[#07131f] p-3"><p className="text-[9px] text-slate-700">Алерты</p><p className="mt-1 font-mono text-sm text-slate-300">{formatNumber(data.usage.alerts_total)}</p></div><div className="bg-[#07131f] p-3"><p className="text-[9px] text-slate-700">Принято кандидатов</p><p className="mt-1 font-mono text-sm text-slate-300">{formatNumber(data.usage.recent.reduce((sum, item) => sum + item.accepted, 0))}</p></div></div>
            </Panel>

            <Panel className="p-5 md:p-6">
              <div className="flex items-center gap-2"><Gauge className="h-4 w-4 text-blue-300" /><h2 className="text-sm font-medium">Мои показатели</h2></div><p className="mt-1.5 text-[11px] text-slate-600">Статистика текущего аккаунта</p>
              <div className="mt-5 grid grid-cols-[150px_1fr] items-center gap-4"><SuccessRing refusalRate={data.my.refusal_rate} requests={data.my.requests} /><div className="space-y-3"><div className="flex justify-between border-b border-white/[.055] pb-2 text-xs"><span className="text-slate-600">Запросов</span><span className="font-mono text-slate-300">{data.my.requests}</span></div><div className="flex justify-between border-b border-white/[.055] pb-2 text-xs"><span className="text-slate-600">Среднее</span><span className="font-mono text-slate-300">{formatMs(data.my.avg_ms)}</span></div><div className="flex justify-between text-xs"><span className="text-slate-600">Алерты</span><span className="font-mono text-slate-300">{data.my.alerts_total}</span></div></div></div>
              {data.my.requests === 0 ? <a href="/app" className="mt-5 flex items-center justify-between rounded-xl border border-blue-300/15 bg-blue-300/[.05] p-4 text-xs text-blue-100 transition hover:bg-blue-300/[.08]"><span><span className="block font-medium">Запустите первый поиск</span><span className="mt-1 block text-[10px] text-slate-500">И здесь появятся личные метрики</span></span><ArrowRight className="h-4 w-4" /></a> : <div className="mt-5"><p className="mb-3 text-[9px] font-bold uppercase tracking-[.14em] text-slate-700">Топ камер</p><div className="flex flex-wrap gap-2">{data.my.top_cameras.slice(0, 5).map((camera) => <span key={camera.camera_group} className="rounded-md border border-white/[.06] bg-white/[.025] px-2.5 py-1.5 text-[10px] text-slate-500">CAM {camera.camera_group} · {camera.count}</span>)}</div></div>}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[.8fr_1.2fr]">
            <Panel className="p-5 md:p-6">
              <div className="mb-5 flex items-start justify-between gap-3"><div><div className="flex items-center gap-2"><MapPinned className="h-4 w-4 text-blue-300" /><h2 className="text-sm font-medium">Покрытие камер</h2></div><p className="mt-1.5 text-[11px] text-slate-600">Лидеры по количеству наблюдений</p></div><span className="rounded-md border border-white/[.06] bg-white/[.025] px-2 py-1 font-mono text-[9px] text-slate-600">TOP 10</span></div>
              <LocationBars locations={data.locations.locations} />
            </Panel>

            <Panel className="p-5 md:p-6">
              <div className="mb-5 flex flex-wrap items-start justify-between gap-3"><div><div className="flex items-center gap-2"><Network className="h-4 w-4 text-blue-300" /><h2 className="text-sm font-medium">Межкамерные треки</h2></div><p className="mt-1.5 text-[11px] text-slate-600">Крупнейшие цепочки повторных наблюдений одного ТС</p></div><span className="rounded-full border border-white/[.08] bg-white/[.025] px-3 py-1.5 text-[9px] text-slate-500">{data.tracks.n_tracks} треков</span></div>
              <div className="am-stagger grid gap-3 md:grid-cols-2">{topTracks.map((track, index) => <TrackCard key={`${track.locations.join("-")}-${index}`} track={track} index={index} />)}</div>
            </Panel>
          </section>

          <Panel className="p-5 md:p-6">
            <div className="mb-5 flex items-start justify-between gap-3"><div><div className="flex items-center gap-2"><History className="h-4 w-4 text-blue-300" /><h2 className="text-sm font-medium">История моих поисков</h2></div><p className="mt-1.5 text-[11px] text-slate-600">Последние операции и лучшие кандидаты</p></div><span className="font-mono text-[10px] text-slate-700">{data.history.length}/20</span></div>
            {data.history.length ? <div className="overflow-x-auto"><table className="w-full min-w-[780px] text-left"><thead><tr className="border-b border-white/[.07] text-[9px] uppercase tracking-[.12em] text-slate-700"><th className="pb-3 font-medium">Дата</th><th className="pb-3 font-medium">Статус</th><th className="pb-3 font-medium">Кадры</th><th className="pb-3 font-medium">Принято</th><th className="pb-3 font-medium">Время</th><th className="pb-3 font-medium">Лучший кандидат</th></tr></thead><tbody className="am-rows">{data.history.map((item, index) => <tr key={item.id} style={{ "--i": Math.min(index, 12) } as CSSProperties} className="border-b border-white/[.045] text-xs last:border-0"><td className="py-4 font-mono text-[10px] text-slate-500">{formatDate(item.created_at)}</td><td className="py-4">{item.refused ? <span className="inline-flex items-center gap-1.5 text-amber-300"><ShieldAlert className="h-3.5 w-3.5" />Отказ</span> : <span className="inline-flex items-center gap-1.5 text-emerald-300"><CheckCircle2 className="h-3.5 w-3.5" />Совпадение</span>}</td><td className="py-4 font-mono text-slate-400">{item.n_images}</td><td className="py-4 font-mono text-slate-400">{item.n_accepted}</td><td className="py-4 font-mono text-slate-400">{formatMs(item.inference_ms)}</td><td className="py-4">{item.top3[0] ? <span className="flex items-center gap-3"><img src={thumbUrl(item.top3[0].gallery_id)} alt="" className="h-9 w-12 rounded-md object-cover" /><span><span className="block max-w-[160px] truncate font-mono text-[10px] text-slate-300">{item.top3[0].gallery_id}</span><span className="mt-1 block text-[9px] text-blue-200/65">{Math.round(item.top3[0].confidence * 100)}% · CAM {item.top3[0].camera_group}</span></span></span> : <span className="text-slate-700">—</span>}</td></tr>)}</tbody></table></div> : <div className="grid min-h-[180px] place-items-center rounded-xl border border-dashed border-white/[.07] bg-white/[.012] text-center"><div><Clock3 className="mx-auto h-6 w-6 text-slate-700" /><p className="mt-3 text-xs text-slate-500">История пока пуста</p><a href="/app" className="mt-2 inline-flex items-center gap-1.5 text-[10px] text-blue-300/70 hover:text-blue-200">Перейти к поиску <ArrowRight className="h-3 w-3" /></a></div></div>}
          </Panel>
        </div>}

        <footer className="relative z-10 mt-8 flex flex-col justify-between gap-3 border-t border-white/[.06] pt-4 text-[10px] text-slate-600 sm:flex-row sm:items-center"><span className="flex items-center gap-2"><Database className="h-3.5 w-3.5" />FALCON ReID · данные из /api/stats</span><a href="/docs#/stats" className="flex items-center gap-2 transition hover:text-slate-300"><CircleHelp className="h-3.5 w-3.5" />Документация Stats API</a></footer>
      </main>
    </div>
  );
}
