import { useEffect, useState, type CSSProperties } from "react";
import {
  ArrowRight,
  Box,
  Car,
  ChartNoAxesColumnIncreasing,
  Check,
  Clipboard,
  Code2,
  Database,
  Github,
  LifeBuoy,
  Menu,
  Network,
  Terminal,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  codeExamples,
  navigation,
  technologyFeatures,
  trustCategories,
  useCases,
  type UseCase,
} from "@/data/content";
import "../landing-motion.css";

const d = (ms: number) => ({ "--d": `${ms}ms` }) as CSSProperties;

const Logo = () => (
  <a
    href="#product"
    className="flex items-center gap-2.5 font-semibold tracking-tight"
    aria-label="FALCON ReID — на главную"
  >
    <img src="/falcon-mark.svg" width="34" height="34" alt="" />
    <span>
      FALCON<span className="font-normal text-muted-foreground">ReID</span>
    </span>
  </a>
);

/**
 * Анимации лендинга: секции [data-reveal] проявляются при прокрутке (старт —
 * после загрузочного экрана), полоса прогресса, уплотнение шапки, параллакс героя.
 */
export function LandingMotion() {
  useEffect(() => {
    const root = document.documentElement;
    root.classList.add("motion-ready");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add("is-visible");
            io.unobserve(e.target);
          }
        }
      },
      { threshold: 0.12, rootMargin: "0px 0px -6% 0px" },
    );
    let started = false;
    const start = () => {
      if (started) return;
      started = true;
      document.querySelectorAll("[data-reveal]").forEach((el) => io.observe(el));
    };
    const mo = new MutationObserver(() => {
      if (root.classList.contains("fl-ready")) window.setTimeout(start, 250);
    });
    if (root.classList.contains("fl-ready")) start();
    else mo.observe(root, { attributes: true, attributeFilter: ["class"] });
    const fallback = window.setTimeout(start, 13000);

    const bar = document.getElementById("scroll-progress");
    const header = document.querySelector(".landing-header");
    const onScroll = () => {
      const max = Math.max(1, root.scrollHeight - root.clientHeight);
      if (bar) bar.style.transform = `scaleX(${Math.min(1, window.scrollY / max)})`;
      header?.classList.toggle("is-scrolled", window.scrollY > 8);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    const hero = document.getElementById("product");
    const layer = document.querySelector<HTMLElement>(".hero-parallax");
    const fine = window.matchMedia("(pointer: fine)").matches;
    const onMove = (ev: PointerEvent) => {
      if (!hero || !layer) return;
      const r = hero.getBoundingClientRect();
      layer.style.setProperty("--mx", (((ev.clientX - r.left) / r.width) * 2 - 1).toFixed(3));
      layer.style.setProperty("--my", (((ev.clientY - r.top) / r.height) * 2 - 1).toFixed(3));
    };
    const onLeave = () => {
      layer?.style.setProperty("--mx", "0");
      layer?.style.setProperty("--my", "0");
    };
    if (fine && !reduce) {
      hero?.addEventListener("pointermove", onMove);
      hero?.addEventListener("pointerleave", onLeave);
    }

    return () => {
      io.disconnect();
      mo.disconnect();
      window.clearTimeout(fallback);
      window.removeEventListener("scroll", onScroll);
      hero?.removeEventListener("pointermove", onMove);
      hero?.removeEventListener("pointerleave", onLeave);
      root.classList.remove("motion-ready");
    };
  }, []);
  return <div id="scroll-progress" className="scroll-progress" aria-hidden="true" />;
}

export function LandingHeader() {
  return (
    <header className="landing-header fixed inset-x-0 top-0 z-50 border-b border-white/[.06] bg-background/80 backdrop-blur-xl">
      <div className="relative flex h-16 w-full items-center px-6 md:px-10 xl:px-16 2xl:px-20">
        <Logo />
        <nav
          className="absolute left-1/2 hidden -translate-x-1/2 items-center gap-7 lg:flex"
          aria-label="Основная навигация"
        >
          {navigation.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              {item.label}
            </a>
          ))}
        </nav>
        <div className="ml-auto hidden items-center gap-3 lg:flex">
          <Button
            asChild
            variant="outline"
            size="sm"
            className="rounded-full px-5"
          >
            <a href="/auth">Войти</a>
          </Button>
          <Button
            asChild
            size="sm"
            className="rounded-full bg-slate-50 px-5 text-slate-950 hover:bg-white"
          >
            <a href="/auth?mode=register">
              Начать бесплатно <ArrowRight className="h-4 w-4" />
            </a>
          </Button>
        </div>
        <Sheet>
          <SheetTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="ml-auto lg:hidden"
              aria-label="Открыть меню"
            >
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent>
            <SheetHeader>
              <SheetTitle>
                <Logo />
              </SheetTitle>
            </SheetHeader>
            <nav
              className="mt-10 flex flex-col gap-2"
              aria-label="Мобильная навигация"
            >
              {navigation.map((item) => (
                <SheetClose asChild key={item.href}>
                  <a
                    href={item.href}
                    className="rounded-lg px-3 py-3 text-lg text-muted-foreground hover:bg-accent hover:text-foreground"
                  >
                    {item.label}
                  </a>
                </SheetClose>
              ))}
              <Button asChild size="lg" className="mt-5">
                <a href="/auth?mode=register">
                  Начать бесплатно <ArrowRight className="h-4 w-4" />
                </a>
              </Button>
            </nav>
          </SheetContent>
        </Sheet>
      </div>
    </header>
  );
}

const SpecTag = ({
  className,
  icon: Icon,
  title,
  value,
  colorDot = false,
  delay = 0,
}: {
  className: string;
  delay?: number;
  icon?: typeof Box;
  title: string;
  value: string;
  colorDot?: boolean;
}) => (
  <Card
    style={d(delay)}
    className={`hero-tag absolute hidden rounded-lg border-blue-300/25 bg-[#071624]/90 shadow-[0_16px_42px_rgba(0,0,0,.4)] backdrop-blur-md md:block ${className}`}
  >
    <CardContent className="flex min-h-[60px] items-center gap-3 px-3.5 py-2.5">
      {colorDot ? (
        <span className="h-4 w-4 shrink-0 rounded-full bg-[#ffc82e] shadow-[0_0_10px_rgba(255,200,46,.45)]" />
      ) : Icon ? (
        <Icon className="h-6 w-6 shrink-0 text-blue-100" strokeWidth={1.6} />
      ) : null}
      <div>
        <p className="text-[10px] leading-4 text-slate-400">{title}</p>
        <p className="whitespace-pre-line text-[11px] font-medium leading-4 text-slate-100">
          {value}
        </p>
      </div>
    </CardContent>
  </Card>
);

// [left %, top %, длительность c, задержка c, дрейф px]
const HERO_PARTICLES: [number, number, number, number, number][] = [
  [8, 72, 9, 0.2, 12], [16, 40, 11, 2.1, -8], [24, 84, 8, 4.3, 6], [33, 58, 12, 1.2, -14],
  [41, 90, 10, 3.4, 10], [48, 36, 9.5, 5.6, -6], [56, 76, 11.5, 0.8, 16], [63, 48, 8.5, 2.9, -10],
  [70, 88, 10.5, 4.9, 8], [77, 30, 12.5, 1.7, -12], [84, 66, 9, 3.8, 14], [91, 52, 11, 0.5, -4],
  [12, 22, 13, 6.2, 6], [37, 18, 10, 7.1, -9], [66, 14, 12, 5.1, 11], [88, 20, 9.5, 6.6, -7],
];
// ключевые точки на машине (фара, решётка, колесо, крыша, заднее колесо): [left %, top %, задержка мс]
const HERO_NODES: [number, number, number][] = [
  [5, 55, 1700], [15, 60, 1850], [44, 83, 2000], [58, 17, 2150], [89, 68, 2300],
];

export function HeroSection() {
  return (
    <section
      id="product"
      className="hero-grid relative min-h-[520px] pt-16 md:min-h-[550px]"
    >
      <div className="hero-glow" aria-hidden="true" />
      <div className="container max-w-7xl relative grid min-h-[470px] items-center gap-0 py-10 md:min-h-[600px] md:grid-cols-[.95fr_1.05fr] md:gap-3 md:py-0">
        <div className="relative z-20 md:-translate-y-1">
          <p data-hero style={d(150)} className="mb-5 text-[10px] font-semibold uppercase tracking-[.28em] text-blue-200/70 sm:text-[11px]">
            Компьютерное зрение для реального мира
          </p>
          <h1 className="max-w-2xl text-[44px]  leading-[1.05] sm:text-[50px] xl:text-[60px] 2xl:text-[66px]">
            <span className="hero-line">
              <span style={d(260)} className="lg:whitespace-nowrap">Один автомобиль.</span>
            </span>
            <span className="hero-line">
              <span style={d(420)} className="hero-shine lg:whitespace-nowrap">Тысячи камер.</span>
            </span>
          </h1>
          <p data-hero style={d(640)} className="mt-3 max-w-md text-[17px] leading-[1.55] text-muted-foreground xl:text-xl">
            Найдите тот же автомобиль
            <br className="hidden sm:block" /> на любых камерах.
          </p>
          <div data-hero style={d(800)} className="mt-7 flex flex-wrap gap-3">
            <Button
              asChild
              size="lg"
              className="hero-cta h-12 rounded-full bg-slate-50 px-[22px] text-slate-950 shadow-[0_0_24px_rgba(148,197,255,.18)] hover:bg-white"
            >
              <a href="/auth?mode=register">
                Попробовать бесплатно <ArrowRight className="h-4 w-4" />
              </a>
            </Button>
          </div>
        </div>
        <div className="hero-visual pointer-events-none absolute inset-0 z-0 md:pointer-events-auto md:relative md:inset-auto md:min-h-[486px]">
          <div className="hero-parallax absolute inset-0">
          <div
            className="absolute inset-0 rounded-full bg-blue-500/10 blur-[90px]"
            aria-hidden="true"
          />
          <div className="hero-particles" aria-hidden="true">
            {HERO_PARTICLES.map((p, i) => (
              <span
                key={i}
                style={{ left: `${p[0]}%`, top: `${p[1]}%`, "--t": `${p[2]}s`, "--d": `${p[3]}s`, "--x": `${p[4]}px` } as CSSProperties}
              />
            ))}
          </div>
          <div className="hero-scan hidden md:block" aria-hidden="true" />
          <div className="hero-detect hidden md:block" aria-hidden="true">
            <i /><i /><i /><i />
            <span className="hero-detect-label">ID A-4172 <b>0.97</b></span>
          </div>
          {HERO_NODES.map(([x, y, delay], i) => (
            <span
              key={i}
              aria-hidden="true"
              className="hero-node hidden md:block"
              style={{ left: `${x}%`, top: `${y}%`, "--d": `${delay}ms` } as CSSProperties}
            />
          ))}
          <img
            src="/hero-car-figma.png"
            width="1670"
            height="942"
            alt="Цифровая трёхмерная модель автомобиля"
            className="hero-car absolute left-1/2 top-[66%] z-10 w-[88%] max-w-none -translate-x-1/2 -translate-y-1/2 opacity-60 drop-shadow-[0_0_25px_rgba(68,153,255,.36)] md:left-[49%] md:top-[51%] md:w-[700px] md:opacity-80"
          />
          <SpecTag
            className="left-[10%] top-[10%] z-20 w-[108px]"
            title="Цвет"
            value="Жёлтый"
            colorDot
            delay={2300}
          />
          <SpecTag
            className="right-[-1%] top-[8%] z-20 w-[126px]"
            icon={Car}
            title="Кузов"
            value="Седан"
            delay={2450}
          />
          <SpecTag
            className="bottom-[6%] left-[7%] z-20 w-[132px]"
            icon={Box}
            title="Геометрия"
            value={"Размеры\nи пропорции"}
            delay={2600}
          />
          <SpecTag
            className="bottom-[7%] right-[-1%] z-20 w-[178px]"
            icon={ChartNoAxesColumnIncreasing}
            title=""
            value={"Уникальный\nвектор признаков"}
            delay={2750}
          />
          </div>
        </div>
      </div>
    </section>
  );
}

function UseCaseCard({ item, index }: { item: UseCase; index: number }) {
  return (
    <Card data-reveal style={d(index * 110)} className="usecase-card group overflow-hidden border-white/[.09] bg-card/70 transition duration-300 hover:-translate-y-1 hover:border-blue-300/25">
      <div className="usecase-media relative aspect-[1.55] overflow-hidden bg-secondary">
        <span className="usecase-scan" aria-hidden="true" />
        <img
          src={item.image}
          alt={item.alt}
          width="1280"
          height="768"
          loading="lazy"
          className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.04]"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-card/70 to-transparent" />
      </div>
      <CardHeader className="p-5">
        <CardTitle className="text-base">{item.title}</CardTitle>
        <CardDescription className="leading-relaxed">
          {item.description}
        </CardDescription>
      </CardHeader>
    </Card>
  );
}

export function UseCasesSection() {
  return (
    <section id="solutions" className="py-16 sm:py-20">
      <div className="container max-w-7xl">
        <div className="mb-10 flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
          <div data-reveal>
            <p className="eyebrow">Реальные задачи. Реальный результат.</p>
            <h2 className="section-title">Где FALCON ReID уже помогает</h2>
          </div>
          <Button asChild variant="outline" data-reveal style={d(150)}>
            <a href="#technology">
              Все сценарии <ArrowRight className="h-4 w-4" />
            </a>
          </Button>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-4">
          {useCases.map((item, index) => (
            <UseCaseCard key={item.title} item={item} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
}

const TECH_MATCHES = [
    {
      target: "left-[21.35%] top-[10.4%] h-[10.7%] w-[6.8%]",
      source: "left-[67.9%] top-[19.2%] h-[12.9%] w-[7.5%]",
      label: "Совпадение 0.882",
      chip: [71.65, 19.2],
    },
    {
      target: "left-[31.1%] top-[33.5%] h-[12%] w-[7.1%]",
      source: "left-[67.9%] top-[34.7%] h-[12.9%] w-[7.5%]",
      label: "Совпадение 0.876",
      chip: [71.65, 34.7],
    },
    {
      target: "left-[52.2%] top-[62.6%] h-[12.8%] w-[8%]",
      source: "left-[67.9%] top-[50.3%] h-[13%] w-[7.5%]",
      label: "Совпадение 0.869",
      chip: [71.65, 50.3],
    },
];

export function TechnologySection() {
  const [activeMatch, setActiveMatch] = useState<number | null>(null);
  const [hovering, setHovering] = useState(false);
  const matches = TECH_MATCHES;
  // пока пользователь не навёл курсор — совпадения подсвечиваются сами по кругу
  useEffect(() => {
    if (hovering) return;
    const id = window.setInterval(
      () => setActiveMatch((i) => (i === null ? 0 : (i + 1) % TECH_MATCHES.length)),
      2300,
    );
    return () => window.clearInterval(id);
  }, [hovering]);
  return (
    <section id="technology" className="section-line py-16 sm:py-20">
      <div className="container max-w-7xl grid items-center gap-14 md:grid-cols-[.75fr_1.25fr]">
        <div data-reveal="left">
          <p className="eyebrow">Технология, которой можно доверять</p>
          <h2 className="section-title max-w-md">
            Больше, чем просто распознавание
          </h2>
          <p className="mt-5 max-w-lg leading-relaxed text-muted-foreground">
            Мы объединяем компьютерное зрение, ReID и масштабируемую
            инфраструктуру, чтобы решать реальные задачи бизнеса и города.
          </p>
          <div className="mt-7 flex flex-wrap gap-2">
            {technologyFeatures.map((f) => (
              <Badge
                key={f.label}
                variant="outline"
                className="tech-badge gap-2 px-3 py-1.5 text-muted-foreground"
              >
                <span>{f.label}</span>
                <strong className="text-blue-200">{f.value}</strong>
              </Badge>
            ))}
          </div>
          <Button asChild variant="outline" className="mt-8">
            <a href="/auth">
              Узнать о технологии <ArrowRight className="h-4 w-4" />
            </a>
          </Button>
        </div>
        <Card data-reveal="scale" style={d(150)} className="tech-panel relative overflow-hidden border-blue-300/15 bg-[#081827] p-2 sm:p-3">
          <div className="relative aspect-[1672/941] overflow-hidden rounded-lg bg-[#0b1c2a]">
            <img
              src="/technology-reid-matches.png"
              alt="Распознавание автомобилей на городском перекрёстке и найденные совпадения"
              width="1672"
              height="941"
              loading="lazy"
              className="h-full w-full object-cover"
            />
            <div className="tech-scan" aria-hidden="true" />
            {activeMatch !== null && (
              <span
                key={activeMatch}
                className="match-chip"
                style={{ left: `${matches[activeMatch].chip[0]}%`, top: `${matches[activeMatch].chip[1]}%` }}
              >
                {matches[activeMatch].label}
              </span>
            )}
            {matches.map((match, index) => (
              <span
                key={match.label}
                aria-hidden="true"
                className={`pointer-events-none absolute rounded-sm border-2 border-blue-200 bg-blue-300/10 transition duration-200 ${match.target} ${activeMatch === index ? "scale-110 opacity-100 shadow-[0_0_26px_7px_rgba(96,165,250,.8)]" : "scale-100 opacity-0"}`}
              />
            ))}
            {matches.map((match, index) => (
              <button
                key={`${match.label}-trigger`}
                type="button"
                aria-label={`Подсветить автомобиль: ${match.label}`}
                className={`absolute cursor-pointer rounded-md border transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-200 ${match.source} ${activeMatch === index ? "border-blue-200/90 bg-blue-300/10 shadow-[0_0_18px_rgba(96,165,250,.65)]" : "border-transparent hover:border-blue-200/60"}`}
                onMouseEnter={() => { setHovering(true); setActiveMatch(index); }}
                onMouseLeave={() => { setHovering(false); setActiveMatch(null); }}
                onFocus={() => { setHovering(true); setActiveMatch(index); }}
                onBlur={() => { setHovering(false); setActiveMatch(null); }}
              />
            ))}
          </div>
        </Card>
      </div>
    </section>
  );
}

export function TrustStrip() {
  return (
    <section className="border-y border-border bg-white/[.012] py-11">
      <div className="container max-w-7xl">
        <p data-reveal className="mb-7 text-xs uppercase tracking-[.2em] text-muted-foreground">
          Нам доверяют
        </p>
        <div data-reveal style={d(120)} className="trust-marquee">
          <div className="trust-track">
            {[0, 1].map((copy) =>
              trustCategories.map(({ label, icon: Icon }) => (
                <div
                  key={`${copy}-${label}`}
                  aria-hidden={copy === 1 || undefined}
                  className="flex shrink-0 items-center gap-3 text-sm text-slate-300"
                >
                  <Icon className="h-7 w-7 text-blue-200/75" strokeWidth={1.5} />
                  <span>{label}</span>
                </div>
              )),
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

export function ApiSection() {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState("python");
  const copy = async (code: string) => {
    const legacyCopy = () => {
      const area = document.createElement("textarea");
      area.value = code;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      const ok = document.execCommand("copy");
      area.remove();
      if (!ok) throw new Error("copy failed");
    };
    try {
      // Clipboard API есть только в защищённом контексте (https/localhost) и может быть запрещён.
      try {
        if (!navigator.clipboard?.writeText)
          throw new Error("no clipboard api");
        await navigator.clipboard.writeText(code);
      } catch {
        legacyCopy();
      }
      setCopied(true);
      toast.success("Код скопирован");
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      toast.error("Не удалось скопировать — выделите код вручную");
    }
  };
  return (
    <section id="api" className="section-line py-16 sm:py-20">
      <div className="container max-w-7xl grid gap-12 md:grid-cols-[.72fr_1.28fr]">
        <div data-reveal="left">
          <p className="eyebrow">Простая интеграция</p>
          <h2 className="section-title">Гибкий API для ваших продуктов</h2>
          <p className="mt-5 max-w-lg leading-relaxed text-muted-foreground">
            Подключите визуальный поиск к существующей инфраструктуре.
            OpenAPI-схема, понятные ответы и единый endpoint.
          </p>
          <Button asChild size="lg" className="mt-8">
            <a href="/docs">
              Перейти к документации <ArrowRight className="h-4 w-4" />
            </a>
          </Button>
          <div className="mt-9 grid grid-cols-2 gap-4 text-sm text-muted-foreground">
            {[
              [Network, "REST API"],
              [Zap, "Высокая скорость"],
              [Database, "Масштабируемость"],
              [Code2, "Подробная документация"],
              [LifeBuoy, "Техническая поддержка"],
            ].map(([Icon, text]) => {
              const C = Icon as typeof Zap;
              return (
                <div key={text as string} className="api-feature flex items-center gap-2">
                  <C className="h-4 w-4 text-blue-300" />
                  {text as string}
                </div>
              );
            })}
          </div>
        </div>
        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          data-reveal="right"
          style={d(150)}
          className="overflow-hidden rounded-xl border border-blue-300/15 bg-[#06131f] shadow-2xl"
        >
          <div className="flex items-center justify-between border-b border-border px-3 py-2">
            <TabsList className="bg-transparent">
              {codeExamples.map((ex) => (
                <TabsTrigger key={ex.id} value={ex.id}>
                  {ex.label}
                </TabsTrigger>
              ))}
            </TabsList>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    copy(
                      codeExamples.find((x) => x.id === activeTab)?.code ??
                        codeExamples[0].code,
                    )
                  }
                >
                  {copied ? (
                    <Check className="h-4 w-4 text-emerald-400" />
                  ) : (
                    <Clipboard className="h-4 w-4" />
                  )}
                  <span className="hidden sm:inline">Копировать</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>Скопировать пример</TooltipContent>
            </Tooltip>
          </div>
          {codeExamples.map((ex) => (
            <TabsContent key={ex.id} value={ex.id} className="m-0">
              <pre className="min-h-[330px] overflow-x-auto p-6 font-mono text-[10px] leading-8 text-slate-400">
                <code>
                  {ex.code.split("\n").map((line, i, all) => (
                    <span key={i} className="code-line" style={{ "--i": i } as CSSProperties}>
                      {line || " "}
                      {i === all.length - 1 && <span className="code-caret" aria-hidden="true" />}
                    </span>
                  ))}
                </code>
              </pre>
            </TabsContent>
          ))}
        </Tabs>
      </div>
    </section>
  );
}

export function FinalCta() {
  return (
    <section id="cta" className="px-6 py-8 sm:py-14">
      <div data-reveal="scale" className="cta-frame container max-w-7xl relative overflow-hidden rounded-2xl border border-blue-300/20 bg-[#081A2A]/50 px-7 py-12 sm:px-12">
        <div className="relative z-10 max-w-lg">
          <h2 className="text-3xl font-medium tracking-tight sm:text-4xl">
            Готовы найти
            <br />
            свой автомобиль?
          </h2>
          <p className="mt-4 text-muted-foreground">
            Начните бесплатно. Никаких обязательств.
          </p>
        </div>
        <Button
          asChild
          size="lg"
          className="cta-button relative z-10 mt-8 bg-slate-50 text-slate-950 hover:bg-white sm:absolute sm:right-12 sm:top-1/2 sm:mt-0 sm:-translate-y-1/2"
        >
          <a href="/auth?mode=register">
            Начать бесплатно <ArrowRight className="h-4 w-4" />
          </a>
        </Button>
        <img
          src="/cta-car-figma.png"
          width="1832"
          height="858"
          alt=""
          className="cta-car pointer-events-none absolute -bottom-4 left-1/2 w-[82%] -translate-x-1/2 opacity-30 sm:bottom-auto sm:left-[38%] sm:top-1/2 sm:w-[50%] sm:translate-x-0 sm:-translate-y-1/2 sm:opacity-65"
        />
      </div>
    </section>
  );
}

export function LandingFooter() {
  return (
    <footer className="border-t border-border py-9">
      <div data-reveal className="container max-w-7xl">
        <div className="flex flex-col items-start justify-between gap-7 md:flex-row md:items-center">
          <Logo />
          <nav className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted-foreground">
            {navigation.slice(0, 3).map((n) => (
              <a key={n.href} href={n.href} className="hover:text-foreground">
                {n.label}
              </a>
            ))}
            <a href="/docs" className="hover:text-foreground">
              Документация
            </a>
          </nav>
          <div className="flex gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button asChild variant="ghost" size="icon">
                  <a href="/openapi.json" aria-label="Открыть OpenAPI-схему">
                    <Terminal className="h-5 w-5" />
                  </a>
                </Button>
              </TooltipTrigger>
              <TooltipContent>OpenAPI</TooltipContent>
            </Tooltip>
            <Button asChild variant="ghost" size="icon">
              <a
                href="https://github.com/Enoti4Ka44/FalconREID"
                target="_blank"
                rel="noreferrer"
                aria-label="Исходный код на GitHub"
              >
                <Github className="h-5 w-5" />
              </a>
            </Button>
          </div>
        </div>
        <Separator className="my-7" />
        <div className="flex flex-col justify-between gap-3 text-xs text-muted-foreground sm:flex-row">
          <span>© 2026 FALCON ReID. Визуальный поиск транспорта.</span>
          <span>Создано для более безопасных городов.</span>
        </div>
      </div>
    </footer>
  );
}
