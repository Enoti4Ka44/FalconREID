import { useState } from "react";
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
  Play,
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

export function LandingHeader() {
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-white/[.06] bg-background/80 backdrop-blur-xl">
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
}: {
  className: string;
  icon?: typeof Box;
  title: string;
  value: string;
  colorDot?: boolean;
}) => (
  <Card
    className={`absolute hidden rounded-lg border-blue-300/25 bg-[#071624]/90 shadow-[0_16px_42px_rgba(0,0,0,.4)] backdrop-blur-md md:block ${className}`}
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

export function HeroSection() {
  return (
    <section
      id="product"
      className="hero-grid relative min-h-[520px] pt-16 md:min-h-[550px]"
    >
      <div className="hero-glow" aria-hidden="true" />
      <div className="container max-w-7xl relative grid min-h-[470px] items-center gap-0 py-10 md:min-h-[600px] md:grid-cols-[.95fr_1.05fr] md:gap-3 md:py-0">
        <div className="relative z-20 md:-translate-y-1">
          <p className="mb-5 text-[10px] font-semibold uppercase tracking-[.28em] text-blue-200/70 sm:text-[11px]">
            Компьютерное зрение для реального мира
          </p>
          <h1 className="max-w-2xl text-[44px]  leading-[1.05] sm:text-[50px] xl:text-[60px] 2xl:text-[66px]">
            <span className="lg:whitespace-nowrap">Один автомобиль.</span>
            <br />
            <span className="text-gradient lg:whitespace-nowrap">
              Тысячи камер.
            </span>
          </h1>
          <p className="mt-3 max-w-md text-[17px] leading-[1.55] text-muted-foreground xl:text-xl">
            Найдите тот же автомобиль
            <br className="hidden sm:block" /> на любых камерах.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Button
              asChild
              size="lg"
              className="h-12 rounded-full bg-slate-50 px-[22px] text-slate-950 shadow-[0_0_24px_rgba(148,197,255,.18)] hover:bg-white"
            >
              <a href="/auth?mode=register">
                Попробовать бесплатно <ArrowRight className="h-4 w-4" />
              </a>
            </Button>
            <Button
              asChild
              variant="outline"
              size="lg"
              className="h-12 rounded-full border-blue-200/25 bg-background/30 px-[22px]"
            >
              <a href="#technology">
                <Play className="h-4 w-4" /> Смотреть демо
              </a>
            </Button>
          </div>
        </div>
        <div className="hero-visual pointer-events-none absolute inset-0 z-0 md:pointer-events-auto md:relative md:inset-auto md:min-h-[486px]">
          <div
            className="absolute inset-0 rounded-full bg-blue-500/10 blur-[90px]"
            aria-hidden="true"
          />
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
          />
          <SpecTag
            className="right-[-1%] top-[8%] z-20 w-[126px]"
            icon={Car}
            title="Кузов"
            value="Седан"
          />
          <SpecTag
            className="bottom-[6%] left-[7%] z-20 w-[132px]"
            icon={Box}
            title="Геометрия"
            value={"Размеры\nи пропорции"}
          />
          <SpecTag
            className="bottom-[7%] right-[-1%] z-20 w-[178px]"
            icon={ChartNoAxesColumnIncreasing}
            title=""
            value={"Уникальный\nвектор признаков"}
          />
        </div>
      </div>
    </section>
  );
}

function UseCaseCard({ item }: { item: UseCase }) {
  return (
    <Card className="group overflow-hidden border-white/[.09] bg-card/70 transition duration-300 hover:-translate-y-1 hover:border-blue-300/25">
      <div className="relative aspect-[1.55] overflow-hidden bg-secondary">
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
          <div>
            <p className="eyebrow">Реальные задачи. Реальный результат.</p>
            <h2 className="section-title">Где FALCON ReID уже помогает</h2>
          </div>
          <Button asChild variant="outline">
            <a href="#technology">
              Все сценарии <ArrowRight className="h-4 w-4" />
            </a>
          </Button>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-4">
          {useCases.map((item) => (
            <UseCaseCard key={item.title} item={item} />
          ))}
        </div>
      </div>
    </section>
  );
}

export function TechnologySection() {
  const [activeMatch, setActiveMatch] = useState<number | null>(null);
  const matches = [
    {
      target: "left-[21.35%] top-[10.4%] h-[10.7%] w-[6.8%]",
      source: "left-[67.9%] top-[19.2%] h-[12.9%] w-[7.5%]",
      label: "Совпадение 0.882",
    },
    {
      target: "left-[31.1%] top-[33.5%] h-[12%] w-[7.1%]",
      source: "left-[67.9%] top-[34.7%] h-[12.9%] w-[7.5%]",
      label: "Совпадение 0.876",
    },
    {
      target: "left-[52.2%] top-[62.6%] h-[12.8%] w-[8%]",
      source: "left-[67.9%] top-[50.3%] h-[13%] w-[7.5%]",
      label: "Совпадение 0.869",
    },
  ];
  return (
    <section id="technology" className="section-line py-16 sm:py-20">
      <div className="container max-w-7xl grid items-center gap-14 md:grid-cols-[.75fr_1.25fr]">
        <div>
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
                className="gap-2 px-3 py-1.5 text-muted-foreground"
              >
                <span>{f.label}</span>
                <strong className="text-blue-200">{f.value}</strong>
              </Badge>
            ))}
          </div>
          <Button asChild variant="outline" className="mt-8">
            <a href="/docs">
              Узнать о технологии <ArrowRight className="h-4 w-4" />
            </a>
          </Button>
        </div>
        <Card className="tech-panel relative overflow-hidden border-blue-300/15 bg-[#081827] p-2 sm:p-3">
          <div className="relative aspect-[1672/941] overflow-hidden rounded-lg bg-[#0b1c2a]">
            <img
              src="/technology-reid-matches.png"
              alt="Распознавание автомобилей на городском перекрёстке и найденные совпадения"
              width="1672"
              height="941"
              loading="lazy"
              className="h-full w-full object-cover"
            />
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
                onMouseEnter={() => setActiveMatch(index)}
                onMouseLeave={() => setActiveMatch(null)}
                onFocus={() => setActiveMatch(index)}
                onBlur={() => setActiveMatch(null)}
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
        <p className="mb-7 text-xs uppercase tracking-[.2em] text-muted-foreground">
          Нам доверяют
        </p>
        <div className="grid gap-7 sm:grid-cols-2 md:grid-cols-4">
          {trustCategories.map(({ label, icon: Icon }) => (
            <div
              key={label}
              className="flex items-center gap-3 text-sm text-slate-300"
            >
              <Icon className="h-7 w-7 text-blue-200/75" strokeWidth={1.5} />
              <span>{label}</span>
            </div>
          ))}
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
        <div>
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
                <div key={text as string} className="flex items-center gap-2">
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
                <code>{ex.code}</code>
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
      <div className="container max-w-7xl relative overflow-hidden rounded-2xl border border-blue-300/20 bg-[#081A2A]/50 px-7 py-12 sm:px-12">
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
          className="relative z-10 mt-8 bg-slate-50 text-slate-950 hover:bg-white sm:absolute sm:right-12 sm:top-1/2 sm:mt-0 sm:-translate-y-1/2"
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
          className="pointer-events-none absolute -bottom-4 left-1/2 w-[82%] -translate-x-1/2 opacity-30 sm:bottom-auto sm:left-[38%] sm:top-1/2 sm:w-[50%] sm:translate-x-0 sm:-translate-y-1/2 sm:opacity-65"
        />
      </div>
    </section>
  );
}

export function LandingFooter() {
  return (
    <footer className="border-t border-border py-9">
      <div className="container max-w-7xl">
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
            <Button
              variant="ghost"
              size="icon"
              disabled
              aria-label="GitHub — ссылка не настроена"
            >
              <Github className="h-5 w-5" />
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
