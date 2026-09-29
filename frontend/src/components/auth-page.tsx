import { useEffect, useMemo, useState, type FormEvent } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Eye,
  EyeOff,
  KeyRound,
  Loader2,
  LockKeyhole,
  Radar,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { safeNextPath, useAuth } from "@/lib/auth";

type Mode = "login" | "register";

export default function AuthPage() {
  const query = useMemo(() => new URLSearchParams(window.location.search), []);
  const [mode, setMode] = useState<Mode>(query.get("mode") === "register" ? "register" : "login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { user, loading, login, register } = useAuth();
  const nextPath = safeNextPath(query.get("next"));

  useEffect(() => {
    if (!loading && user) window.location.replace(nextPath);
  }, [loading, nextPath, user]);

  const switchMode = (nextMode: Mode) => {
    setMode(nextMode);
    setError(null);
    setPassword("");
    setConfirmation("");
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    const normalizedName = username.trim().toLowerCase();
    if (mode === "register" && normalizedName.length < 3) {
      setError("Имя пользователя должно содержать не меньше 3 символов.");
      return;
    }
    if (mode === "register" && password.length < 8) {
      setError("Пароль должен содержать не меньше 8 символов.");
      return;
    }
    if (mode === "register" && password !== confirmation) {
      setError("Пароли не совпадают.");
      return;
    }

    setBusy(true);
    try {
      if (mode === "login") await login(normalizedName, password);
      else await register(normalizedName, password);
      window.location.assign(nextPath);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Что-то пошло не так.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="auth-shell relative min-h-screen overflow-hidden bg-[#050d16] text-slate-100">
      <div className="am-breathe pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_18%_25%,rgba(66,144,230,.16),transparent_32%),radial-gradient(circle_at_84%_75%,rgba(27,89,151,.12),transparent_30%)]" />
      <div className="am-grid-drift pointer-events-none absolute inset-0 opacity-30 [background-image:linear-gradient(rgba(125,184,255,.045)_1px,transparent_1px),linear-gradient(90deg,rgba(125,184,255,.045)_1px,transparent_1px)] [background-size:52px_52px]" />
      <div className="am-beam" aria-hidden="true" />

      <div className="relative mx-auto grid min-h-screen max-w-[1440px] lg:grid-cols-[1.05fr_.95fr]">
        <section className="am-side hidden min-h-screen flex-col justify-between border-r border-white/[.07] p-14 lg:flex xl:p-20">
          <a href="/" className="flex w-fit items-center gap-3" aria-label="FALCON — на главную">
            <img src="/falcon-mark.svg" width="38" height="38" alt="" />
            <span className="text-lg font-semibold tracking-[.28em]">FALCON</span>
          </a>

          <div className="max-w-xl">
            <div className="mb-7 flex w-fit items-center gap-2 rounded-full border border-blue-200/[.12] bg-blue-300/[.05] px-3 py-2 text-[10px] font-semibold uppercase tracking-[.2em] text-blue-200/65">
              <Radar className="h-3.5 w-3.5" /> Vehicle intelligence platform
            </div>
            <h1 className="text-5xl font-medium leading-[1.08] tracking-[-.045em] xl:text-[62px]">
              Поиск автомобиля<br />в едином <span className="am-shine-text">пространстве.</span>
            </h1>
            <p className="mt-7 max-w-lg text-base leading-7 text-slate-400">
              Войдите в защищённый контур FALCON, чтобы запускать визуальный поиск и работать с галереей наблюдений.
            </p>
          </div>

          <div className="grid max-w-xl grid-cols-2 gap-3 text-xs text-slate-400">
            <div className="am-card rounded-xl border border-white/[.07] bg-white/[.025] p-4"><ShieldCheck className="mb-3 h-5 w-5 text-blue-300" />Каждый запрос к защищённому API подписывается JWT-токеном</div>
            <div className="am-card rounded-xl border border-white/[.07] bg-white/[.025] p-4"><LockKeyhole className="mb-3 h-5 w-5 text-blue-300" />Пароль передаётся только при входе и хранится как хеш</div>
          </div>
        </section>

        <section className="flex min-h-screen items-center justify-center px-5 py-12 sm:px-10 lg:px-16">
          <div className="am-form w-full max-w-[440px]">
            <div className="mb-10 flex items-center justify-between lg:hidden">
              <a href="/" className="flex items-center gap-3"><img src="/falcon-mark.svg" width="34" height="34" alt="" /><span className="font-semibold tracking-[.25em]">FALCON</span></a>
              <a href="/" className="flex items-center gap-2 text-xs text-slate-500 hover:text-slate-200"><ArrowLeft className="h-4 w-4" />На главную</a>
            </div>

            <a href="/" className="mb-9 hidden w-fit items-center gap-2 text-xs text-slate-500 transition hover:text-slate-200 lg:flex"><ArrowLeft className="h-4 w-4" />Вернуться на главную</a>
            <p className="mb-3 text-[10px] font-bold uppercase tracking-[.22em] text-blue-200/60">Доступ к платформе</p>
            <h2 className="text-3xl font-medium tracking-[-.035em]">{mode === "login" ? "С возвращением" : "Создать аккаунт"}</h2>
            <p className="mt-3 text-sm leading-6 text-slate-500">{mode === "login" ? "Введите данные аккаунта, чтобы продолжить работу." : "Регистрация займёт меньше минуты."}</p>

            <div className="mt-8 grid grid-cols-2 rounded-xl border border-white/[.08] bg-white/[.025] p-1">
              {([['login', 'Вход'], ['register', 'Регистрация']] as const).map(([value, label]) => (
                <button key={value} type="button" onClick={() => switchMode(value)} className={`rounded-lg px-4 py-2.5 text-sm transition ${mode === value ? "bg-blue-300/10 text-blue-100 shadow-[inset_0_0_0_1px_rgba(125,184,255,.18)]" : "text-slate-500 hover:text-slate-300"}`}>{label}</button>
              ))}
            </div>

            <form onSubmit={submit} className="mt-7 space-y-5">
              <label className="block">
                <span className="mb-2 block text-xs font-medium text-slate-400">Имя пользователя</span>
                <input required minLength={mode === "register" ? 3 : 1} maxLength={64} pattern={mode === "register" ? "[A-Za-zА-Яа-яЁё0-9._-]+" : undefined} autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} placeholder="например, falcon.operator" className="h-12 w-full rounded-xl border border-white/[.09] bg-[#07131f]/90 px-4 text-sm text-slate-100 outline-none transition placeholder:text-slate-700 focus:border-blue-300/45 focus:ring-4 focus:ring-blue-300/[.06]" />
              </label>
              <label className="block">
                <span className="mb-2 block text-xs font-medium text-slate-400">Пароль</span>
                <span className="relative block">
                  <input required minLength={mode === "register" ? 8 : 1} maxLength={mode === "register" ? 72 : 128} type={showPassword ? "text" : "password"} autoComplete={mode === "login" ? "current-password" : "new-password"} value={password} onChange={(event) => setPassword(event.target.value)} placeholder={mode === "register" ? "Не меньше 8 символов" : "Введите пароль"} className="h-12 w-full rounded-xl border border-white/[.09] bg-[#07131f]/90 px-4 pr-12 text-sm text-slate-100 outline-none transition placeholder:text-slate-700 focus:border-blue-300/45 focus:ring-4 focus:ring-blue-300/[.06]" />
                  <button type="button" onClick={() => setShowPassword((value) => !value)} className="absolute right-0 top-0 grid h-12 w-12 place-items-center text-slate-600 transition hover:text-slate-300" aria-label={showPassword ? "Скрыть пароль" : "Показать пароль"}>{showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}</button>
                </span>
              </label>
              {mode === "register" && (
                <label className="block">
                  <span className="mb-2 block text-xs font-medium text-slate-400">Повторите пароль</span>
                  <input required minLength={8} maxLength={72} type={showPassword ? "text" : "password"} autoComplete="new-password" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="Тот же пароль ещё раз" className="h-12 w-full rounded-xl border border-white/[.09] bg-[#07131f]/90 px-4 text-sm text-slate-100 outline-none transition placeholder:text-slate-700 focus:border-blue-300/45 focus:ring-4 focus:ring-blue-300/[.06]" />
                </label>
              )}

              {error && <div role="alert" className="rounded-xl border border-red-400/20 bg-red-500/[.07] px-4 py-3 text-sm leading-5 text-red-200">{error}</div>}

              <Button type="submit" disabled={busy || loading} className="h-12 w-full rounded-xl bg-blue-200 text-slate-950 hover:bg-blue-100">
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <>{mode === "login" ? "Войти" : "Создать аккаунт"}<ArrowRight className="h-4 w-4" /></>}
              </Button>

              {mode === "login" && (
                <div className="flex items-center justify-between gap-3 rounded-xl border border-blue-200/[.14] bg-blue-300/[.05] px-4 py-3">
                  <p className="flex items-center gap-2.5 text-xs leading-5 text-slate-400">
                    <KeyRound className="h-4 w-4 shrink-0 text-blue-300" />
                    <span>Тестовый доступ: логин <code className="rounded bg-white/[.06] px-1.5 py-0.5 font-mono text-blue-100">test</code> · пароль <code className="rounded bg-white/[.06] px-1.5 py-0.5 font-mono text-blue-100">test</code></span>
                  </p>
                  <button type="button" onClick={() => { setUsername("test"); setPassword("test"); setError(null); }} className="shrink-0 rounded-lg border border-blue-200/20 px-3 py-1.5 text-xs text-blue-100 transition hover:border-blue-200/40 hover:bg-blue-300/10">Подставить</button>
                </div>
              )}
            </form>

            {mode === "register" && <p className="mt-5 flex items-start gap-2 text-xs leading-5 text-slate-600"><Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-400" />После регистрации вход выполнится автоматически.</p>}
          </div>
        </section>
      </div>
    </main>
  );
}
