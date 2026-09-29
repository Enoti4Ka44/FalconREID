import { useEffect, useRef, useState } from "react";

// число в начале строки: «1 234», «87.2», «0,95»; дальше — единицы («%», « мс»)
const NUMERIC = /^(\d[\d\s ]*(?:[.,]\d+)?)(.*)$/s;
const grouped = new Intl.NumberFormat("ru-RU");

/** Число «набегает» к новому значению; нечисловые строки («—») выводятся как есть. */
export function CountUp({ value, duration = 900 }: { value: string | number; duration?: number }) {
  const text = String(value);
  const [shown, setShown] = useState(() => {
    const match = NUMERIC.exec(text);
    return match ? "0" + match[2] : text;
  });
  const last = useRef(0);

  useEffect(() => {
    const match = NUMERIC.exec(text);
    if (!match || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setShown(text);
      return;
    }
    const [, raw, suffix] = match;
    const sep = /[.,](\d+)$/.exec(raw);
    const decimals = sep ? sep[1].length : 0;
    const target = Number(raw.replace(/[\s ]/g, "").replace(",", "."));
    const format = (n: number) =>
      decimals ? n.toFixed(decimals).replace(".", sep![0][0]) : /[\s ]/.test(raw) ? grouped.format(Math.round(n)) : String(Math.round(n));
    const from = last.current;
    const start = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setShown(format(from + (target - from) * eased) + suffix);
      if (t < 1) frame = requestAnimationFrame(tick);
      else last.current = target;
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [text, duration]);

  return <>{shown}</>;
}
