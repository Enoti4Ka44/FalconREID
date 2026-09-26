import type { Config } from "tailwindcss"

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: { center: true, padding: "1.5rem", screens: { sm: "100%", md: "840px", lg: "calc(100% - 7.5rem)", xl: "calc(100% - 7.5rem)", "2xl": "1760px" } },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: { DEFAULT: "hsl(var(--primary))", foreground: "hsl(var(--primary-foreground))" },
        secondary: { DEFAULT: "hsl(var(--secondary))", foreground: "hsl(var(--secondary-foreground))" },
        muted: { DEFAULT: "hsl(var(--muted))", foreground: "hsl(var(--muted-foreground))" },
        accent: { DEFAULT: "hsl(var(--accent))", foreground: "hsl(var(--accent-foreground))" },
        card: { DEFAULT: "hsl(var(--card))", foreground: "hsl(var(--card-foreground))" },
      },
      borderRadius: { lg: "var(--radius)", md: "calc(var(--radius) - 2px)", sm: "calc(var(--radius) - 4px)" },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: [
          "Cascadia Code",
          "JetBrains Mono",
          "SFMono-Regular",
          "SF Mono",
          "Roboto Mono",
          "Consolas",
          "Liberation Mono",
          "Menlo",
          "monospace",
        ],
      },
      fontSize: {
        xs: ["0.875rem", { lineHeight: "1.25rem" }],
        sm: ["0.9375rem", { lineHeight: "1.375rem" }],
        base: ["1.0625rem", { lineHeight: "1.625rem" }],
        lg: ["1.1875rem", { lineHeight: "1.75rem" }],
        xl: ["1.3125rem", { lineHeight: "1.875rem" }],
        "2xl": ["1.5625rem", { lineHeight: "2rem" }],
      },
      keyframes: {
        float: { "0%, 100%": { transform: "translateY(0)" }, "50%": { transform: "translateY(-8px)" } },
        glow: { "0%, 100%": { opacity: ".45" }, "50%": { opacity: ".85" } },
      },
      animation: { float: "float 6s ease-in-out infinite", glow: "glow 4s ease-in-out infinite" },
    },
  },
  plugins: [],
} satisfies Config
