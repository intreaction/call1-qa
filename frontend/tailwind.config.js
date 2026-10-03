/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './evaluate.html',
    './store-console.html',
    './process.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Palette tokens resolve through the CSS custom properties in
        // src/index.css (#call1-theme-tokens), so every utility class follows
        // the active [data-theme] automatically. Colors using
        // `rgb(var(--x-rgb) / <alpha-value>)` also support Tailwind opacity
        // modifiers such as bg-canvas/60, border-border/60,
        // hover:bg-primer-redHover.
        canvas: {
          DEFAULT: 'rgb(var(--canvas-rgb) / <alpha-value>)',
          subtle: 'rgb(var(--canvas-subtle-rgb) / <alpha-value>)',
          inset: 'var(--canvas-inset)',
        },
        border: {
          DEFAULT: 'rgb(var(--border-rgb) / <alpha-value>)',
          muted: 'var(--border-muted)',
          control: 'var(--border-control)',
        },
        fg: {
          DEFAULT: 'var(--fg)',
          muted: 'var(--fg-muted)',
          subtle: 'var(--fg-subtle)',
        },
        primer: {
          // accent teal = primary/agent/seek
          blue: 'var(--primer-blue)',
          blueFg: 'var(--primer-blueFg)',
          blueHover: 'var(--primer-blueHover)',
          blueSubtle: 'var(--primer-blueSubtle)',
          blueBorder: 'var(--primer-blueBorder)',
          green: 'var(--primer-green)',
          greenFg: 'var(--primer-greenFg)',
          greenSubtle: 'var(--primer-greenSubtle)',
          greenBorder: 'var(--primer-greenBorder)',
          greenHover: 'var(--primer-greenHover)',
          red: 'var(--primer-red)',
          redFg: 'var(--primer-redFg)',
          redSubtle: 'var(--primer-redSubtle)',
          redBorder: 'var(--primer-redBorder)',
          redHover: 'var(--primer-redHover)',
          yellow: 'var(--primer-yellow)',
          yellowFg: 'var(--primer-yellowFg)',
          yellowSubtle: 'var(--primer-yellowSubtle)',
          yellowBorder: 'var(--primer-yellowBorder)',
          magenta: 'var(--primer-magenta)',
          magentaFg: 'var(--primer-magentaFg)',
          magentaSubtle: 'var(--primer-magentaSubtle)',
          magentaBorder: 'var(--primer-magentaBorder)',
          // caller = amber, agent = cyan
          caller: 'var(--primer-caller)',
          callerFg: 'var(--primer-callerFg)',
          callerSubtle: 'var(--primer-callerSubtle)',
          callerBorder: 'var(--primer-callerBorder)',
          agentBg: 'var(--primer-agentBg)',
          callerBg: 'var(--primer-callerBg)',
        },
      },
      fontFamily: {
        // IBM Plex Sans (UI), self-hosted under /fonts/
        sans: ['"IBM Plex Sans"', 'system-ui', '-apple-system', 'sans-serif'],
        // `mono` is deliberately NOT remapped here (judge condition 5, 2026-09-25): this config is
        // shared by all four Vite entries (index.html/legacy + the three new apps), so remapping it
        // changed the legacy app's `font-mono` typography (WaveformDeck, ToneTimeline,
        // TranscriptPane, views/EscalationsView) the next time anyone ran `npm run build`, even
        // though nothing in the legacy app was touched. "IBM Plex Sans only, no monospace"
        // (docs/SplitBuild.md Design principles) applies to Evaluate, the Store console and
        // Process, none of which use the `font-mono` utility class (grep before reaching for it —
        // it is plain Tailwind `ui-monospace` here, not the sans stack); their bare
        // `code`/`kbd`/`pre`/`samp` elements are handled per-entry in src/index.css instead
        // (`.call1-app code`, …), which does not touch legacy's own bare `<kbd>`/`<code>`.
      },
      fontWeight: {
        bold: '600', // remap font-bold→600; no fake-bold 700 face needed
      },
    },
  },
  plugins: [],
};
