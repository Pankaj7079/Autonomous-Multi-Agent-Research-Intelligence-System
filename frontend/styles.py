"""Design tokens and CSS injection. Every colour is defined here once — see docs/DESIGN.md."""

from __future__ import annotations

import streamlit as st

# a modern product palette: violet-tinted near-black, glass surfaces, indigo→fuchsia accent.
# status colours are reserved for status and are never used for decoration.
TOKENS = {
    "bg": "#0a0a12",
    "bg-2": "#0e0e1a",
    "text": "#f2f3f7",
    "dim": "#a4a9be",
    "faint": "#6d7390",
    "line": "rgba(255,255,255,0.09)",
    "line-2": "rgba(255,255,255,0.14)",
    "glass": "rgba(255,255,255,0.035)",
    "glass-2": "rgba(255,255,255,0.06)",
    "accent": "#818cf8",
    "accent-2": "#c084fc",
    "accent-3": "#22d3ee",
    "good": "#34d399",
    "warn": "#fbbf24",
    "bad": "#fb7185",
}

# below this a score is bad; the good floor comes from settings so the UI can never
# call something green that the critic sent back for revision
WARN_FLOOR = 0.55

SANS = "'Inter','Segoe UI Variable','Segoe UI',system-ui,-apple-system,sans-serif"
MONO = "'JetBrains Mono','SFMono-Regular',Consolas,'Liberation Mono',monospace"

_VARS = "\n".join(f"  --{name}: {value};" for name, value in TOKENS.items())

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
{_VARS}
  --sans: {SANS};
  --mono: {MONO};
  --grad: linear-gradient(135deg, #818cf8 0%, #c084fc 55%, #22d3ee 100%);
  --grad-soft: linear-gradient(135deg, rgba(129,140,248,0.16), rgba(192,132,252,0.10));
  --shadow: 0 8px 32px rgba(0,0,0,0.45);
  --shadow-lg: 0 16px 48px rgba(0,0,0,0.55);
}}

/* ── ground: deep base with two fixed colour glows ─────────── */
.stApp {{
  background:
    radial-gradient(900px 600px at 12% -8%, rgba(129,140,248,0.13), transparent 60%),
    radial-gradient(800px 560px at 92% 4%, rgba(192,132,252,0.10), transparent 62%),
    radial-gradient(700px 500px at 60% 100%, rgba(34,211,238,0.06), transparent 60%),
    var(--bg);
  background-attachment: fixed;
  color: var(--text);
  font-family: var(--sans);
  -webkit-font-smoothing: antialiased;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.1rem; padding-bottom: 4rem; max-width: 1360px; }}
.stApp, p, li, span, div {{ font-size: 0.875rem; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  font-size: 0.9rem; line-height: 1.7; color: var(--dim);
}}
[data-testid="stMarkdownContainer"] h1 {{ font-size: 1.4rem; letter-spacing: -0.02em; }}
[data-testid="stMarkdownContainer"] h2 {{
  font-size: 1.12rem; letter-spacing: -0.015em; margin: 1.1rem 0 0.5rem; color: var(--text);
}}
[data-testid="stMarkdownContainer"] h3 {{
  font-size: 0.98rem; margin: 0.9rem 0 0.35rem; color: var(--accent);
}}
[data-testid="stMarkdownContainer"] strong {{ color: var(--text); font-weight: 650; }}

section[data-testid="stSidebar"] {{
  background: rgba(10,10,20,0.72); backdrop-filter: blur(18px);
  border-right: 1px solid var(--line); width: 262px !important;
}}
section[data-testid="stSidebar"] .block-container {{ padding: 1.1rem 0.95rem; }}

@keyframes rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes glow {{ 0%,100% {{ opacity: 0.55; }} 50% {{ opacity: 1; }} }}
@keyframes shimmer {{ to {{ background-position: 200% center; }} }}

/* ── glass card, the one surface everything is built from ─── */
.card, .grid, .flow, .timeline, .log, .metric, .verdict, .src, .task, .trace, .boxed,
[data-testid="stForm"], [data-testid="stExpander"] details {{
  background: var(--glass) !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(14px);
  box-shadow: var(--shadow);
}}

/* ── top bar ──────────────────────────────────────────────── */
.topbar {{
  display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;
  background: var(--glass); border: 1px solid var(--line); border-radius: 14px;
  padding: 0.6rem 1rem; margin-bottom: 0.9rem; backdrop-filter: blur(14px);
  box-shadow: var(--shadow); animation: rise 0.45s ease both;
}}
.topbar .mark {{
  font-size: 1rem; font-weight: 800; letter-spacing: -0.02em;
  background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.topbar .what {{ color: var(--faint); font-size: 0.78rem; }}
.topbar .grow {{ flex: 1; }}

/* ── hero ─────────────────────────────────────────────────── */
.hero {{ padding: 0.4rem 0 1.1rem 0; animation: rise 0.5s ease both; }}
.hero .eyebrow {{
  display: inline-flex; align-items: center; gap: 0.45rem;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
  color: var(--accent); background: var(--grad-soft);
  border: 1px solid rgba(129,140,248,0.28); border-radius: 999px;
  padding: 0.3rem 0.75rem; margin-bottom: 0.9rem;
}}
.hero .eyebrow .pip {{
  width: 6px; height: 6px; border-radius: 50%; background: var(--good);
  box-shadow: 0 0 10px var(--good); animation: glow 2s infinite;
}}
/* fit-content, or the gradient spans the full container and only white lands on the text.
   block not inline-block, or the eyebrow badge above it flows onto the same line */
.hero .h1 {{
  display: block; width: fit-content;
  font-size: 2.7rem !important; font-weight: 800; letter-spacing: -0.04em; line-height: 1.08;
  margin: 0 0 0.75rem 0;
  background: linear-gradient(115deg, #ffffff 0%, #c7d2fe 32%, #c084fc 66%, #22d3ee 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.hero .lede {{
  color: var(--dim); font-size: 1rem; line-height: 1.65; max-width: 66ch; margin: 0;
}}
.hero .lede b {{ color: var(--text); font-weight: 600; }}

/* ── stat strip ───────────────────────────────────────────── */
.stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem; margin: 1.2rem 0 0.5rem; }}
.stat {{
  background: var(--glass); border: 1px solid var(--line); border-radius: 14px;
  padding: 0.85rem 1rem; backdrop-filter: blur(14px); animation: rise 0.5s ease both;
}}
.stat .n {{
  font-size: 1.55rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1;
  background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.stat .l {{ color: var(--faint); font-size: 0.74rem; font-weight: 500; margin-top: 0.15rem; }}

/* ── section heading ──────────────────────────────────────── */
.lbl {{
  display: flex; align-items: center; gap: 0.55rem;
  font-size: 1.02rem; font-weight: 650; letter-spacing: -0.015em; color: var(--text);
  margin: 1.9rem 0 0.35rem 0;
}}
.lbl .n {{
  font-family: var(--mono); font-size: 0.68rem; font-weight: 600; color: var(--accent);
  background: var(--grad-soft); border: 1px solid rgba(129,140,248,0.25);
  border-radius: 999px; padding: 0.12rem 0.55rem;
}}
.desc {{ color: var(--dim); font-size: 0.875rem; line-height: 1.65; margin: 0 0 0.9rem 0; max-width: 92ch; }}

/* ── chips ────────────────────────────────────────────────── */
.chip {{
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-size: 0.72rem; font-weight: 550; letter-spacing: 0.01em;
  padding: 0.3rem 0.68rem; border-radius: 999px;
  border: 1px solid var(--line-2); background: var(--glass-2); color: var(--dim);
  backdrop-filter: blur(10px);
}}
.chip .dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
.chip.on {{ color: var(--good); border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.1); }}
.chip.on .dot {{ box-shadow: 0 0 9px var(--good); animation: glow 2.4s infinite; }}
.chip.hot {{ color: var(--warn); border-color: rgba(251,191,36,0.32); background: rgba(251,191,36,0.1); }}
.chip.off {{ color: var(--faint); }}
.chip.accent {{ color: var(--accent); border-color: rgba(129,140,248,0.35); background: var(--grad-soft); }}

/* ── agent cards ──────────────────────────────────────────── */
.agents {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem; }}
.agent {{
  background: var(--glass); border: 1px solid var(--line); border-radius: 16px;
  padding: 1rem; backdrop-filter: blur(14px); box-shadow: var(--shadow);
  transition: transform 200ms ease, border-color 200ms ease, box-shadow 200ms ease;
  animation: rise 0.5s ease both; position: relative; overflow: hidden;
}}
.agent::before {{
  content: ""; position: absolute; inset: 0 0 auto 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
}}
.agent:hover {{
  transform: translateY(-3px); border-color: var(--line-2); box-shadow: var(--shadow-lg);
}}
.agent .badge {{
  width: 38px; height: 38px; border-radius: 11px; display: flex; align-items: center;
  justify-content: center; font-family: var(--mono); font-size: 0.72rem; font-weight: 700;
  color: #0b0b14; background: var(--grad); margin-bottom: 0.7rem;
  box-shadow: 0 6px 18px rgba(129,140,248,0.32);
}}
.agent.core .badge {{ background: linear-gradient(135deg, #c084fc, #f0abfc); }}
.agent .nm {{ font-size: 0.98rem; font-weight: 650; letter-spacing: -0.01em; margin-bottom: 0.1rem; }}
.agent .role {{
  font-family: var(--mono); font-size: 0.68rem; color: var(--accent-2); margin-bottom: 0.5rem;
}}
.agent .dec {{ color: var(--dim); font-size: 0.815rem; line-height: 1.55; }}
.agent .dec b {{ color: var(--text); font-weight: 600; }}

/* ── routing transcript ───────────────────────────────────── */
.trace {{
  font-family: var(--mono); font-size: 0.775rem; line-height: 2.05;
  padding: 1rem 1.15rem !important; overflow-x: auto;
}}
.trace .who {{ color: var(--accent-2); }}
.trace .why {{ color: var(--faint); }}
.trace .to {{ color: var(--good); font-weight: 600; }}
.trace .loop {{
  color: var(--warn); background: rgba(251,191,36,0.12);
  border: 1px solid rgba(251,191,36,0.25); border-radius: 6px; padding: 0.02rem 0.4rem;
  font-size: 0.7rem;
}}
.trace .arr {{ color: var(--faint); }}

/* ── pipeline flow ────────────────────────────────────────── */
.flow {{
  display: flex; align-items: stretch; overflow-x: auto; padding: 1rem 0.8rem !important;
}}
.flow .node {{
  display: flex; flex-direction: column; align-items: center; gap: 0.45rem;
  padding: 0 0.6rem; min-width: 92px;
}}
.flow .node .box {{
  width: 46px; height: 46px; border-radius: 14px; display: flex; align-items: center;
  justify-content: center; font-family: var(--mono); font-size: 0.68rem; font-weight: 700;
  border: 1px solid var(--line-2); background: var(--glass-2); color: var(--faint);
  transition: all 240ms ease;
}}
.flow .node .nm {{ font-size: 0.75rem; color: var(--faint); font-weight: 500; }}
.flow .node .hits {{
  font-family: var(--mono); font-size: 0.64rem; color: var(--warn); font-weight: 600;
}}
.flow .node.done .box {{
  border-color: rgba(52,211,153,0.35); color: #0b0b14;
  background: linear-gradient(135deg, #34d399, #10b981);
  box-shadow: 0 6px 18px rgba(52,211,153,0.28);
}}
.flow .node.done .nm {{ color: var(--text); }}
.flow .node.active .box {{
  border-color: transparent; color: #0b0b14; background: var(--grad);
  box-shadow: 0 0 0 4px rgba(129,140,248,0.18), 0 8px 24px rgba(129,140,248,0.4);
  animation: pulse 1.8s infinite;
}}
.flow .node.active .nm {{ color: var(--accent); font-weight: 650; }}
.flow .node.failed .box {{
  border-color: transparent; color: #0b0b14;
  background: linear-gradient(135deg, #fb7185, #f43f5e);
  box-shadow: 0 8px 24px rgba(251,113,133,0.35);
}}
.flow .node.failed .nm {{ color: var(--bad); }}
.flow .node.pending {{ opacity: 0.38; }}
.flow .sep {{
  display: flex; align-items: center; color: var(--line-2);
  padding-bottom: 1.6rem; font-size: 1.1rem;
}}
@keyframes pulse {{
  0%,100% {{ box-shadow: 0 0 0 4px rgba(129,140,248,0.18), 0 8px 24px rgba(129,140,248,0.4); }}
  50% {{ box-shadow: 0 0 0 9px rgba(129,140,248,0.06), 0 8px 30px rgba(129,140,248,0.55); }}
}}

/* ── run bar ──────────────────────────────────────────────── */
.runbar {{
  display: flex; align-items: center; gap: 0.85rem; flex-wrap: wrap;
  background: var(--glass); border: 1px solid var(--line); border-radius: 14px;
  padding: 0.7rem 1rem; margin-bottom: 0.75rem; font-size: 0.8rem; color: var(--dim);
  backdrop-filter: blur(14px);
}}
.runbar b {{ color: var(--text); font-weight: 650; font-family: var(--mono); }}
.runbar .sep {{ color: var(--line-2); }}
.runbar .live {{
  color: var(--accent); font-weight: 600; display: inline-flex; align-items: center; gap: 0.4rem;
}}
.runbar .live::before {{
  content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 10px var(--accent); animation: glow 1.4s infinite;
}}

/* ── timeline ─────────────────────────────────────────────── */
.timeline {{ overflow: hidden; }}
.tl-row {{
  display: grid; grid-template-columns: 14px 116px 62px 1fr; gap: 0.8rem; align-items: center;
  padding: 0.6rem 1rem; border-top: 1px solid var(--line); font-size: 0.845rem;
  transition: background 160ms ease;
}}
.tl-row:first-child {{ border-top: none; }}
.tl-row:hover {{ background: rgba(255,255,255,0.025); }}
.tl-row .dot {{
  width: 9px; height: 9px; border-radius: 50%; background: var(--line-2); justify-self: center;
}}
.tl-row .nm {{ font-weight: 600; letter-spacing: -0.01em; }}
.tl-row .at {{
  font-family: var(--mono); font-size: 0.74rem; color: var(--faint); text-align: right;
}}
.tl-row .msg {{ color: var(--dim); }}
.tl-row.done .dot {{ background: var(--good); box-shadow: 0 0 10px rgba(52,211,153,0.55); }}
.tl-row.active {{ background: rgba(129,140,248,0.07); }}
.tl-row.active .dot {{
  background: var(--accent); box-shadow: 0 0 12px var(--accent); animation: glow 1.5s infinite;
}}
.tl-row.active .nm {{ color: var(--accent); }}
.tl-row.failed .dot {{ background: var(--bad); box-shadow: 0 0 10px rgba(251,113,133,0.55); }}
.tl-row.failed .nm {{ color: var(--bad); }}
.tl-row.pending {{ opacity: 0.34; }}

/* ── log ──────────────────────────────────────────────────── */
.log {{
  font-family: var(--mono); font-size: 0.76rem; line-height: 1.95;
  padding: 0.9rem 1.1rem !important; max-height: 320px; overflow-y: auto;
  background: rgba(0,0,0,0.32) !important;
}}
.log .t {{ color: var(--faint); }}
.log .a {{ color: var(--accent); }}
.log .m {{ color: var(--dim); }}
.log .ok {{ color: var(--good); }}
.log .err {{ color: var(--bad); }}

/* ── metrics ──────────────────────────────────────────────── */
.metrics {{ display: flex; flex-wrap: wrap; gap: 0.7rem; margin: 0.3rem 0 1rem 0; }}
.metric {{
  flex: 1 1 128px; padding: 0.85rem 1rem !important; animation: rise 0.45s ease both;
  transition: transform 200ms ease, border-color 200ms ease;
}}
.metric:hover {{ transform: translateY(-2px); border-color: var(--line-2) !important; }}
.metric .k {{
  color: var(--faint); font-size: 0.72rem; font-weight: 500; letter-spacing: 0.01em;
}}
.metric .v {{
  font-size: 1.5rem; font-weight: 800; letter-spacing: -0.03em; margin-top: 0.1rem;
  background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.metric.good .v {{ background: linear-gradient(135deg,#34d399,#10b981); -webkit-background-clip: text; background-clip: text; }}
.metric.warn .v {{ background: linear-gradient(135deg,#fbbf24,#f59e0b); -webkit-background-clip: text; background-clip: text; }}
.metric.bad .v {{ background: linear-gradient(135deg,#fb7185,#f43f5e); -webkit-background-clip: text; background-clip: text; }}

/* ── verdict ──────────────────────────────────────────────── */
.verdict {{
  display: flex; align-items: center; gap: 0.95rem; padding: 1rem 1.15rem !important;
  margin-bottom: 0.9rem; animation: rise 0.45s ease both;
}}
.verdict .ic {{
  width: 38px; height: 38px; border-radius: 12px; display: flex; align-items: center;
  justify-content: center; font-size: 1.05rem; font-weight: 700; flex: 0 0 auto; color: #0b0b14;
}}
.verdict b {{ display: block; font-size: 0.98rem; font-weight: 650; letter-spacing: -0.015em; }}
.verdict .sub {{ color: var(--dim); font-size: 0.78rem; font-family: var(--mono); }}
.verdict.good {{ border-color: rgba(52,211,153,0.3) !important; }}
.verdict.good .ic {{ background: linear-gradient(135deg,#34d399,#10b981); box-shadow: 0 6px 20px rgba(52,211,153,0.32); }}
.verdict.good b {{ color: var(--good); }}
.verdict.warn {{ border-color: rgba(251,191,36,0.3) !important; }}
.verdict.warn .ic {{ background: linear-gradient(135deg,#fbbf24,#f59e0b); box-shadow: 0 6px 20px rgba(251,191,36,0.3); }}
.verdict.warn b {{ color: var(--warn); }}
.verdict.bad {{ border-color: rgba(251,113,133,0.3) !important; }}
.verdict.bad .ic {{ background: linear-gradient(135deg,#fb7185,#f43f5e); box-shadow: 0 6px 20px rgba(251,113,133,0.32); }}
.verdict.bad b {{ color: var(--bad); }}

/* ── tables ───────────────────────────────────────────────── */
.tbl {{
  width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.83rem;
  background: var(--glass); border: 1px solid var(--line); border-radius: 16px;
  overflow: hidden; backdrop-filter: blur(14px);
}}
.tbl th {{
  text-align: left; padding: 0.7rem 0.95rem; color: var(--faint);
  background: rgba(255,255,255,0.03); font-weight: 600; font-size: 0.73rem;
  border-bottom: 1px solid var(--line);
}}
.tbl td {{ padding: 0.62rem 0.95rem; border-bottom: 1px solid var(--line); color: var(--dim); }}
.tbl tr:last-child td {{ border-bottom: none; }}
.tbl tbody tr {{ transition: background 160ms ease; }}
.tbl tbody tr:hover {{ background: rgba(255,255,255,0.028); }}
.tbl b {{ color: var(--text); font-weight: 650; }}
.tbl code, .mono {{
  font-family: var(--mono); background: rgba(255,255,255,0.06); color: var(--accent);
  padding: 0.12rem 0.4rem; border-radius: 6px; font-size: 0.73rem;
}}
.diverged {{
  color: var(--warn); background: rgba(251,191,36,0.13);
  border: 1px solid rgba(251,191,36,0.28); border-radius: 6px;
  padding: 0.04rem 0.42rem; font-size: 0.68rem; font-weight: 600;
}}
.unscored {{ color: var(--faint); font-style: italic; }}
.v-good {{ color: var(--good); font-weight: 700; font-family: var(--mono); }}
.v-warn {{ color: var(--warn); font-weight: 700; font-family: var(--mono); }}
.v-bad {{ color: var(--bad); font-weight: 700; font-family: var(--mono); }}

/* ── sources ──────────────────────────────────────────────── */
.src {{
  padding: 0.8rem 1rem !important; margin-bottom: 0.55rem;
  transition: transform 180ms ease, border-color 180ms ease;
}}
.src:hover {{ transform: translateX(3px); border-color: var(--line-2) !important; }}
.src .hd {{ display: flex; gap: 0.55rem; align-items: baseline; }}
.src .n {{
  font-family: var(--mono); font-size: 0.72rem; color: var(--accent);
  background: var(--grad-soft); border-radius: 6px; padding: 0.05rem 0.42rem; flex: 0 0 auto;
}}
.src a {{ color: var(--text); text-decoration: none; font-size: 0.9rem; font-weight: 600; }}
.src a:hover {{ color: var(--accent); }}
.src .u {{
  font-family: var(--mono); font-size: 0.7rem; color: var(--faint);
  word-break: break-all; margin-top: 0.2rem;
}}
.src .s {{ color: var(--dim); font-size: 0.82rem; line-height: 1.6; margin-top: 0.45rem; }}

/* ── key/value ────────────────────────────────────────────── */
.boxed {{ padding: 0.35rem 1rem !important; }}
.kv {{
  display: flex; justify-content: space-between; gap: 0.8rem; align-items: baseline;
  font-size: 0.8rem; padding: 0.48rem 0; border-bottom: 1px solid var(--line);
}}
.kv:last-child {{ border-bottom: none; }}
.kv .k {{ color: var(--faint); flex: 0 0 auto; font-weight: 500; }}
/* min-width:0 or the value refuses to shrink and overflows, clipping its tail */
.kv .v {{
  color: var(--text); text-align: right; min-width: 0; flex: 1 1 auto;
  overflow-wrap: anywhere; font-family: var(--mono); font-size: 0.76rem;
}}

/* ── plan tasks ───────────────────────────────────────────── */
.task {{
  display: grid; grid-template-columns: 44px 1fr; gap: 0.85rem; align-items: start;
  padding: 0.85rem 1rem !important; margin-bottom: 0.5rem;
  transition: transform 180ms ease, border-color 180ms ease;
}}
.task:hover {{ transform: translateX(3px); border-color: var(--line-2) !important; }}
.task .id {{
  font-family: var(--mono); font-size: 0.7rem; font-weight: 700; color: #0b0b14;
  background: var(--grad); border-radius: 8px; padding: 0.22rem 0; text-align: center;
  box-shadow: 0 4px 14px rgba(129,140,248,0.28);
}}
.task .w {{ font-size: 0.9rem; color: var(--text); line-height: 1.5; font-weight: 500; }}
.task .m {{ font-family: var(--mono); font-size: 0.7rem; color: var(--faint); margin-top: 0.2rem; }}

/* ── sidebar ──────────────────────────────────────────────── */
/* the section heading is a page-level size; in a 262px rail it has to come down */
section[data-testid="stSidebar"] .lbl {{
  font-size: 0.78rem; font-weight: 600; color: var(--faint);
  text-transform: uppercase; letter-spacing: 0.08em; margin: 1.3rem 0 0.5rem 0;
}}
section[data-testid="stSidebar"] .chip {{ font-size: 0.68rem; padding: 0.22rem 0.5rem; }}
.sb-mark {{
  font-size: 1.2rem; font-weight: 800; letter-spacing: -0.03em;
  background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.sb-sub {{ color: var(--faint); font-size: 0.72rem; margin-bottom: 0.5rem; }}
.hist-m {{ color: var(--faint); font-size: 0.7rem; font-family: var(--mono); margin: -0.3rem 0 0.6rem 0.2rem; }}

/* ── streamlit widgets, pulled into the language ──────────── */
.stTextInput input {{
  background: rgba(255,255,255,0.045) !important; border: 1px solid var(--line-2) !important;
  color: var(--text) !important; font-size: 0.95rem !important; font-family: var(--sans) !important;
  border-radius: 13px !important; padding: 0.8rem 1rem !important;
}}
.stTextInput input:focus {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 4px rgba(129,140,248,0.16) !important;
}}
.stTextInput input::placeholder {{ color: var(--faint) !important; }}
[data-testid="stForm"] {{ padding: 0.85rem 0.9rem !important; }}
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {{ align-items: flex-end; gap: 0.6rem; }}

.stButton > button, [data-testid="stBaseButton-secondary"] {{
  font-family: var(--sans) !important; font-size: 0.82rem !important; font-weight: 550 !important;
  border-radius: 11px !important; padding: 0.6rem 0.9rem !important;
  border: 1px solid var(--line-2) !important; background: var(--glass-2) !important;
  color: var(--dim) !important; transition: all 180ms ease !important; min-height: 0 !important;
  backdrop-filter: blur(10px);
}}
.stButton > button:hover, [data-testid="stBaseButton-secondary"]:hover {{
  border-color: rgba(129,140,248,0.5) !important; color: var(--text) !important;
  background: var(--grad-soft) !important; transform: translateY(-2px);
}}
/* streamlit names the form's submit button primaryFormSubmit, not primary */
button[kind="primary"], button[kind="primaryFormSubmit"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{
  font-family: var(--sans) !important; font-size: 0.9rem !important; font-weight: 650 !important;
  border-radius: 13px !important; padding: 0.82rem 1.5rem !important; border: none !important;
  background: var(--grad) !important; background-size: 200% auto !important; color: #0b0b14 !important;
  box-shadow: 0 8px 26px rgba(129,140,248,0.42) !important; transition: all 220ms ease !important;
}}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover,
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {{
  transform: translateY(-2px); box-shadow: 0 12px 34px rgba(129,140,248,0.55) !important;
  animation: shimmer 1.6s linear infinite;
}}
/* the label sits in a nested div that otherwise keeps the page font size and colour */
.stButton button *, [data-testid^="stBaseButton"] * {{
  font-size: inherit !important; font-family: inherit !important; color: inherit !important;
  font-weight: inherit !important;
}}
.stDownloadButton > button {{
  font-family: var(--sans) !important; font-size: 0.8rem !important;
  border-radius: 11px !important; border: 1px solid var(--line-2) !important;
  background: var(--glass-2) !important; color: var(--dim) !important;
}}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 0.35rem; border-bottom: 1px solid var(--line); padding-bottom: 0.15rem;
}}
[data-testid="stTabs"] button {{
  font-family: var(--sans) !important; font-size: 0.86rem !important; font-weight: 550 !important;
  color: var(--faint) !important; padding: 0.55rem 0.95rem !important; border-radius: 11px 11px 0 0;
}}
[data-testid="stTabs"] button:hover {{ color: var(--dim) !important; background: rgba(255,255,255,0.03); }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color: var(--accent) !important; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: var(--grad) !important; height: 2px; }}

[data-testid="stExpander"] details {{ border-radius: 14px !important; }}
[data-testid="stExpander"] summary {{ font-size: 0.85rem; font-weight: 550; color: var(--dim); }}
[data-testid="stCaptionContainer"] {{ color: var(--faint) !important; font-size: 0.78rem !important; }}
[data-testid="stProgress"] > div > div > div {{ background: var(--grad) !important; }}
[data-testid="stAlert"] {{ border-radius: 13px !important; backdrop-filter: blur(12px); }}
code {{ font-family: var(--mono) !important; font-size: 0.78rem !important; color: var(--accent) !important; }}
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.12); border-radius: 999px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.2); }}
</style>
"""


def inject_css() -> None:
    """Write the token sheet into the page. Safe to call on every rerun."""
    st.markdown(_CSS, unsafe_allow_html=True)


def badge_class(score: float, good_floor: float) -> str:
    """good / warn / bad. good_floor is the critic's approval threshold, not a UI constant."""
    if score >= good_floor:
        return "good"
    return "warn" if score >= WARN_FLOOR else "bad"
