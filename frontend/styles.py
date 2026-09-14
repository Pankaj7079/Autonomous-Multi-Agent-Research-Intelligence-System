"""Design tokens and CSS injection. Every colour is defined here once — see docs/DESIGN.md."""

from __future__ import annotations

import html

import streamlit as st

# warm paper ground, ink text, one deep-teal accent. Light on purpose: the page is read for
# minutes at a time. Status colours are reserved for status and never used for decoration.
TOKENS = {
    "bg": "#faf9f7",
    "bg-2": "#f2efea",
    "surface": "#ffffff",
    "surface-2": "#f6f4f1",
    "surface-3": "#edeae5",
    "line": "#e7e2db",
    "line-2": "#d6cfc5",
    "text": "#1c1a17",
    "dim": "#5b5650",
    "faint": "#8a837a",
    "ghost": "#aaa298",
    "accent": "#0f766e",
    "accent-2": "#9a5b2d",
    "accent-dim": "#7fc4bd",
    "accent-soft": "#e6f2f0",
    "good": "#15803d",
    "warn": "#b45309",
    "bad": "#be123c",
}

# below this a score is bad; the good floor comes from settings so the UI can never
# call something green that the critic sent back for revision
WARN_FLOOR = 0.55

SANS = "'Inter','Segoe UI Variable','Segoe UI',system-ui,-apple-system,sans-serif"
MONO = "'JetBrains Mono','SFMono-Regular',Consolas,'Liberation Mono',monospace"

_VARS = "\n".join(f"  --{name}: {value};" for name, value in TOKENS.items())

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
{_VARS}
  --sans: {SANS};
  --mono: {MONO};
  --r: 10px;
  --r-sm: 6px;
  --r-lg: 14px;
  /* on paper, depth comes from a soft drop shadow rather than an inset highlight */
  --lift: 0 1px 2px rgba(28,26,23,0.04), 0 1px 3px rgba(28,26,23,0.03);
  --lift-2: 0 2px 4px rgba(28,26,23,0.05), 0 8px 24px rgba(28,26,23,0.05);
  --ease: 140ms cubic-bezier(0.4, 0, 0.2, 1);
}}

/* ═══ ground ═══════════════════════════════════════════════════════════════ */
.stApp {{
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 0.875rem;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}}
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}

.stMain .block-container {{ padding: 0 2rem 6.5rem; max-width: 1040px; }}
::selection {{ background: #cfe8e4; color: var(--text); }}

/* streamlit stacks every element with a 1rem gap, which is what makes an app look
   like a form. the rhythm is set per component instead. */
[data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}
[data-testid="stHorizontalBlock"] {{ gap: 0.45rem; }}

/* the base size is set on .stApp alone and everything inherits it. listing bare span/div/p
   here instead set the size ON each element, which beats inheritance — that is what rendered
   the hero mark as a 96px "AMA" with a 14px "RIS" stuck to it. never reset a bare tag. */
/* every number in this ui is data, so digits must not jitter between frames */
.mono, .metric .v, .tl-row .at, .runbar, .tbl, .kv .v, .turn-meta, .vchip {{
  font-variant-numeric: tabular-nums;
}}

/* ═══ sidebar ══════════════════════════════════════════════════════════════ */
/* the sidebar is part of the console, not a drawer: it carries provider health and the whole
   thread. collapsing it hid the reopen arrow with the header, so there is no collapse at all —
   these override streamlit's own collapsed state, whatever it sets */
section[data-testid="stSidebar"] {{
  background: var(--bg-2);
  border-right: 1px solid var(--line);
  width: 236px !important;
  min-width: 236px !important;
  max-width: 236px !important;
  transform: none !important;
  visibility: visible !important;
  margin-left: 0 !important;
}}
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarNavCollapseButton"] {{ display: none !important; }}
section[data-testid="stSidebar"] .block-container {{ padding: 1.4rem 0.85rem 2rem; }}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.3rem; }}

/* ═══ wordmark ═════════════════════════════════════════════════════════════ */
/* no flex gap: "AMA" and the RIS span are separate flex items, so any gap here splits the
   name into two words. the only space in the mark is the diamond's own margin. */
.wordmark {{
  display: inline-flex; align-items: baseline; gap: 0;
  font-weight: 800; letter-spacing: -0.035em; color: var(--text);
  line-height: 1;
}}
.wordmark .dotmark {{
  width: 7px; height: 7px; border-radius: 2px; background: var(--accent);
  transform: rotate(45deg); align-self: center; margin-right: 0.15rem;
}}
/* both halves of the name are one size, always — the mark's size is set on .wordmark only */
.wordmark span {{ font-size: inherit; }}
.wordmark .ris {{ color: var(--accent); }}

/* the acronym on its own means nothing to a first-time reader — the full name sits
   right under it, small, everywhere the mark appears at a readable size.
   nowrap here previously forced a box wider than the viewport at 51 characters, which
   Streamlit's own overflow-x:hidden then clipped — taking the sidebar off-screen with it.
   bounding the width and letting it wrap two lines is what actually fits any screen. */
.wm-group {{
  display: inline-flex; flex-direction: column; align-items: flex-start; gap: 0.2rem;
  max-width: 100%;
}}
.wm-full {{
  font-family: var(--mono); font-size: 0.6rem; font-weight: 500; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--ghost); max-width: 240px; line-height: 1.5;
}}
/* the group itself stays unbounded — it must size to the big mark, not to the caption.
   only the caption gets a width cap, so it wraps to two short lines instead of stretching
   the whole flex column wider than the mark (and, upstream, wider than the viewport) */
.hero .wm-group {{ align-items: center; gap: 0.55rem; }}
.hero .wm-full {{
  font-size: 0.72rem; letter-spacing: 0.07em; color: var(--faint);
  text-align: center; line-height: 1.6; max-width: min(90vw, 420px);
}}

/* three chips wrap ragged in a 236px column — one per row reads as a status list instead.
   styled on the chip itself, not a wrapper div: a block wrapper here measured 11px short of
   its own content and streamlit printed the next element on top of it */
section[data-testid="stSidebar"] .chip {{
  display: flex; width: 100%; margin: 0 0 0.3rem;
}}

.sb-mark {{ font-size: 1.32rem; }}
.sb-sub {{
  font-family: var(--mono); font-size: 0.63rem; color: var(--ghost);
  letter-spacing: 0.1em; text-transform: uppercase; margin: 0.4rem 0 1.3rem 0.05rem;
}}
.hist-m {{
  font-family: var(--mono); font-size: 0.62rem; color: var(--ghost);
  margin: -0.15rem 0 0.5rem 0.6rem; letter-spacing: 0.04em;
}}

/* ═══ hero ═════════════════════════════════════════════════════════════════ */
.hero {{
  display: flex; flex-direction: column; align-items: center;
  text-align: center; padding: 2.1rem 0 1rem;
}}
.hero-mark {{
  font-size: clamp(3.6rem, 9vw, 6rem);
  font-weight: 800; letter-spacing: -0.055em;
}}
.hero-mark .dotmark {{
  width: 19px; height: 19px; border-radius: 5px; margin-right: 0.36rem;
}}
.tagline {{
  margin: 1.4rem 0 0; font-size: 1.12rem; color: var(--dim);
  font-weight: 400; letter-spacing: -0.008em; max-width: 44rem; line-height: 1.55;
}}
.eyebrow {{
  display: inline-flex; align-items: center; gap: 0.5rem;
  font-family: var(--mono); font-size: 0.63rem; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--faint);
  border: 1px solid var(--line); background: var(--surface);
  border-radius: 999px; padding: 0.3rem 0.8rem; margin-top: 1.5rem;
  box-shadow: var(--lift);
}}
.pip {{
  width: 5px; height: 5px; border-radius: 50%; background: var(--good);
  box-shadow: 0 0 0 3px rgba(21,128,61,0.13);
}}

/* ═══ stat strip ═══════════════════════════════════════════════════════════ */
.stats {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
  background: var(--line); border: 1px solid var(--line);
  border-radius: var(--r); overflow: hidden; box-shadow: var(--lift);
  margin: 1.6rem 0 0.5rem;
}}
.stat {{ background: var(--surface); padding: 1rem 1.1rem; text-align: center; }}
.stat .n {{
  font-size: 1.7rem; font-weight: 800; letter-spacing: -0.035em;
  line-height: 1.05; color: var(--accent); font-variant-numeric: tabular-nums;
}}
.stat .l {{
  font-family: var(--mono); font-size: 0.63rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--ghost); margin-top: 0.35rem;
}}

/* ═══ agent grid ═══════════════════════════════════════════════════════════ */
.agents {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.55rem; }}
.agent {{
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
  padding: 0.9rem 0.95rem; box-shadow: var(--lift);
  transition: border-color var(--ease), box-shadow var(--ease);
}}
.agent:hover {{ border-color: var(--accent-dim); box-shadow: var(--lift-2); }}
.agent .badge {{
  display: inline-grid; place-items: center; width: 30px; height: 30px;
  border-radius: var(--r-sm); font-family: var(--mono); font-size: 0.6rem;
  font-weight: 600; letter-spacing: 0.05em; margin-bottom: 0.6rem;
  color: var(--accent); background: var(--accent-soft); border: 1px solid #cde6e2;
}}
/* the supervisor is the one agent whose whole job is deciding, so it is marked */
.agent.core .badge {{ color: #ffffff; background: var(--accent); border-color: var(--accent); }}
.agent .nm {{ font-size: 0.88rem; font-weight: 700; letter-spacing: -0.012em; }}
.agent .role {{
  font-family: var(--mono); font-size: 0.63rem; color: var(--accent-2);
  margin: 0.1rem 0 0.45rem; letter-spacing: 0.03em;
}}
.agent .dec {{ color: var(--faint); font-size: 0.775rem; line-height: 1.55; }}

/* ═══ section heading ══════════════════════════════════════════════════════ */
.lbl {{
  display: flex; align-items: center; gap: 0.55rem; margin: 2rem 0 0.35rem;
  font-family: var(--mono); font-size: 0.67rem; letter-spacing: 0.15em;
  text-transform: uppercase; color: var(--faint); font-weight: 500;
}}
.lbl::after {{ content: ''; flex: 1; height: 1px; background: var(--line); }}
.lbl .n {{
  font-size: 0.61rem; padding: 0.1rem 0.45rem; border-radius: 999px;
  background: var(--surface-3); color: var(--faint); letter-spacing: 0.06em;
}}
.desc {{
  color: var(--faint); font-size: 0.79rem; line-height: 1.65;
  margin-bottom: 0.9rem; max-width: 76ch;
}}

/* ═══ chips ════════════════════════════════════════════════════════════════ */
.chip {{
  display: inline-flex; align-items: center; gap: 0.35rem; font-family: var(--mono);
  font-size: 0.67rem; padding: 0.24rem 0.58rem; border-radius: var(--r-sm);
  border: 1px solid var(--line); background: var(--surface); color: var(--dim);
  margin: 0 0.28rem 0.28rem 0; letter-spacing: 0.03em; box-shadow: var(--lift);
}}
.chip .dot {{ width: 5px; height: 5px; border-radius: 50%; background: currentColor; }}
.chip.on {{ color: var(--good); }}
.chip.hot {{ color: var(--warn); }}
.chip.off {{ color: var(--ghost); }}

/* ═══ the turn card — the centrepiece ══════════════════════════════════════ */
.turn {{
  position: relative; overflow: hidden;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  box-shadow: var(--lift-2);
  padding: 1.3rem 1.5rem 1.1rem;
  margin-top: 1.2rem;
}}
.turn::before {{
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--accent); opacity: 0;
}}
.turn.current::before {{ opacity: 1; }}
.turn.past {{ box-shadow: var(--lift); background: var(--surface-2); }}

.turn-head {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.55rem; }}
.turn-head .who {{
  font-family: var(--mono); font-size: 0.61rem; letter-spacing: 0.17em;
  text-transform: uppercase; color: var(--ghost);
}}
.turn-head .rule {{ flex: 1; height: 1px; background: var(--line); }}
.depth-chip {{
  font-family: var(--mono); font-size: 0.61rem; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--accent);
  background: var(--accent-soft); border: 1px solid #cde6e2;
  border-radius: 999px; padding: 0.14rem 0.58rem;
}}
.turn-q {{
  font-size: 1.22rem; font-weight: 700; color: var(--text); line-height: 1.34;
  letter-spacing: -0.022em; margin-bottom: 1.15rem;
}}

/* the rendered report lives here, so every markdown element is styled explicitly */
.turn-a {{ color: var(--dim); font-size: 0.92rem; line-height: 1.75; }}
.turn-a > *:first-child {{ margin-top: 0; }}
.turn-a > *:last-child {{ margin-bottom: 0; }}
.turn-a h2 {{
  font-family: var(--mono); font-size: 0.65rem; font-weight: 600;
  letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent);
  margin: 1.7rem 0 0.6rem; padding-bottom: 0.45rem; border-bottom: 1px solid var(--line);
}}
.turn-a h3 {{
  font-size: 0.88rem; font-weight: 700; color: var(--text); margin: 1.2rem 0 0.35rem;
  letter-spacing: -0.01em;
}}
.turn-a p {{ margin: 0 0 0.9rem; }}
.turn-a ul, .turn-a ol {{ margin: 0 0 0.95rem; padding-left: 1.2rem; }}
.turn-a li {{ margin-bottom: 0.45rem; }}
.turn-a li::marker {{ color: var(--accent-dim); }}
.turn-a strong {{ color: var(--text); font-weight: 650; }}
.turn-a a {{
  color: var(--accent); text-decoration: none; border-bottom: 1px solid #b9ded9;
}}
.turn-a a:hover {{ border-bottom-color: var(--accent); }}
.turn-a code {{
  font-family: var(--mono); font-size: 0.8rem; color: var(--accent-2);
  background: var(--surface-3); padding: 0.1rem 0.35rem; border-radius: 4px;
}}
.turn-a pre {{
  background: var(--surface-2); border: 1px solid var(--line); border-radius: var(--r-sm);
  padding: 0.85rem 1rem; overflow-x: auto; margin: 0 0 0.95rem;
}}
.turn-a pre code {{ background: none; padding: 0; color: var(--dim); }}
.turn-a blockquote {{
  margin: 0 0 0.95rem; padding-left: 0.95rem; border-left: 3px solid var(--accent-dim);
  color: var(--faint);
}}
.turn-a table {{ width: 100%; border-collapse: collapse; font-size: 0.83rem; margin-bottom: 0.95rem; }}
.turn-a th, .turn-a td {{ padding: 0.45rem 0.65rem; border-bottom: 1px solid var(--line); text-align: left; }}
.turn-a th {{ color: var(--faint); font-weight: 600; }}
/* the reference block is one paragraph of hard-broken lines; no `h2 + p` rule here, because
   that also matches the answer paragraph and would set the actual answer in 8pt mono */
.turn-a br {{ line-height: 2.15; }}

/* instrument strip at the foot of a turn */
.turn-meta {{
  display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;
  margin-top: 1.2rem; padding-top: 0.85rem; border-top: 1px solid var(--line);
  font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.03em;
}}
.turn-meta .m {{ color: var(--ghost); }}
.turn-meta .m b {{ color: var(--dim); font-weight: 600; margin-right: 0.22rem; }}
.turn-meta .bar {{ width: 1px; height: 11px; background: var(--line-2); }}
.vchip {{
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-size: 0.7rem; font-weight: 600; padding: 0.22rem 0.6rem;
  border-radius: var(--r-sm); letter-spacing: 0.02em;
}}
.vchip .dot {{ width: 5px; height: 5px; border-radius: 50%; background: currentColor; }}
.vchip.good {{ color: var(--good); background: #e7f5ec; }}
.vchip.warn {{ color: var(--warn); background: #fbf0e2; }}
.vchip.bad {{ color: var(--bad); background: #fceaee; }}

/* ═══ pipeline flow ════════════════════════════════════════════════════════ */
.flow {{
  display: flex; align-items: flex-start; gap: 0.15rem;
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
  box-shadow: var(--lift); padding: 1.05rem 0.95rem 0.9rem; overflow-x: auto;
}}
.flow .node {{ text-align: center; min-width: 60px; }}
.flow .box {{
  width: 38px; height: 38px; margin: 0 auto; border-radius: var(--r-sm);
  display: grid; place-items: center; font-family: var(--mono); font-size: 0.6rem;
  font-weight: 600; letter-spacing: 0.05em;
  border: 1px solid var(--line); background: var(--surface-2); color: var(--ghost);
  transition: all var(--ease);
}}
.flow .node .nm {{
  display: block; font-size: 0.64rem; color: var(--ghost); margin-top: 0.4rem;
  font-family: var(--mono); letter-spacing: 0.03em;
}}
.flow .node .hits {{
  display: block; font-family: var(--mono); font-size: 0.6rem; color: var(--warn);
  margin-top: 0.1rem;
}}
.flow .node.done .box {{
  border-color: #cde6e2; background: var(--accent-soft); color: var(--accent);
}}
.flow .node.done .nm {{ color: var(--dim); }}
.flow .node.active .box {{
  border-color: var(--accent); color: var(--accent); background: var(--surface);
  box-shadow: 0 0 0 3px rgba(15,118,110,0.12);
  animation: breathe 1.6s ease-in-out infinite;
}}
.flow .node.failed .box {{ border-color: #f0c2cc; color: var(--bad); background: #fceaee; }}
.flow .sep {{ color: var(--line-2); margin-top: 0.78rem; font-size: 0.75rem; }}
@keyframes breathe {{ 0%,100% {{ opacity: 0.7; }} 50% {{ opacity: 1; }} }}

/* ═══ run bar ══════════════════════════════════════════════════════════════ */
.runbar {{
  display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;
  font-family: var(--mono); font-size: 0.7rem; color: var(--faint);
  padding: 0.6rem 0.9rem; border: 1px solid var(--line); border-radius: var(--r);
  background: var(--surface); box-shadow: var(--lift); letter-spacing: 0.03em;
}}
.runbar b {{ color: var(--text); font-weight: 600; }}
.runbar .sep {{ color: var(--line-2); }}
.live {{ color: var(--accent); display: inline-flex; align-items: center; gap: 0.35rem; }}
.live::before {{
  content: ''; width: 5px; height: 5px; border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 0 3px rgba(15,118,110,0.16); animation: breathe 1.4s ease-in-out infinite;
}}

/* ═══ timeline ═════════════════════════════════════════════════════════════ */
.timeline {{
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
  box-shadow: var(--lift); padding: 0.5rem 0.35rem;
}}
.tl-row {{
  display: grid; grid-template-columns: 12px 104px 54px 1fr; align-items: center;
  gap: 0.6rem; padding: 0.38rem 0.75rem; border-radius: var(--r-sm);
  transition: background var(--ease);
}}
.tl-row:hover {{ background: var(--surface-2); }}
.tl-row .dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--line-2); }}
.tl-row .nm {{ font-weight: 500; color: var(--dim); font-size: 0.8rem; }}
.tl-row .at {{ font-family: var(--mono); font-size: 0.7rem; color: var(--faint); text-align: right; }}
.tl-row .msg {{ font-size: 0.75rem; color: var(--ghost); }}
.tl-row.done .dot {{ background: var(--good); }}
.tl-row.done .nm {{ color: var(--text); }}
.tl-row.active .dot {{
  background: var(--accent); box-shadow: 0 0 0 3px rgba(15,118,110,0.16);
  animation: breathe 1.4s ease-in-out infinite;
}}
.tl-row.active .nm {{ color: var(--accent); }}
.tl-row.failed .dot {{ background: var(--bad); }}
.tl-row.pending {{ opacity: 0.45; }}

/* ═══ event log ════════════════════════════════════════════════════════════ */
.log {{
  background: var(--surface-2); border: 1px solid var(--line); border-radius: var(--r);
  padding: 0.75rem 0.95rem; max-height: 300px; overflow-y: auto;
  font-family: var(--mono); font-size: 0.7rem; line-height: 1.95;
}}
.log .t {{ color: var(--ghost); margin-right: 0.6rem; }}
.log .a {{ color: var(--accent); margin-right: 0.6rem; }}
.log .m {{ color: var(--faint); }}
.log .ok {{ color: var(--good); }}
.log .err {{ color: var(--bad); }}

/* ═══ metric strip ═════════════════════════════════════════════════════════ */
.metrics {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(104px, 1fr)); gap: 1px;
  background: var(--line); border: 1px solid var(--line);
  border-radius: var(--r); overflow: hidden; margin: 1.1rem 0 0.3rem;
  box-shadow: var(--lift);
}}
.metric {{ background: var(--surface); padding: 0.75rem 0.9rem; }}
.metric .k {{
  font-family: var(--mono); font-size: 0.59rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--ghost);
}}
.metric .v {{
  display: block; font-size: 1.35rem; font-weight: 700; color: var(--text);
  margin-top: 0.25rem; letter-spacing: -0.025em; line-height: 1.1;
}}
.metric.good .v {{ color: var(--good); }}
.metric.warn .v {{ color: var(--warn); }}
.metric.bad .v {{ color: var(--bad); }}

/* ═══ verdict banner ═══════════════════════════════════════════════════════ */
.verdict {{
  display: flex; align-items: center; gap: 0.85rem;
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
  box-shadow: var(--lift); padding: 0.85rem 1.05rem;
}}
.verdict .ic {{ font-size: 0.9rem; flex: none; }}
.verdict b {{ display: block; font-size: 0.87rem; font-weight: 650; }}
.verdict .sub {{
  display: block; font-family: var(--mono); font-size: 0.7rem;
  color: var(--ghost); margin-top: 0.2rem; letter-spacing: 0.03em;
}}
.verdict.good {{ border-color: #bfe3cc; background: #f3faf5; }}
.verdict.good .ic, .verdict.good b {{ color: var(--good); }}
.verdict.warn {{ border-color: #ecd5ad; background: #fdf8f0; }}
.verdict.warn .ic, .verdict.warn b {{ color: var(--warn); }}
.verdict.bad {{ border-color: #f0c2cc; background: #fdf4f6; }}
.verdict.bad .ic, .verdict.bad b {{ color: var(--bad); }}

/* ═══ tables ═══════════════════════════════════════════════════════════════ */
.tbl {{
  width: 100%; border-collapse: collapse; font-size: 0.78rem;
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--r); overflow: hidden; box-shadow: var(--lift);
}}
.tbl th {{
  text-align: left; font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--faint); font-weight: 500;
  padding: 0.6rem 0.8rem; background: var(--surface-2); border-bottom: 1px solid var(--line);
}}
.tbl td {{ padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--line); color: var(--dim); }}
.tbl tr:last-child td {{ border-bottom: none; }}
.tbl tbody tr {{ transition: background var(--ease); }}
.tbl tbody tr:hover {{ background: var(--surface-2); }}
.tbl b {{ color: var(--text); font-weight: 650; }}
.tbl code, .mono {{
  font-family: var(--mono); font-size: 0.71rem; color: var(--accent);
  background: var(--accent-soft); padding: 0.1rem 0.4rem; border-radius: 4px;
}}
.diverged {{
  font-family: var(--mono); font-size: 0.59rem; color: var(--warn);
  background: #fbf0e2; padding: 0.1rem 0.38rem; border-radius: 4px;
  margin-left: 0.45rem; letter-spacing: 0.06em;
}}
.unscored {{ color: var(--ghost) !important; font-style: italic; }}
.v-good {{ color: var(--good) !important; font-weight: 650; }}
.v-warn {{ color: var(--warn) !important; font-weight: 650; }}
.v-bad {{ color: var(--bad) !important; font-weight: 650; }}

/* ═══ sources ══════════════════════════════════════════════════════════════ */
.src {{
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
  box-shadow: var(--lift); padding: 0.8rem 1rem; margin-bottom: 0.45rem;
  transition: border-color var(--ease);
}}
.src:hover {{ border-color: var(--accent-dim); }}
.src .hd {{ display: flex; align-items: baseline; gap: 0.6rem; }}
.src .n {{
  font-family: var(--mono); font-size: 0.65rem; color: var(--accent); flex: none;
  background: var(--accent-soft); padding: 0.12rem 0.42rem; border-radius: 4px;
}}
.src a {{ color: var(--text); text-decoration: none; font-weight: 600; font-size: 0.85rem; }}
.src a:hover {{ color: var(--accent); }}
.src .u {{
  display: block; font-family: var(--mono); font-size: 0.65rem; color: var(--ghost);
  margin: 0.3rem 0 0.35rem; word-break: break-all;
}}
.src .s {{ font-size: 0.76rem; color: var(--faint); line-height: 1.6; }}

/* ═══ key/value ════════════════════════════════════════════════════════════ */
.boxed {{
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
  box-shadow: var(--lift); padding: 0.35rem 0.95rem;
}}
.kv {{ display: flex; gap: 0.9rem; padding: 0.48rem 0; border-bottom: 1px solid var(--line); }}
.kv:last-child {{ border-bottom: none; }}
.kv .k {{
  flex: none; width: 128px; font-family: var(--mono); font-size: 0.68rem;
  color: var(--ghost); letter-spacing: 0.04em;
}}
.kv .v {{ flex: 1; min-width: 0; color: var(--dim); font-size: 0.79rem; }}

/* ═══ plan tasks ═══════════════════════════════════════════════════════════ */
.task {{
  display: grid; grid-template-columns: 38px 1fr; gap: 0.75rem;
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
  box-shadow: var(--lift); padding: 0.75rem 0.95rem; margin-bottom: 0.4rem;
}}
.task .id {{
  font-family: var(--mono); font-size: 0.66rem; color: var(--accent);
  background: var(--accent-soft); border-radius: 4px; padding: 0.12rem 0;
  text-align: center; height: fit-content;
}}
.task .w {{ color: var(--dim); font-size: 0.83rem; line-height: 1.55; }}
.task .m {{ font-family: var(--mono); font-size: 0.66rem; color: var(--ghost); }}

/* ═══ streamlit widgets ════════════════════════════════════════════════════ */
[data-testid="stChatInput"] {{
  background: var(--surface) !important;
  border: 1px solid var(--line-2) !important;
  border-radius: var(--r) !important;
  box-shadow: 0 4px 20px rgba(28,26,23,0.08) !important;
}}
[data-testid="stChatInput"]:focus-within {{
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(15,118,110,0.1) !important;
}}
[data-testid="stChatInput"] textarea {{
  color: var(--text) !important; font-size: 0.9rem !important; font-family: var(--sans) !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--ghost) !important; }}
[data-testid="stBottomBlockContainer"] {{
  background: linear-gradient(180deg, transparent, var(--bg) 30%) !important;
  padding-bottom: 1.2rem; max-width: 1040px; margin: 0 auto;
}}

.stButton > button, .stDownloadButton > button {{
  background: var(--surface) !important;
  border: 1px solid var(--line-2) !important;
  color: var(--dim) !important;
  border-radius: var(--r-sm) !important;
  font-size: 0.77rem !important; font-weight: 600 !important;
  padding: 0.4rem 0.85rem !important; min-height: 0 !important;
  box-shadow: var(--lift) !important;
  transition: all var(--ease) !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--accent) !important; color: var(--accent) !important;
  background: var(--accent-soft) !important;
}}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
  outline: 2px solid var(--accent) !important; outline-offset: 1px !important;
}}
button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: var(--accent) !important; border-color: var(--accent) !important;
  color: #ffffff !important;
}}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
  background: #0b5f58 !important; color: #ffffff !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.1rem; border-bottom: 1px solid var(--line); margin-bottom: 0.6rem;
}}
.stTabs [data-baseweb="tab"] {{
  font-family: var(--mono); font-size: 0.69rem !important; letter-spacing: 0.11em;
  text-transform: uppercase; color: var(--ghost); padding: 0.55rem 0.9rem;
  transition: color var(--ease);
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--dim); }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent) !important; height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

[data-testid="stExpander"] details {{
  background: var(--surface) !important; border: 1px solid var(--line) !important;
  border-radius: var(--r) !important; box-shadow: var(--lift);
}}
[data-testid="stExpander"] summary {{ font-size: 0.79rem; color: var(--dim); }}
[data-testid="stCaptionContainer"] p {{
  font-size: 0.68rem !important; color: var(--ghost) !important;
  font-family: var(--mono); letter-spacing: 0.04em;
}}
/* anchored on the track's own testid: `.stProgress > div > div` also matched the label's
   markdown wrapper and clamped the status text to 3px, so it overlapped the block below */
[data-testid="stProgressBarTrack"] {{
  background: var(--surface-3) !important; height: 3px !important; border-radius: 999px;
}}
[data-testid="stProgressBarTrack"] > div {{ background: var(--accent) !important; }}
[data-testid="stProgress"] [data-testid="stMarkdownContainer"] p {{
  font-family: var(--mono); font-size: 0.7rem !important; color: var(--faint) !important;
  letter-spacing: 0.04em; line-height: 1.7;
}}
[data-testid="stAlert"] {{
  border-radius: var(--r) !important; font-size: 0.8rem;
  background: var(--surface) !important; border: 1px solid var(--line) !important;
  box-shadow: var(--lift);
}}
[data-testid="stJson"] {{
  background: var(--surface-2) !important; border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
}}

/* charts sit on their own ground, which reads as a hole unless it is filled */
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {{
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
  padding: 0.65rem 0.45rem 0.25rem; box-shadow: var(--lift);
}}

::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--line-2); border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--faint); }}

@media (max-width: 820px) {{
  .stats, .agents {{ grid-template-columns: repeat(2, 1fr); }}
  .hero {{ padding: 2rem 0 1rem; }}
  .stMain .block-container {{ padding: 0 1rem 8rem; }}
}}
</style>
"""


def inject_css() -> None:
    """Safe to call on every rerun — Streamlit replaces the block rather than stacking it."""
    st.markdown(_CSS, unsafe_allow_html=True)


# the acronym expanded once, here, so every place that shows it spells it the same way
FULL_NAME = "autonomous multi-agent research & intelligence system"


def wordmark(extra: str = "", *, full_form: bool = False) -> str:
    """The product name as one piece of markup, so the hero and sidebar cannot drift."""
    mark = (
        f'<span class="wordmark {extra}"><span class="dotmark"></span>'
        f"AMA<span class='ris'>RIS</span></span>"
    )
    if not full_form:
        return mark
    # escaped because this goes straight into unsafe_allow_html — a bare & is not valid markup
    full = html.escape(FULL_NAME)
    return f'<span class="wm-group">{mark}<span class="wm-full">{full}</span></span>'


def badge_class(value: float, floor: float) -> str:
    """good / warn / bad for a score. The good floor is the caller's, never hard-coded here."""
    if value >= floor:
        return "good"
    return "warn" if value >= WARN_FLOOR else "bad"
