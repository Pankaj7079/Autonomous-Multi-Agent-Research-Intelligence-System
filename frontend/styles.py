"""Design tokens and CSS injection. Every colour is defined here once — see docs/DESIGN.md."""

from __future__ import annotations

import streamlit as st

# the token block from docs/DESIGN.md — never hard-code a colour at a call site
TOKENS = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "surface-2": "#1c2128",
    "border": "#30363d",
    "text": "#e6edf3",
    "text-dim": "#8b949e",
    "accent": "#58a6ff",
    "good": "#3fb950",
    "warn": "#d29922",
    "bad": "#f85149",
}

# below this a score is bad; the good floor comes from settings so the UI can never
# call something green that the critic sent back for revision
WARN_FLOOR = 0.55

# technical values read as data in a mono face and as prose in a UI face — the split is
# what makes a developer tool look like one rather than like a consumer app
MONO = "'JetBrains Mono','SFMono-Regular',Consolas,'Liberation Mono',monospace"

_VARS = "\n".join(f"  --{name}: {value};" for name, value in TOKENS.items())

_CSS = f"""
<style>
:root {{
{_VARS}
  --mono: {MONO};
}}

.stApp {{ background: var(--bg); color: var(--text); }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; max-width: 1180px; }}
section[data-testid="stSidebar"] {{
  background: var(--surface); border-right: 1px solid var(--border);
}}

/* ── masthead ─────────────────────────────────────────────── */
.amaris-mast {{
  display: flex; align-items: baseline; gap: 0.75rem; flex-wrap: wrap;
  border-bottom: 1px solid var(--border); padding-bottom: 0.7rem; margin-bottom: 1rem;
}}
.amaris-title {{
  font-size: 1.45rem; font-weight: 800; letter-spacing: -0.03em; margin: 0;
  background: linear-gradient(92deg, var(--text) 30%, var(--accent));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.amaris-sub {{ color: var(--text-dim); font-size: 0.82rem; margin: 0; }}
.amaris-pill {{
  display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px;
  border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text-dim); font-family: var(--mono);
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.04em;
}}
.amaris-pill.on {{ color: var(--good); border-color: var(--good); }}
.amaris-pill.off {{ color: var(--text-dim); opacity: 0.6; }}

/* ── run header ───────────────────────────────────────────── */
.amaris-runbar {{
  display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap;
  font-family: var(--mono); font-size: 0.78rem; color: var(--text-dim);
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.5rem 0.8rem; margin-bottom: 0.6rem;
}}
.amaris-runbar b {{ color: var(--text); font-weight: 600; }}
.amaris-runbar .sep {{ opacity: 0.35; }}

/* ── pipeline timeline ────────────────────────────────────── */
.amaris-timeline {{
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface); overflow: hidden; margin-bottom: 0.8rem;
}}
.amaris-step {{
  display: grid; grid-template-columns: 16px 108px 62px 1fr;
  gap: 0.6rem; align-items: center;
  padding: 0.45rem 0.8rem; border-bottom: 1px solid var(--border);
  font-size: 0.82rem;
}}
.amaris-step:last-child {{ border-bottom: none; }}
.amaris-step .dot {{
  width: 8px; height: 8px; border-radius: 50%; background: var(--border);
  justify-self: center;
}}
.amaris-step .name {{ font-weight: 600; font-family: var(--mono); font-size: 0.78rem; }}
.amaris-step .at {{ font-family: var(--mono); font-size: 0.72rem; color: var(--text-dim); }}
.amaris-step .note {{ color: var(--text-dim); font-size: 0.78rem; }}

.amaris-step.done .dot {{ background: var(--good); }}
.amaris-step.done .name {{ color: var(--text); }}
.amaris-step.active {{ background: var(--surface-2); }}
.amaris-step.active .dot {{
  background: var(--accent); box-shadow: 0 0 0 3px rgba(88,166,255,0.18);
}}
.amaris-step.active .name {{ color: var(--accent); }}
.amaris-step.failed .dot {{ background: var(--bad); }}
.amaris-step.failed .name {{ color: var(--bad); }}
.amaris-step.pending {{ opacity: 0.4; }}

/* ── event log ────────────────────────────────────────────── */
.amaris-log {{
  font-family: var(--mono); font-size: 0.74rem; line-height: 1.7;
  background: #0a0d12; border: 1px solid var(--border); border-radius: 8px;
  padding: 0.6rem 0.8rem; max-height: 260px; overflow-y: auto;
}}
.amaris-log .t {{ color: var(--text-dim); }}
.amaris-log .a {{ color: var(--accent); }}
.amaris-log .m {{ color: var(--text); }}
.amaris-log .ok {{ color: var(--good); }}
.amaris-log .err {{ color: var(--bad); }}

/* ── stats ────────────────────────────────────────────────── */
.amaris-stats {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.2rem 0 1rem 0; }}
.amaris-stat {{
  flex: 1 1 110px; padding: 0.5rem 0.7rem; border-radius: 6px;
  background: var(--surface-2); border: 1px solid var(--border);
}}
.amaris-stat .label {{
  color: var(--text-dim); font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.08em;
}}
.amaris-stat .value {{ font-size: 1.2rem; font-weight: 700; font-family: var(--mono); }}

/* ── score badges ─────────────────────────────────────────── */
.amaris-scores {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.4rem 0 1rem 0; }}
.amaris-badge {{
  flex: 1 1 140px; padding: 0.6rem 0.8rem; border-radius: 6px;
  background: var(--surface); border: 1px solid var(--border); border-left-width: 3px;
}}
.amaris-badge .label {{
  color: var(--text-dim); font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.08em;
}}
.amaris-badge .value {{ font-size: 1.4rem; font-weight: 700; font-family: var(--mono); }}
.amaris-badge.good {{ border-left-color: var(--good); }}
.amaris-badge.good .value {{ color: var(--good); }}
.amaris-badge.warn {{ border-left-color: var(--warn); }}
.amaris-badge.warn .value {{ color: var(--warn); }}
.amaris-badge.bad {{ border-left-color: var(--bad); }}
.amaris-badge.bad .value {{ color: var(--bad); }}

/* ── tables ───────────────────────────────────────────────── */
.amaris-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
.amaris-table th {{
  text-align: left; padding: 0.4rem 0.6rem; color: var(--text-dim);
  border-bottom: 1px solid var(--border); font-weight: 600;
  text-transform: uppercase; font-size: 0.64rem; letter-spacing: 0.08em;
}}
.amaris-table td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }}
.amaris-table tr:last-child td {{ border-bottom: none; }}
.amaris-table code, .amaris-mono {{
  font-family: var(--mono); background: var(--surface-2);
  padding: 0.08rem 0.32rem; border-radius: 3px; font-size: 0.74rem;
}}
.amaris-diverged {{
  color: var(--warn); font-family: var(--mono); font-size: 0.66rem;
  letter-spacing: 0.05em; font-weight: 700;
}}
.amaris-unscored {{ color: var(--text-dim); font-style: italic; }}
.amaris-good {{ color: var(--good); font-weight: 700; font-family: var(--mono); }}
.amaris-warn {{ color: var(--warn); font-weight: 700; font-family: var(--mono); }}
.amaris-bad {{ color: var(--bad); font-weight: 700; font-family: var(--mono); }}

/* ── citations ────────────────────────────────────────────── */
.amaris-cite {{ padding: 0.3rem 0; border-bottom: 1px solid var(--border); font-size: 0.82rem; }}
.amaris-cite:last-child {{ border-bottom: none; }}
.amaris-cite .num {{ color: var(--accent); font-family: var(--mono); margin-right: 0.4rem; }}
.amaris-cite a {{ color: var(--text); text-decoration: none; }}
.amaris-cite a:hover {{ color: var(--accent); }}

/* ── sidebar config ───────────────────────────────────────── */
.amaris-kv {{
  display: flex; justify-content: space-between; gap: 0.6rem;
  font-family: var(--mono); font-size: 0.72rem;
  padding: 0.22rem 0; border-bottom: 1px dashed var(--border);
}}
.amaris-kv:last-child {{ border-bottom: none; }}
.amaris-kv .k {{ color: var(--text-dim); }}
.amaris-kv .v {{ color: var(--text); text-align: right; word-break: break-all; }}
.amaris-section {{
  font-size: 0.64rem; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--text-dim); margin: 0.9rem 0 0.3rem 0; font-weight: 700;
}}
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
