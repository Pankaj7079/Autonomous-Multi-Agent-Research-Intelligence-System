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

_VARS = "\n".join(f"  --{name}: {value};" for name, value in TOKENS.items())

_CSS = f"""
<style>
:root {{
{_VARS}
}}

.stApp {{ background: var(--bg); color: var(--text); }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2.5rem; max-width: 1050px; }}

.amaris-title {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; }}
.amaris-sub {{ color: var(--text-dim); font-size: 0.9rem; margin: 0.2rem 0 1.2rem 0; }}

.amaris-pill {{
  display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
  border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text-dim); font-size: 0.75rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
}}

.amaris-track {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.4rem 0 1rem 0; }}
.amaris-card {{
  flex: 1 1 130px; padding: 0.6rem 0.75rem; border-radius: 8px;
  border: 1px solid var(--border); background: var(--surface);
}}
.amaris-card .name {{ font-weight: 600; font-size: 0.88rem; }}
.amaris-card .note {{ color: var(--text-dim); font-size: 0.75rem; margin-top: 0.2rem; }}
.amaris-card.active {{ border-color: var(--accent); background: var(--surface-2); }}
.amaris-card.active .name {{ color: var(--accent); }}
.amaris-card.done {{ border-color: var(--good); }}
.amaris-card.failed {{ border-color: var(--bad); }}
.amaris-card.pending {{ opacity: 0.45; border-style: dashed; }}

.amaris-scores {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.4rem 0 1rem 0; }}
.amaris-badge {{
  flex: 1 1 140px; padding: 0.7rem 0.8rem; border-radius: 8px;
  background: var(--surface); border: 1px solid var(--border); border-left-width: 4px;
}}
.amaris-badge .label {{
  color: var(--text-dim); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.06em;
}}
.amaris-badge .value {{ font-size: 1.5rem; font-weight: 700; line-height: 1.3; }}
.amaris-badge.good {{ border-left-color: var(--good); }}
.amaris-badge.good .value {{ color: var(--good); }}
.amaris-badge.warn {{ border-left-color: var(--warn); }}
.amaris-badge.warn .value {{ color: var(--warn); }}
.amaris-badge.bad {{ border-left-color: var(--bad); }}
.amaris-badge.bad .value {{ color: var(--bad); }}

.amaris-cite {{ padding: 0.35rem 0; border-bottom: 1px solid var(--border); font-size: 0.85rem; }}
.amaris-cite:last-child {{ border-bottom: none; }}
.amaris-cite .num {{ color: var(--accent); font-weight: 700; margin-right: 0.4rem; }}
.amaris-cite a {{ color: var(--text); text-decoration: none; }}
.amaris-cite a:hover {{ color: var(--accent); }}

.amaris-report {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.2rem 1.5rem;
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
