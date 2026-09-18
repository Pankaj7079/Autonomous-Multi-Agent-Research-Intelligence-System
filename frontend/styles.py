"""Design tokens and CSS injection. Every colour is defined here once — see docs/DESIGN.md."""

from __future__ import annotations

import html

import streamlit as st

# palette rationale: warm paper/ink, 4.5:1 contrast ratio (sidebar ghost was 2.20:1)
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
    "faint": "#655e56",
    "ghost": "#726b62",
    "accent": "#0f766e",
    "accent-2": "#9a5b2d",
    "accent-dim": "#7fc4bd",
    "accent-soft": "#e6f2f0",
    "good": "#15803d",
    "warn": "#b45309",
    "bad": "#be123c",
}

# bad-score color: floor comes from settings so UI never miscolors a revision
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

/* flow-root, so a markdown block contains its children's margins instead of letting them
   collapse out of it. without this the wrapper measured ~27px whatever was inside, and every
   heading with a hint under it was drawn over by the chips, table or caption that followed. */
[data-testid="stMarkdown"] {{ display: flow-root; }}

/* the base size is set on .stApp alone and everything inherits it. listing bare span/div/p
   here instead set the size ON each element, which beats inheritance — that is what rendered
   the hero mark as a 96px "AMA" with a 14px "RIS" stuck to it. never reset a bare tag. */
/* every number in this ui is data, so digits must not jitter between frames */
.mono, .metric .v, .tl-row .at, .runbar, .tbl, .kv .v, .turn-meta, .vchip {{
  font-variant-numeric: tabular-nums;
}}

/* ═══ sidebar ══════════════════════════════════════════════════════════════ */
/* the sidebar is part of the console, not a drawer: it carries provider health and the whole
   thread. these pin its width and override streamlit's own collapsed state, whatever it sets —
   our toggle below is what collapses it */
section[data-testid="stSidebar"] {{
  background: var(--bg-2);
  border-right: 1px solid var(--line);
  width: 268px !important;
  min-width: 268px !important;
  max-width: 268px !important;
  transform: none !important;
  visibility: visible !important;
  margin-left: 0 !important;
}}
/* streamlit's own collapse control is still hidden: its reopen arrow lives inside the app
   header this sheet removes, so using it would collapse the sidebar with no way back. the
   pair of buttons below replaces it and is ours end to end. */
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarNavCollapseButton"] {{ display: none !important; }}

/* hide rides in the header row beside the wordmark, show floats over the page once the
   sidebar is gone — both are ordinary buttons keyed with st-key-, so neither depends on
   streamlit internals. absolutely positioned it sat above the mark and read as an orphan. */
.st-key-sb_hide {{
  display: flex !important; justify-content: flex-end; width: auto !important;
}}
.st-key-sb_show {{
  position: fixed !important; top: 0.9rem; left: 0.9rem; width: auto !important; z-index: 1000;
}}
.st-key-sb_hide button, .st-key-sb_show button {{
  min-height: 0 !important; width: 26px; height: 26px;
  padding: 0 !important; line-height: 1 !important;
  display: flex !important; align-items: center; justify-content: center;
  font-size: 0.62rem !important; color: var(--faint) !important;
  background: var(--surface) !important; border: 1px solid var(--line-2) !important;
  border-radius: 50% !important; box-shadow: var(--lift) !important;
}}
.st-key-sb_hide button:hover, .st-key-sb_show button:hover {{
  color: var(--accent) !important; border-color: var(--accent-dim) !important;
}}
/* streamlit 1.63 has no .block-container in the sidebar, so the padding rule that used to
   live here selected nothing — stSidebarUserContent is the real one, and it needs the top
   and bottom only or the side padding stacks on the wrapper above it and clips the table */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
  padding-top: 1rem !important; padding-bottom: 2rem !important;
}}
/* the sidebar header is 60px of empty chrome holding a logo spacer and the collapse button
   this sheet already hides — 60px of nothing above the wordmark */
[data-testid="stSidebarHeader"] {{ display: none !important; }}
/* the sidebar scrolls on its own, and the platform scrollbar took 20px out of a 235px
   column — squeezing every chip and shifting the layout the moment content overflowed.
   a stable gutter reserves the space once, so nothing moves when it appears. */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
  scrollbar-width: thin;
  scrollbar-color: var(--line-2) transparent;
  scrollbar-gutter: stable;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar {{ width: 6px; }}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {{
  background: var(--line-2); border-radius: 3px;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.45rem; }}
/* the main pane's button padding wrapped "export .docx" onto two lines in a half-width
   sidebar column. the toggle is excluded — it is a fixed 26px circle with its own type size */
section[data-testid="stSidebar"]
  [data-testid="stElementContainer"]:not(.st-key-sb_hide) .stButton > button,
section[data-testid="stSidebar"] .stDownloadButton > button {{
  font-size: 0.72rem !important; padding: 0.35rem 0.55rem !important;
}}
/* the turn list is a list: centred labels put every score in a different place down the
   column. the ellipsis is CSS rather than a character budget — a 20-character question still
   wrapped the button to two lines once the prefix was counted */
[class*="st-key-thr_"] button [data-testid="stMarkdownContainer"] {{
  width: 100%; text-align: left !important;
}}
[class*="st-key-thr_"] button p {{
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  font-weight: 500 !important; font-variant-numeric: tabular-nums;
}}

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
  font-family: var(--mono); font-size: 0.64rem; font-weight: 500; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--ghost); max-width: 240px; line-height: 1.5;
}}
/* the group itself stays unbounded — it must size to the big mark, not to the caption.
   only the caption gets a width cap, so it wraps to two short lines instead of stretching
   the whole flex column wider than the mark (and, upstream, wider than the viewport) */
.hero .wm-group {{ align-items: center; gap: 0.55rem; }}
.hero .wm-full {{
  font-family: var(--sans); font-size: 0.74rem; font-weight: 500; letter-spacing: 0.1em;
  color: var(--faint); text-align: center; line-height: 1.6; max-width: min(90vw, 460px);
}}

/* ═══ sidebar surfaces ═════════════════════════════════════════════════════
   the sidebar is rendered as a product surface, not a console: soft cards, sans labels at a
   comfortable size, and monospace kept for data values only. an earlier pass answered
   "dense and technical" with small monospace everywhere and hairline rules, which reads as a
   terminal rather than as something a senior developer designed. */

section[data-testid="stSidebar"] .card {{
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--r-lg); box-shadow: var(--lift);
}}

/* section headings: sentence case in the body sans, no rule and no letter-spacing. the
   all-caps mono heading with a hairline running off it is the console look being removed */
section[data-testid="stSidebar"] .lbl {{
  font-family: var(--sans); font-size: 0.82rem; font-weight: 600; color: var(--text);
  text-transform: none; letter-spacing: 0; margin: 1.5rem 0 0.5rem;
}}
section[data-testid="stSidebar"] .lbl::after {{ display: none; }}
section[data-testid="stSidebar"] .lbl .n {{
  font-family: var(--sans); font-size: 0.7rem; font-weight: 600;
  color: var(--accent); background: var(--accent-soft); border: 0;
  border-radius: 99px; padding: 0.1rem 0.45rem; letter-spacing: 0;
}}
section[data-testid="stSidebar"] .lbl-hint {{
  font-family: var(--sans); font-size: 0.72rem; color: var(--faint);
  letter-spacing: 0; line-height: 1.5; margin: 0 0 0.6rem;
}}

/* every sidebar panel is an expander: the summary row carries enough to skip opening it, so
   the column is a short list of headers until something is actually wanted */
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
  border: 1px solid var(--line); border-radius: var(--r-lg);
  background: var(--surface); box-shadow: var(--lift); overflow: hidden;
  margin: 0.45rem 0 0.5rem;
}}
/* one line for the whole column, not one per panel: four captions stacked between four cards
   was more words than the panels they described */
.sb-tagline {{
  font-size: 0.74rem; color: var(--faint); line-height: 1.5; margin: 0 0 0.7rem 0.1rem;
}}
/* the key a visitor supplies, and what happens without one */
.key-on, .key-off {{
  font-size: 0.72rem; line-height: 1.5; margin: 0.5rem 0 0;
}}
.key-on {{ color: var(--accent); font-weight: 500; }}
.key-off {{ color: var(--ghost); }}
section[data-testid="stSidebar"] [data-testid="stExpander"] details {{
  border: 0 !important; background: transparent !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
  padding: 0.6rem 0.75rem !important; background: transparent !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
  background: var(--surface-2) !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
  font-family: var(--sans) !important; font-size: 0.8rem !important;
  font-weight: 600 !important; color: var(--text) !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {{
  fill: var(--faint) !important; color: var(--faint) !important;
}}
/* an open panel is separated from its header by a rule rather than by the header changing
   colour, which on a white card read as two different cards stacked */
section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{
  padding: 0.75rem 0.75rem 1.1rem !important; border-top: 1px solid var(--line);
}}

/* the agents, one row each. the decision line is the column that shows autonomy — a list of
   duties would read as a pipeline, which is the opposite of what this system is */
.ag-list {{ display: flex; flex-direction: column; gap: 0.6rem; }}
.ag-row {{ padding-left: 0.5rem; border-left: 2px solid var(--line); }}
.ag-row.core {{ border-left-color: var(--accent); }}
.ag-hd {{ display: flex; align-items: baseline; gap: 0.35rem; flex-wrap: wrap; }}
.ag-hd .tag {{
  font-family: var(--mono); font-size: 0.62rem; font-weight: 600; color: var(--accent);
  background: var(--accent-soft); border-radius: 4px; padding: 0.05rem 0.28rem;
}}
.ag-hd .nm {{ font-size: 0.78rem; font-weight: 600; color: var(--text); }}
.ag-hd .role {{ font-size: 0.7rem; color: var(--ghost); }}
.ag-row .dec {{ font-size: 0.72rem; color: var(--faint); line-height: 1.45; margin-top: 0.1rem; }}

/* the system card — an icon, a title and a count badge per block, then the items */
.sysbox {{ padding: 0; border: 0 !important; box-shadow: none !important; margin: 0; }}
.sys-hd {{
  display: flex; align-items: center; gap: 0.45rem;
  font-family: var(--sans); font-size: 0.78rem; font-weight: 600; color: var(--text);
  margin-bottom: 0.6rem;
}}
.sys-hd .ic {{ width: 15px; height: 15px; color: var(--accent); flex: none; }}
.sys-hd .t {{ flex: 1; }}
.sys-hd .pill {{
  font-family: var(--mono); font-size: 0.68rem; font-weight: 600;
  border-radius: 99px; padding: 0.12rem 0.5rem;
}}
.sys-hd .pill.ok {{ color: var(--accent); background: var(--accent-soft); }}
.sys-hd .pill.warn {{ color: var(--warn); background: #fdf3e5; }}
.sys-sep {{ height: 1px; background: var(--line); margin: 0.9rem 0; }}

.sys-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem; }}
/* the chain is an order, not a set, so it stays one wrapping line read left to right */
.sys-grid.chain {{ display: flex; flex-wrap: wrap; gap: 0.32rem; }}
.sys-item {{
  display: inline-flex; align-items: center; gap: 0.4rem; min-width: 0;
  font-family: var(--sans); font-size: 0.75rem; color: var(--dim); white-space: nowrap;
  background: var(--surface-2); border-radius: 99px; padding: 0.22rem 0.5rem;
}}
/* only the dot carries the status colour — seven coloured words read as an alarm */
.sys-item .dot {{
  width: 6px; height: 6px; border-radius: 50%; background: var(--good); flex: none;
}}
.sys-item.hot .dot {{ background: var(--warn); }}
.sys-item.off {{ color: var(--ghost); background: var(--bg); }}
.sys-item.off .dot {{ background: var(--line-2); }}

/* an attached file is a pill like the ones above it, just carrying the accent because it is
   the one thing in this column the user put there themselves */
section[data-testid="stSidebar"] .chip {{
  display: inline-flex; width: auto; font-family: var(--sans); font-size: 0.74rem;
  letter-spacing: 0; border-radius: 99px; padding: 0.26rem 0.65rem;
  margin: 0 0.3rem 0.35rem 0;
}}

/* the strip is built for the main pane's width; here it is two soft tiles per row */
section[data-testid="stSidebar"] .metrics.mini {{
  grid-template-columns: 1fr 1fr; gap: 0.45rem; margin: 0.35rem 0 0.2rem;
}}
section[data-testid="stSidebar"] .metrics.mini .metric {{ padding: 0.55rem 0.65rem; }}
section[data-testid="stSidebar"] .metrics.mini .k {{
  font-family: var(--sans); font-size: 0.68rem; font-weight: 500;
  letter-spacing: 0; text-transform: none; color: var(--faint);
}}
section[data-testid="stSidebar"] .metrics.mini .v {{ font-size: 1.05rem; margin-top: 0.15rem; }}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
  margin: 0 0 0.5rem !important;
}}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
  margin: 0 !important; line-height: 1.45;
}}

/* the empty thread reads as a waiting slot rather than as missing content */
.sb-empty {{
  font-family: var(--sans); font-size: 0.75rem; color: var(--ghost);
  background: var(--bg); border: 1px dashed var(--line-2); border-radius: var(--r);
  padding: 0.7rem 0.8rem; text-align: center; margin: 0.2rem 0 0.3rem;
}}

/* the four budgets side by side, shown only before a conversation starts */
.dtbl {{ font-size: 0.75rem; color: var(--dim); overflow: hidden; margin: 0; }}
.dtbl .r {{
  display: grid; grid-template-columns: 1fr 2.6rem 2.2rem 3rem;
  gap: 0.2rem; padding: 0.45rem 0.6rem; align-items: baseline; white-space: nowrap;
}}
.dtbl .r + .r {{ border-top: 1px solid var(--line); }}
.dtbl .h {{
  font-family: var(--sans); font-weight: 600; color: var(--faint); font-size: 0.7rem;
  background: var(--surface-2);
}}
.dtbl .r span:not(:first-child) {{
  text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums;
}}
.dtbl .d {{ font-family: var(--sans); color: var(--text); font-weight: 600; }}

/* ═══ sidebar header ═══════════════════════════════════════════════════════ */
.sb-mark {{ font-size: 1.45rem; }}
.sb-sub {{
  display: flex; align-items: center; gap: 0.5rem;
  font-family: var(--sans); font-size: 0.78rem; color: var(--faint);
  letter-spacing: 0; text-transform: none;
  margin: 0.35rem 0 1.15rem; padding-bottom: 0.9rem; border-bottom: 1px solid var(--line);
}}
.sb-mode {{
  display: inline-flex; align-items: center; gap: 0.3rem;
  font-size: 0.7rem; font-weight: 600; color: var(--accent);
  background: var(--accent-soft); border-radius: 99px; padding: 0.12rem 0.5rem;
  text-transform: capitalize;
}}
.sb-mode::before {{
  content: ''; width: 5px; height: 5px; border-radius: 50%; background: var(--accent);
}}
.hist-m {{
  font-family: var(--mono); font-size: 0.68rem; color: var(--ghost);
  margin: -0.1rem 0 0.55rem 0.75rem; letter-spacing: 0;
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
/* sans, sentence case: as 0.63rem mono at 0.16em tracking this was the console look, and the
   one line on the page that says the system is actually running */
.eyebrow {{
  display: inline-flex; align-items: center; gap: 0.55rem;
  font-family: var(--sans); font-size: 0.78rem; font-weight: 500; letter-spacing: 0;
  text-transform: none; color: var(--dim);
  border: 1px solid var(--line); background: var(--surface);
  border-radius: 999px; padding: 0.38rem 0.95rem; margin-top: 1.5rem;
  box-shadow: var(--lift);
}}
.eyebrow .bar {{ width: 1px; height: 11px; background: var(--line-2); }}
.pip {{
  width: 5px; height: 5px; border-radius: 50%; background: var(--good);
  box-shadow: 0 0 0 3px rgba(21,128,61,0.13);
}}

/* ═══ stat strip ═══════════════════════════════════════════════════════════ */
/* not on the landing page any more; the rules stay with the component that draws it */
.stats {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1px;
  background: var(--line); border: 1px solid var(--line);
  border-radius: var(--r-lg); overflow: hidden; box-shadow: var(--lift);
  margin: 1.8rem 0 0.5rem;
}}
.stat {{ background: var(--surface); padding: 1.05rem 1.15rem; }}
.stat .n {{
  font-size: 1.85rem; font-weight: 800; letter-spacing: -0.035em;
  line-height: 1.05; color: var(--accent); font-variant-numeric: tabular-nums;
}}
.stat .l {{
  font-family: var(--sans); font-size: 0.82rem; font-weight: 600; letter-spacing: -0.01em;
  text-transform: none; color: var(--text); margin-top: 0.2rem;
}}

/* ═══ try one ══════════════════════════════════════════════════════════════ */
.try-hd {{
  font-size: 1.05rem; font-weight: 700; color: var(--text);
  letter-spacing: -0.018em; margin: 2rem 0 0.25rem;
}}
.try-sub {{
  font-size: 0.83rem; color: var(--faint); line-height: 1.6;
  margin-bottom: 0.9rem; max-width: 64ch;
}}
.depth-note {{
  font-size: 0.74rem; color: var(--ghost); line-height: 1.5; margin: 0.55rem 0 0;
}}
.lvl-hd {{
  font-size: 0.92rem; font-weight: 700; color: var(--text);
  letter-spacing: -0.015em; margin: 1.9rem 0 0.45rem;
}}
.depth-note .lvl {{
  display: inline-block; font-weight: 600; color: var(--accent);
  background: var(--accent-soft); border-radius: 99px; padding: 0.06rem 0.5rem;
  margin-right: 0.45rem;
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
  border-radius: var(--r-sm); font-family: var(--mono); font-size: 0.64rem;
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
  font-size: 0.64rem; padding: 0.1rem 0.45rem; border-radius: 999px;
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
/* an indexed file is the one chip that is a thing the user put there, so it carries a
   heavier border and weight than the status chips around it */
.chip.accent {{
  color: var(--accent); background: var(--accent-soft); border-color: var(--accent-dim);
  font-weight: 500; box-shadow: var(--lift);
}}

/* the composer's attach, mic and send controls shipped at 40-60% opacity, which on this
   ground made the paperclip almost invisible — the one control that opens the file feature */
[data-testid="stBottom"] button[aria-label="Upload files"],
[data-testid="stChatInputMicButton"] {{
  color: var(--dim) !important; opacity: 1 !important;
}}
[data-testid="stBottom"] button[aria-label="Upload files"]:hover,
[data-testid="stChatInputMicButton"]:hover {{
  color: var(--accent) !important; background: var(--accent-soft) !important;
}}
[data-testid="stChatInputSubmitButton"] {{
  color: var(--accent) !important; background: var(--accent-soft) !important;
  border: 1px solid var(--accent-dim) !important; opacity: 1 !important;
}}
[data-testid="stChatInputSubmitButton"]:disabled {{
  color: var(--faint) !important; background: var(--surface-3) !important;
  border-color: var(--line) !important;
}}
/* a queued upload sat on the page ground with no border, so it read as blank space next to
   the placeholder rather than as a file waiting to be sent */
[data-testid="stFileChip"] {{
  background: var(--surface) !important; border: 1px solid var(--accent-dim) !important;
  border-radius: var(--r-sm) !important; box-shadow: var(--lift) !important;
}}
[data-testid="stFileChipName"] {{ font-weight: 500 !important; color: var(--text) !important; }}
[data-testid="stFileChipImagePreview"] {{
  border: 1px solid var(--line-2) !important; border-radius: 4px !important;
  background: var(--surface-2) !important;
}}
[data-testid="stFileChipDeleteBtn"] {{ color: var(--faint) !important; }}
[data-testid="stFileChipDeleteBtn"]:hover {{ color: var(--bad) !important; }}

/* the composer is the primary control on the page and read as a disabled input box */
[data-testid="stChatInput"] {{
  border: 1px solid var(--line-2) !important; box-shadow: var(--lift) !important;
}}
[data-testid="stChatInput"]:focus-within {{
  border-color: var(--accent) !important;
}}

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
  font-family: var(--mono); font-size: 0.64rem; letter-spacing: 0.17em;
  text-transform: uppercase; color: var(--ghost);
}}
.turn-head .rule {{ flex: 1; height: 1px; background: var(--line); }}
.depth-chip {{
  font-family: var(--mono); font-size: 0.64rem; letter-spacing: 0.13em;
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

/* the answer says when its own evidence is thin, rather than letting a weak answer look
   exactly like a strong one. amber, not red: an unverified claim is a caution, not a failure */
.caveat {{
  display: flex; align-items: flex-start; gap: 0.6rem;
  background: #fdf8f0; border: 1px solid #ecd5ad; border-radius: var(--r);
  padding: 0.7rem 0.9rem; margin-top: 1.1rem;
  font-size: 0.82rem; line-height: 1.55; color: var(--warn);
}}
.caveat .ic {{ flex: none; line-height: 1.5; }}

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
  display: grid; place-items: center; font-family: var(--mono); font-size: 0.64rem;
  font-weight: 600; letter-spacing: 0.05em;
  border: 1px solid var(--line); background: var(--surface-2); color: var(--ghost);
  transition: all var(--ease);
}}
.flow .node .nm {{
  display: block; font-size: 0.64rem; color: var(--ghost); margin-top: 0.4rem;
  font-family: var(--mono); letter-spacing: 0.03em;
}}
.flow .node .hits {{
  display: block; font-family: var(--mono); font-size: 0.64rem; color: var(--warn);
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

/* ═══ the question, while its run is in flight ═════════════════════════════ */
.asking {{
  display: flex; align-items: baseline; gap: 0.7rem; flex-wrap: wrap;
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r);
  box-shadow: var(--lift); padding: 0.85rem 1.05rem; margin-bottom: 0.55rem;
}}
.asking .who {{
  font-family: var(--mono); font-size: 0.64rem; letter-spacing: 0.17em;
  text-transform: uppercase; color: var(--accent); flex: none;
}}
.asking .q {{
  font-size: 1.02rem; font-weight: 650; color: var(--text);
  letter-spacing: -0.015em; line-height: 1.4;
}}

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
  font-family: var(--mono); font-size: 0.63rem; letter-spacing: 0.1em;
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
  text-align: left; font-family: var(--mono); font-size: 0.64rem; letter-spacing: 0.12em;
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
  font-family: var(--mono); font-size: 0.63rem; color: var(--warn);
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
  .agents {{ grid-template-columns: repeat(2, 1fr); }}
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


_COLLAPSED_CSS = """
<style>
/* injected only while the sidebar is hidden — width, not display, so the widgets inside
   keep their state and the show button can bring them straight back */
section[data-testid="stSidebar"] {
  width: 0 !important; min-width: 0 !important; max-width: 0 !important;
  border-right: none !important; overflow: hidden !important;
}
</style>
"""


def collapsed_css() -> str:
    """The extra sheet that hides the sidebar. Injected per-rerun while it is collapsed."""
    return _COLLAPSED_CSS
