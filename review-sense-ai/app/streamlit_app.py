import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import io
import base64
from pathlib import Path
import plotly.express as px
from collections import Counter


def style_plotly_fig(fig):
    """Make Plotly charts blend into the premium dark-glass UI."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E9ECF8"),
        colorway=["#FF3D8D", "#FF8A3D", "#FFC83D", "#35D07F", "#20C7B5", "#9B6CFF"],
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(font=dict(color="#DCE1F2")),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.10)",
        zeroline=False,
        tickfont=dict(color="#AEB6D4"),
        title_font=dict(color="#C9D0E8"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.10)",
        zeroline=False,
        tickfont=dict(color="#AEB6D4"),
        title_font=dict(color="#C9D0E8"),
    )
    return fig


def render_sentiment_card_html(cls, label, count, pct):
    """Return a single-line HTML sentiment card so Streamlit never treats it as a code block."""
    colors = {
        "strong-neg": ("#FF3D72", "rgba(255,61,114,.30)"),
        "neg": ("#FF963D", "rgba(255,150,61,.28)"),
        "mixed": ("#FFC83D", "rgba(255,200,61,.25)"),
        "pos": ("#35D07F", "rgba(53,208,127,.28)"),
        "strong-pos": ("#20C7B5", "rgba(32,199,181,.28)"),
    }
    color, glow = colors.get(cls, ("#FF5A9D", "rgba(255,90,157,.25)"))
    pct = max(0.0, min(100.0, float(pct)))
    # No newlines/leading indentation: Streamlit's Markdown parser cannot turn this into a code block.
    return (
        f'<div class="rs-sentiment-card {cls}" style="--pct:{pct:.2f}%;--sentiment:{color};--sentiment-glow:{glow};">'
        f'<div class="label">{label}</div>'
        f'<div class="visual-row">'
        f'<div class="ring" style="background:conic-gradient({color} {pct:.2f}%,rgba(255,255,255,.08) 0);box-shadow:0 0 16px {glow};">'
        f'<div class="ring-inner"><div class="ring-pct">{pct:.1f}%</div><div class="ring-small">share</div></div></div>'
        f'<div class="count-wrap"><div class="count" style="color:{color};text-shadow:0 0 13px {glow};">{int(count):,}</div><div class="count-label">reviews</div></div>'
        f'</div>'
        f'<div class="track"><div class="fill" style="width:{pct:.2f}%;background:{color};box-shadow:0 0 12px {glow};"></div></div>'
        f'<div class="foot"><span>Dataset share</span><strong>{pct:.1f}%</strong></div>'
        f'</div>'
    )


# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model_calibrated.joblib"
REPORTS_PATH = PROJECT_ROOT / "outputs" / "reports"


# PAGE CONFIG
st.set_page_config(page_title="ReviewSense", page_icon="💬", layout="wide")

# ============================================================
# REVIEW SENSE — PREMIUM PINK GLASS UI
# ============================================================
# Put your downloaded image here:
#     review-sense-ai/static/background.jpg
#
# .streamlit/config.toml must contain:
#     [server]
#     enableStaticServing = true
# ============================================================
# ============================================================
# REVIEW SENSE — PREMIUM PINK ANALYTICS UI
# ============================================================

background_candidates = [
    PROJECT_ROOT / "static" / "background.jpg",
    PROJECT_ROOT / "static" / "background.jpeg",
    PROJECT_ROOT / "static" / "background.png",
    PROJECT_ROOT / "app" / "static" / "background.jpg",
    PROJECT_ROOT / "app" / "static" / "background.png",
]

background_uri = ""
for _bg_path in background_candidates:
    if _bg_path.exists():
        _mime = "image/png" if _bg_path.suffix.lower() == ".png" else "image/jpeg"
        _encoded = base64.b64encode(_bg_path.read_bytes()).decode("utf-8")
        background_uri = f"data:{_mime};base64,{_encoded}"
        break

background_layer = (
    f'url("{background_uri}")'
    if background_uri
    else "linear-gradient(135deg,#fff4f8,#f9eaf2)"
)

st.markdown(f"""
<style>

html, body, .stApp {{
    font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif !important;
}}

.stApp {{
    background: transparent !important;
    color: #F7F8FF !important;
}}

[data-testid="stAppViewContainer"] {{
    background:
        linear-gradient(135deg, rgba(6, 9, 27, .48), rgba(11, 8, 36, .30)),
        {background_layer}
        center center / cover fixed no-repeat !important;
}}

[data-testid="stMain"],
.main,
.main .block-container {{
    background: transparent !important;
}}

.main .block-container {{
    max-width: 1500px !important;
    padding: 2rem 2.5rem 5rem !important;
}}

[data-testid="stHeader"] {{
    height: 58px !important;
    background: rgba(7, 11, 31, .48) !important;
    border-bottom: 1px solid rgba(255,255,255,.10) !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}}

/* SIDEBAR */
section[data-testid="stSidebar"] {{ width: 280px !important; }}

[data-testid="stSidebar"] {{
    background: rgba(7, 11, 31, .88) !important;
    border-right: 1px solid rgba(255,255,255,.10) !important;
    box-shadow: 10px 0 40px rgba(0,0,0,.25);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
}}

[data-testid="stSidebar"] > div:first-child {{ background: transparent !important; }}

.rs-side-brand {{
    display:flex; align-items:center; gap:11px; padding:8px 7px 2px;
}}

.rs-side-logo {{
    width:42px; height:42px; border-radius:13px;
    display:flex; align-items:center; justify-content:center;
    color:white;
    background:linear-gradient(135deg,#FF3D8D,#C92CFF);
    box-shadow:0 8px 25px rgba(255,61,141,.30);
    font-size:20px; font-weight:800;
}}

.rs-side-brand-name {{
    font-size:1.18rem; line-height:1; font-weight:850;
    letter-spacing:-.6px; color:#FFFFFF;
}}

.rs-side-brand-name span {{ color:#FF3D8D; }}

.rs-side-caption {{
    margin:6px 0 27px 53px; font-size:.70rem;
    font-weight:600; color:#AEB6D4;
}}

.rs-nav-label {{
    margin:0 7px 10px; font-size:.67rem;
    text-transform:uppercase; letter-spacing:.12em;
    font-weight:800; color:#FF5A9D;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    min-height:40px; margin:3px 0 !important; padding:9px 10px !important;
    border-radius:12px !important; font-size:.88rem !important;
    font-weight:650 !important; color:#D7DCEF !important;
    transition:all .18s ease;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background:rgba(255,61,141,.10) !important;
    color:#FFFFFF !important;
}}

.rs-side-info {{
    margin-top:20px; padding:16px 15px;
    border:1px solid rgba(255,255,255,.11);
    border-radius:17px;
    background:linear-gradient(145deg,rgba(255,255,255,.10),rgba(255,255,255,.035));
    box-shadow:0 10px 30px rgba(0,0,0,.16);
    backdrop-filter:blur(18px);
}}

.rs-side-info-title {{ font-size:.83rem; font-weight:800; color:#F5F7FF; }}

.rs-side-info-text {{
    margin-top:7px; font-size:.72rem; line-height:1.55; color:#AEB6D4;
}}

/* HERO */
.rs-hero {{ padding:.8rem .2rem 1rem; }}

.rs-hero-row {{ display:flex; align-items:center; gap:14px; }}

.rs-hero-icon {{
    width:58px; height:58px; border-radius:18px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#FF3D8D,#C94EFF);
    color:white; font-size:28px;
    box-shadow:0 12px 32px rgba(255,61,141,.32), inset 0 1px 0 rgba(255,255,255,.25);
}}

.rs-hero-title {{
    font-size:2.7rem; line-height:1; font-weight:850;
    letter-spacing:-2px; color:#FFFFFF;
    text-shadow:0 4px 25px rgba(0,0,0,.20);
}}

.rs-hero-title span {{ color:#FF3D8D; }}

.rs-hero-subtitle {{
    margin:10px 0 0 72px; font-size:.94rem; color:#C1C8DF;
}}

.rs-mode-row {{
    margin:16px 0 0 72px; display:flex; align-items:center; gap:10px;
}}

.rs-mode-pill {{
    padding:7px 11px; border-radius:999px;
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.12);
    color:#C9D0E8; font-size:.73rem; font-weight:700;
}}

/* TYPOGRAPHY */
.stApp p, .stApp span, .stApp label {{ color:#C4CADF; }}

.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
    color:#FFFFFF !important; font-weight:800 !important;
}}

[data-testid="stCaptionContainer"] {{ color:#AEB6D4 !important; }}

hr {{ border-color:rgba(255,255,255,.10) !important; }}

/* SECTIONS */
.rs-section {{
    margin:1.35rem 0 .75rem; display:flex; align-items:center; gap:10px;
}}

.rs-section-icon {{
    width:34px; height:34px; display:flex; align-items:center; justify-content:center;
    border-radius:10px; background:rgba(255,255,255,.11);
    border:1px solid rgba(255,255,255,.12); color:#FF5A9D; font-size:17px;
    box-shadow:0 6px 18px rgba(0,0,0,.12);
}}

.rs-section-title {{
    font-size:1.3rem; font-weight:820; letter-spacing:-.5px; color:#FFFFFF;
}}

.rs-section-sub {{ margin-left:2px; font-size:.73rem; color:#AEB6D4; }}

/* GLASS CARDS */
.rs-card,
.rs-review-panel,
.rs-review-panel.positive,
.rs-review-panel.negative,
.rs-theme-panel,
.rs-quick,
.rs-kpi-card,
.rs-chart-head {{
    background:linear-gradient(145deg,rgba(255,255,255,.13),rgba(255,255,255,.045)) !important;
    border:1px solid rgba(255,255,255,.15) !important;
    box-shadow:0 14px 38px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.10) !important;
    backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
}}

/* KPI */
.rs-kpi-card {{
    position:relative; min-height:108px; display:flex; align-items:center;
    gap:14px; padding:17px 18px; border-radius:20px; overflow:hidden;
}}

.rs-kpi-icon {{
    flex:0 0 47px; width:47px; height:47px;
    display:flex; align-items:center; justify-content:center;
    border-radius:50%; font-size:21px; color:white;
    box-shadow:0 7px 22px var(--kpi-shadow); background:var(--kpi-color);
}}

.rs-kpi-icon.pink {{ --kpi-color:#FF3D8D; --kpi-shadow:rgba(255,61,141,.32); }}
.rs-kpi-icon.green {{ --kpi-color:#35D07F; --kpi-shadow:rgba(53,208,127,.30); }}
.rs-kpi-icon.red {{ --kpi-color:#FF5570; --kpi-shadow:rgba(255,85,112,.30); }}
.rs-kpi-icon.purple {{ --kpi-color:#9B6CFF; --kpi-shadow:rgba(155,108,255,.30); }}

.rs-kpi-label {{ font-size:.70rem; font-weight:750; color:#B8C0D8; }}

.rs-kpi-value {{
    margin-top:4px; font-size:1.7rem; line-height:1;
    font-weight:850; letter-spacing:-.7px; color:#FF5A9D;
}}

.rs-kpi-value.green {{ color:#35D07F; }}
.rs-kpi-value.red {{ color:#FF5570; }}
.rs-kpi-value.purple {{ color:#9B6CFF; }}

.rs-kpi-sub {{ margin-top:6px; font-size:.66rem; color:#929BB7; }}

/* BUSINESS INSIGHTS — PREMIUM REVIEW VISUALS */
.rs-bi-review-card {{
    position:relative;
    min-height:158px;
    padding:18px 18px 16px;
    border-radius:20px;
    overflow:hidden;
    background:linear-gradient(145deg,rgba(255,255,255,.105),rgba(255,255,255,.035));
    border:1px solid rgba(255,255,255,.13);
    box-shadow:0 14px 38px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.08);
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
    transition:transform .22s ease, border-color .22s ease, box-shadow .22s ease;
}}
.rs-bi-review-card:hover {{
    transform:translateY(-4px);
    border-color:var(--bi-color);
    box-shadow:0 18px 42px rgba(0,0,0,.28), 0 0 24px var(--bi-glow);
}}
.rs-bi-review-card::before {{
    content:"";
    position:absolute;
    left:0; top:0; bottom:0;
    width:4px;
    background:var(--bi-color);
    box-shadow:0 0 18px var(--bi-color);
}}
.rs-bi-review-card.positive {{ --bi-color:#35D07F; --bi-glow:rgba(53,208,127,.24); }}
.rs-bi-review-card.negative {{ --bi-color:#FF5570; --bi-glow:rgba(255,85,112,.24); }}
.rs-bi-review-card.mixed {{ --bi-color:#FFC21C; --bi-glow:rgba(255,194,28,.22); }}
.rs-bi-review-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
.rs-bi-review-title {{ color:#F7F9FF; font-size:.88rem; font-weight:850; }}
.rs-bi-review-icon {{
    width:34px; height:34px; border-radius:11px;
    display:flex; align-items:center; justify-content:center;
    color:var(--bi-color); background:rgba(255,255,255,.07);
    border:1px solid color-mix(in srgb,var(--bi-color) 55%, transparent);
    box-shadow:0 0 18px var(--bi-glow);
}}
.rs-bi-review-stat {{ margin-top:10px; display:flex; align-items:baseline; gap:9px; }}
.rs-bi-review-count {{ color:var(--bi-color); font-size:1.9rem; line-height:1; font-weight:950; text-shadow:0 0 16px var(--bi-glow); }}
.rs-bi-review-pct {{ color:#F1F4FB; font-size:1.05rem; font-weight:850; }}
.rs-bi-review-sub {{ margin-top:5px; color:#929BB7; font-size:.68rem; }}
.rs-bi-review-track {{ margin-top:12px; height:9px; border-radius:999px; background:rgba(255,255,255,.09); overflow:hidden; }}
.rs-bi-review-fill {{ height:100%; width:var(--bi-pct); border-radius:999px; background:linear-gradient(90deg,var(--bi-color),color-mix(in srgb,var(--bi-color) 62%,white)); box-shadow:0 0 14px var(--bi-glow); }}
.rs-bi-review-foot {{ margin-top:10px; display:flex; justify-content:space-between; color:#8E97B1; font-size:.64rem; font-weight:700; }}
.rs-bi-theme-card {{
    border-radius:18px; padding:16px 17px;
    background:linear-gradient(145deg,rgba(255,255,255,.09),rgba(255,255,255,.035));
    border:1px solid rgba(255,255,255,.12);
    box-shadow:0 12px 30px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.07);
}}
.rs-bi-theme-card.positive {{ border-color:rgba(53,208,127,.28); }}
.rs-bi-theme-card.negative {{ border-color:rgba(255,85,112,.28); }}
.rs-bi-theme-label {{ font-size:.68rem; color:#929BB7; margin-bottom:10px; }}
.rs-bi-theme-chips {{ display:flex; flex-wrap:wrap; gap:7px; }}
.rs-bi-theme-chip {{
    display:inline-flex; padding:6px 10px; border-radius:999px;
    font-size:.68rem; font-weight:750; color:#E9ECF7;
    background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10);
}}
.rs-bi-theme-card.positive .rs-bi-theme-chip {{ color:#72E5A5; border-color:rgba(53,208,127,.25); background:rgba(53,208,127,.08); }}
.rs-bi-theme-card.negative .rs-bi-theme-chip {{ color:#FF8296; border-color:rgba(255,85,112,.25); background:rgba(255,85,112,.08); }}
.rs-bi-quote {{
    margin-top:12px; padding:13px 14px; border-radius:15px;
    background:rgba(0,0,0,.16); border:1px solid rgba(255,255,255,.09);
    color:#DDE2F0; font-size:.72rem; line-height:1.55;
}}
.rs-bi-quote strong {{ color:var(--bi-color); }}

/* INFO BAR */
.rs-info-bar {{
    margin:15px 0 13px; padding:11px 15px;
    border:1px solid rgba(255,255,255,.13); border-radius:14px;
    background:rgba(255,255,255,.07); color:#D8DDF0;
    font-size:.73rem; box-shadow:0 8px 24px rgba(0,0,0,.12);
    backdrop-filter:blur(15px);
}}

/* ============================================================
   SENTIMENT BREAKDOWN — VISUAL, COMPACT & PREMIUM
   ============================================================ */
.rs-sentiment-card {{
    --sentiment:#FF5A9D;
    --sentiment-soft:rgba(255,90,157,.12);
    --sentiment-glow:rgba(255,90,157,.25);
    position:relative; overflow:hidden; min-height:205px;
    padding:17px 16px 15px; border-radius:19px;
    background:linear-gradient(145deg,rgba(255,255,255,.105),rgba(255,255,255,.035));
    border:1px solid rgba(255,255,255,.13);
    box-shadow:0 12px 30px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.06);
    backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
    transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;
}}
.rs-sentiment-card:hover {{
    transform:translateY(-4px);
    border-color:var(--sentiment);
    box-shadow:0 16px 36px rgba(0,0,0,.25),0 0 22px var(--sentiment-glow);
}}
.rs-sentiment-card::before {{
    content:""; position:absolute; left:0; right:0; top:0; height:4px;
    background:var(--sentiment); box-shadow:0 0 13px var(--sentiment-glow);
}}
.rs-sentiment-card .label {{
    min-height:36px; display:flex; align-items:center;
    font-size:.75rem; line-height:1.2; font-weight:800; color:#E7EAF5;
}}
.rs-sentiment-card .visual-row {{
    display:flex; align-items:center; gap:13px; margin-top:5px;
}}
.rs-sentiment-card .ring {{
    width:72px; height:72px; flex:0 0 72px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    background:conic-gradient(var(--sentiment) var(--pct),rgba(255,255,255,.08) 0);
    box-shadow:0 0 16px var(--sentiment-glow);
}}
.rs-sentiment-card .ring-inner {{
    width:54px; height:54px; border-radius:50%;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    background:#11152F; border:1px solid rgba(255,255,255,.08);
}}
.rs-sentiment-card .ring-pct {{
    font-size:.78rem; line-height:1; font-weight:900; color:#FFFFFF;
}}
.rs-sentiment-card .ring-small {{
    margin-top:3px; font-size:.48rem; font-weight:700; color:#8990AD; text-transform:uppercase;
}}
.rs-sentiment-card .count-wrap {{ display:flex; flex-direction:column; min-width:0; }}
.rs-sentiment-card .count {{
    margin:0; font-size:1.62rem; line-height:1; font-weight:950; color:var(--sentiment);
    text-shadow:0 0 13px var(--sentiment-glow);
}}
.rs-sentiment-card .count-label {{ margin-top:5px; font-size:.61rem; color:#929BB7; font-weight:650; }}
.rs-sentiment-card .track {{
    height:7px; margin-top:15px; border-radius:99px;
    background:rgba(255,255,255,.08); overflow:hidden;
}}
.rs-sentiment-card .fill {{
    height:100%; width:var(--pct); border-radius:99px;
    background:var(--sentiment);
    box-shadow:0 0 12px var(--sentiment-glow);
}}
.rs-sentiment-card .foot {{
    display:flex; justify-content:space-between; align-items:center;
    margin-top:7px; font-size:.58rem; color:#858DAA; font-weight:650;
}}
.rs-sentiment-card .foot strong {{ color:#E7EAF5; }}

.rs-sentiment-card.strong-neg {{ --sentiment:#FF3D72; --sentiment-glow:rgba(255,61,114,.30); }}
.rs-sentiment-card.neg {{ --sentiment:#FF963D; --sentiment-glow:rgba(255,150,61,.28); }}
.rs-sentiment-card.mixed {{ --sentiment:#FFC83D; --sentiment-glow:rgba(255,200,61,.25); }}
.rs-sentiment-card.pos {{ --sentiment:#35D07F; --sentiment-glow:rgba(53,208,127,.28); }}
.rs-sentiment-card.strong-pos {{ --sentiment:#20C7B5; --sentiment-glow:rgba(32,199,181,.28); }}

/* BUTTONS */
div.stButton > button {{
    border-radius:15px !important;
    border:1px solid rgba(255,255,255,.13) !important;
    background:rgba(255,255,255,.075) !important;
    color:#E8EBF7 !important; font-weight:700 !important;
    min-height:70px !important; box-shadow:0 8px 22px rgba(0,0,0,.14);
    white-space:pre-line !important; backdrop-filter:blur(14px);
    transition:all .18s ease !important;
}}

div.stButton > button:hover {{
    border-color:#FF4F99 !important; background:rgba(255,61,141,.12) !important;
    color:#FFFFFF !important; transform:translateY(-2px);
}}

/* CHARTS */
.rs-chart-shell {{ padding:5px 0 0; }}

.rs-chart-head {{
    display:flex; align-items:center; gap:8px; margin:0 0 5px 3px;
    padding:11px 13px; border-radius:15px 15px 0 0;
}}

.rs-chart-icon {{
    display:flex; align-items:center; justify-content:center;
    width:29px; height:29px; border-radius:9px;
    background:rgba(255,61,141,.12); border:1px solid rgba(255,61,141,.16);
    color:#FF5A9D; font-size:15px;
}}

.rs-chart-title {{ font-size:.84rem; font-weight:800; color:#EEF1FB; }}

div[data-testid="stPlotlyChart"],
.stPlotlyChart {{
    background:linear-gradient(145deg,rgba(13,18,42,.76),rgba(10,14,34,.62)) !important;
    border:1px solid rgba(255,255,255,.13) !important;
    border-radius:0 0 17px 17px !important;
    padding:7px 8px 2px !important;
    box-shadow:0 12px 32px rgba(0,0,0,.20) !important;
    overflow:hidden;
}}

.js-plotly-plot {{ background:transparent !important; }}

/* QUICK ANALYZE */
.rs-quick {{ margin-top:18px; padding:18px; border-radius:20px; }}
.rs-quick-title {{ font-size:1.1rem; font-weight:820; color:#FFFFFF; }}
.rs-quick-sub {{ margin-top:4px; font-size:.73rem; color:#AEB6D4; }}

textarea, input {{
    background:rgba(7,11,31,.52) !important; color:#F7F8FF !important;
    border:1px solid rgba(255,255,255,.14) !important; border-radius:13px !important;
}}

textarea::placeholder, input::placeholder {{ color:#8F98B8 !important; }}

[data-baseweb="select"] > div {{
    background:rgba(7,11,31,.52) !important; color:#FFFFFF !important;
    border-radius:13px !important; border:1px solid rgba(255,255,255,.14) !important;
}}

/* PREMIUM FILE UPLOAD */
.rs-upload-card {{
    position:relative;
    padding:18px 18px 14px;
    margin-top:2px;
    border-radius:20px;
    background:linear-gradient(145deg,rgba(255,255,255,.075),rgba(255,255,255,.025));
    border:1px solid rgba(255,255,255,.13);
    box-shadow:0 14px 34px rgba(0,0,0,.18);
}}

.rs-upload-head {{
    display:flex; align-items:center; gap:12px; margin-bottom:5px;
}}

.rs-upload-icon {{
    width:42px; height:42px; border-radius:13px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,rgba(255,61,141,.20),rgba(155,108,255,.20));
    border:1px solid rgba(255,255,255,.12);
    box-shadow:0 8px 20px rgba(255,61,141,.12);
    font-size:20px;
}}

.rs-upload-title {{ font-size:1rem; font-weight:850; color:#F5F7FF; }}
.rs-upload-sub {{ margin:2px 0 0 54px; font-size:.70rem; color:#9EA7C5; }}

.rs-upload-formats {{
    display:flex; gap:7px; flex-wrap:wrap; margin:12px 0 3px 54px;
}}

.rs-file-chip {{
    padding:4px 8px; border-radius:999px;
    font-size:.62rem; font-weight:800; color:#CBD2E8;
    background:rgba(255,255,255,.055);
    border:1px solid rgba(255,255,255,.10);
}}

[data-testid="stFileUploader"] {{
    background:transparent !important;
    border:none !important;
    padding:0 !important;
    margin:8px 0 0 !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    min-height:118px !important;
    padding:15px !important;
    border-radius:15px !important;
    background:rgba(7,11,31,.38) !important;
    border:1px dashed rgba(255,255,255,.22) !important;
    transition:all .18s ease !important;
}}

[data-testid="stFileUploaderDropzone"]:hover {{
    border-color:rgba(255,61,141,.55) !important;
    background:rgba(255,61,141,.055) !important;
    box-shadow:inset 0 0 0 1px rgba(255,61,141,.08), 0 8px 24px rgba(255,61,141,.08) !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] {{
    color:#AEB6D4 !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] span {{
    color:#DDE2F2 !important; font-weight:700 !important;
}}

[data-testid="stFileUploaderDropzone"] button {{
    border-radius:10px !important;
    border:1px solid rgba(255,255,255,.16) !important;
    background:linear-gradient(135deg,#FF3D8D,#C94EFF) !important;
    color:white !important;
    font-weight:800 !important;
}}

.rs-upload-success {{
    margin-top:10px; padding:9px 11px; border-radius:11px;
    background:rgba(53,208,127,.09);
    border:1px solid rgba(53,208,127,.18);
    color:#83E8AE; font-size:.72rem; font-weight:750;
}}

button[kind="primary"] {{
    background:linear-gradient(135deg,#FF3D8D,#C94EFF) !important;
    color:white !important; border:none !important;
    box-shadow:0 9px 26px rgba(255,61,141,.28) !important;
}}

/* BATCH SUMMARY */
.rs-batch-summary {{
    margin-top:12px; padding:14px 17px; border-radius:15px;
    background:rgba(53,208,127,.10); border:1px solid rgba(53,208,127,.22);
    color:#72E6A5; font-size:.86rem; font-weight:650;
}}

/* REVIEW PANELS */
.rs-review-panel {{ padding:17px; border-radius:18px; }}
.rs-review-panel.positive {{ border-top:4px solid #35D07F !important; }}
.rs-review-panel.negative {{ border-top:4px solid #FF5570 !important; }}

.rs-review-title {{
    display:flex; align-items:center; gap:8px; font-size:1.02rem;
    font-weight:820; color:#FFFFFF; margin-bottom:11px;
}}

.rs-review-title .badge {{
    padding:4px 8px; border-radius:999px; font-size:.68rem; font-weight:800;
}}

.rs-review-title .badge.green {{ color:#6EE7A0; background:rgba(53,208,127,.12); }}
.rs-review-title .badge.red {{ color:#FF8A9C; background:rgba(255,85,112,.12); }}

.rs-review-item {{
    padding:12px 13px; margin-top:8px; border-radius:12px;
    background:rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.09);
    color:#CDD3E6; font-size:.82rem; line-height:1.55;
}}

.rs-review-score {{
    margin-bottom:4px; font-size:.68rem; font-weight:800; color:#929BB7;
    text-transform:uppercase; letter-spacing:.04em;
}}

/* THEMES */
.rs-theme-panel {{ margin-top:14px; padding:15px 17px; border-radius:16px; }}

.rs-theme-title {{
    font-size:.76rem; font-weight:800; color:#AEB6D4;
    text-transform:uppercase; letter-spacing:.05em; margin-bottom:9px;
}}

.rs-theme-chip {{
    display:inline-block; margin:3px 5px 3px 0; padding:6px 9px;
    border-radius:999px; font-size:.72rem; font-weight:700;
}}

.rs-theme-chip.green {{
    color:#6EE7A0; background:rgba(53,208,127,.12);
    border:1px solid rgba(53,208,127,.16);
}}

.rs-theme-chip.red {{
    color:#FF8A9C; background:rgba(255,85,112,.12);
    border:1px solid rgba(255,85,112,.16);
}}

/* HIGHLIGHTS */
.hl {{
    display:inline-block; padding:2px 6px; border-radius:6px;
    font-weight:700; color:#FFFFFF !important;
}}
.hl-pos {{ background:#35B978; box-shadow:0 2px 8px rgba(53,185,120,.20); }}
.hl-neg {{ background:#E84B62; box-shadow:0 2px 8px rgba(232,75,98,.20); }}

/* ALERTS / TABLES */
[data-testid="stAlert"] {{
    border-radius:14px !important; background:rgba(255,255,255,.08) !important;
    border:1px solid rgba(255,255,255,.12) !important; color:#E7EAF5 !important;
}}

[data-testid="stDataFrame"] {{ border-radius:15px !important; overflow:hidden !important; }}

/* SCROLLBAR */
::-webkit-scrollbar {{ width:8px; }}
::-webkit-scrollbar-track {{ background:#070B1F; }}
::-webkit-scrollbar-thumb {{
    background:linear-gradient(#FF3D8D,#9B6CFF); border-radius:10px;
}}

</style>
""", unsafe_allow_html=True)


# LOAD MODEL + REPORTS
model = None

# Model (required)
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    st.error(
        "Model file not found in this deployment.\n\n"
        f"Expected: {MODEL_PATH}\n\n"
        "Fix:\n"
        "1) Commit and push: artifacts/best_model_calibrated.joblib\n"
        "2) Redeploy Streamlit Cloud\n"
        "(Training inside Streamlit Cloud is disabled to avoid crashes.)"
    )
    st.stop()

# Reports
misclass_path = REPORTS_PATH / "misclassified.csv"
compare_path  = REPORTS_PATH / "model_comparison.csv"

df_errors  = pd.read_csv(misclass_path) if misclass_path.exists() else pd.DataFrame()
df_compare = pd.read_csv(compare_path)  if compare_path.exists()  else pd.DataFrame()

if df_errors.empty:
    st.info(
        "Baseline report missing: outputs/reports/misclassified.csv\n"
        "Fix: Run locally: `python main.py` then commit/push outputs/reports/."
    )

if df_compare.empty:
    st.info(
        "Baseline report missing: outputs/reports/model_comparison.csv\n"
        "Fix: Run locally: `python main.py` then commit/push outputs/reports/."
    )



# HEADER + MODE
st.markdown("""
<div class="rs-hero">
    <div class="rs-hero-row">
        <div class="rs-hero-icon">💬</div>
        <div class="rs-hero-title">Welcome to Review<span>Sense</span></div>
    </div>
    <div class="rs-hero-subtitle">A customer-ready review intelligence dashboard</div>
</div>
""", unsafe_allow_html=True)

# Keep mode in session_state (safer)
st.session_state.mode = st.radio("🧠 Explanation Mode", ["Simple Language", "Technical"], horizontal=True)

def explain(simple, technical):
    return simple if st.session_state.get("mode", "Simple Language") == "Simple Language" else technical


# 5-level buckets
BUCKETS = [
    ("Strongly Negative 😡", 0.00, 0.25, "Customers are very unhappy"),
    ("Negative 🙁",          0.25, 0.45, "Some disappointment reported"),
    ("Mixed 😐",             0.45, 0.65, "Customers have mixed feelings"),
    ("Positive 🙂",          0.65, 0.85, "Generally satisfied customers"),
    ("Strongly Positive 😍", 0.85, 1.01, "Customers really love this"),
]
BUCKET_ORDER = [b[0] for b in BUCKETS]

def bucketize(p: float) -> str:
    for name, lo, hi, _ in BUCKETS:
        if lo <= p < hi:
            return name
    return "Mixed 😐"

def bucket_explainer(bucket: str) -> str:
    for name, _, _, desc in BUCKETS:
        if name == bucket:
            return desc
    return ""

# Baseline report safety + derived cols
if not df_errors.empty:
    # Ensure required columns exist if misclassified.csv exists
    REQUIRED_ERROR_COLS = {"proba_pos", "review_raw"}
    missing = REQUIRED_ERROR_COLS - set(df_errors.columns)
    if missing:
        st.error(
            f"misclassified.csv is missing required columns: {missing}. "
            "Please re-run training to regenerate reports."
        )
        st.stop()

    # Ensure confidence exists consistently (avoid confidence_margin drift)
    if "confidence" not in df_errors.columns:
        df_errors["confidence"] = np.round(np.abs(df_errors["proba_pos"] - 0.5) * 2, 4)

    # Ensure review_clean exists (used in keyword sections)
    if "review_clean" not in df_errors.columns:
        df_errors["review_clean"] = (
            df_errors["review_raw"]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Buckets
    df_errors["bucket"] = df_errors["proba_pos"].apply(bucketize)


# Global highlighting util
def highlight_words(text, words, color):
    if not words:
        return text
    for w in set(words):
        text = re.sub(
            rf"\b({re.escape(w)})\b",
            rf"<span style='background-color:{color}; padding:3px 4px; border-radius:6px;'>\1</span>",
            text,
            flags=re.IGNORECASE
        )
    return text


# Model extraction (CalibratedClassifierCV-safe)
def _get_fitted_pipeline_from_calibrated(calibrated_model):
    if hasattr(calibrated_model, "calibrated_classifiers_") and calibrated_model.calibrated_classifiers_:
        cc0 = calibrated_model.calibrated_classifiers_[0]
        if hasattr(cc0, "estimator"):
            return cc0.estimator
    if hasattr(calibrated_model, "estimator"):
        return calibrated_model.estimator
    return None

def get_top_words(calibrated_model, n=10):
    pipe = _get_fitted_pipeline_from_calibrated(calibrated_model)
    if pipe is None:
        return [], []

    tfidf = pipe.named_steps.get("tfidf", None)
    svm = pipe.named_steps.get("svm", None)

    if tfidf is None or svm is None or not hasattr(tfidf, "vocabulary_"):
        return [], []

    feature_names = tfidf.get_feature_names_out()
    coefs = svm.coef_[0]

    top_pos = feature_names[np.argsort(coefs)[-n:]][::-1]
    top_neg = feature_names[np.argsort(coefs)[:n]]
    return list(top_pos), list(top_neg)

TOP_POS_WORDS, TOP_NEG_WORDS = get_top_words(model, n=12)


# Review-level explainability (dual meaning)

def _get_fitted_tfidf_and_svm(calibrated_model):
    pipe = _get_fitted_pipeline_from_calibrated(calibrated_model)
    if pipe is None:
        return None, None
    tfidf = pipe.named_steps.get("tfidf", None)
    svm = pipe.named_steps.get("svm", None)
    if tfidf is None or svm is None or not hasattr(tfidf, "vocabulary_"):
        return None, None
    return tfidf, svm

def explain_review_terms(calibrated_model, text: str, top_k_each=6):
    """
    Returns (pos_terms, neg_terms) based on TF-IDF(feature in review) * linear weight.
    Model-faithful for TF-IDF + LinearSVC.
    """
    tfidf, svm = _get_fitted_tfidf_and_svm(calibrated_model)
    if tfidf is None or svm is None or not text.strip():
        return [], []

    X = tfidf.transform([text])
    row = X.tocoo()
    if row.nnz == 0:
        return [], []

    feature_names = tfidf.get_feature_names_out()
    coefs = svm.coef_[0]

    contrib = row.data * coefs[row.col]
    terms = feature_names[row.col]
    items = list(zip(terms, contrib))

    pos = sorted([it for it in items if it[1] > 0], key=lambda x: x[1], reverse=True)[:top_k_each]
    neg = sorted([it for it in items if it[1] < 0], key=lambda x: x[1])[:top_k_each]

    pos_terms = []
    for t, _ in pos:
        if t not in pos_terms:
            pos_terms.append(t)

    neg_terms = []
    for t, _ in neg:
        if t not in neg_terms:
            neg_terms.append(t)

    return pos_terms, neg_terms

def highlight_terms_both(text: str, pos_terms, neg_terms):
    """
    Highlights terms using CSS classes (.hl-pos / .hl-neg).
    Negative first so phrases like 'not good' stay red.
    """
    if not text:
        return ""

    # Escape HTML
    safe = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )

    pos_terms = sorted(set(pos_terms), key=len, reverse=True)
    neg_terms = sorted(set(neg_terms), key=len, reverse=True)

    def repl(span_text, cls):
        return f'<span class="hl {cls}">{span_text}</span>'

    # Negative first
    for term in neg_terms:
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(term)}(?!\w)")
        safe = pattern.sub(lambda m: repl(m.group(0), "hl-neg"), safe)

    for term in pos_terms:
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(term)}(?!\w)")
        safe = pattern.sub(lambda m: repl(m.group(0), "hl-pos"), safe)

    return safe


# Better explanations + noise filtering
GENERIC_TERMS = {
    "thing", "things", "lot", "well", "really", "very", "much", "early", "right", "made",
    "get", "got", "one", "also", "still", "even", "just", "like", "would", "could", "dont",
    "don't", "im", "i'm", "ive", "i've", "movie", "book"
}
def filter_terms(terms):
    cleaned = []
    for t in terms:
        tt = t.strip().lower()
        if len(tt) <= 2:
            continue
        if tt in GENERIC_TERMS:
            continue
        cleaned.append(t)
    return cleaned

BUSINESS_STOP = {
    "good","great","bad","love","like","best","better","worst","nice","ok","okay",
    "book","movie","read","story","series","song","cd","dvd",
    "one","really","very","much","also","still","even","just","would","could","make","made",
    "get","got","dont","don't","im","i'm","ive","i've","cant","can't","time","work",
    "buy","bought","use","used","using","product"
}

def business_filter_terms(terms):
    out = []
    for t in terms:
        tt = t.strip().lower()
        if len(tt) <= 2:
            continue
        if tt in BUSINESS_STOP:
            continue
        out.append(t)
    return out

def get_highlight_k(level: str):
    if level == "Low":
        return 4
    if level == "Medium":
        return 7
    return 12

def get_active_insights_df():
    """
    Prefer uploaded batch results (if present), else use baseline df_errors.
    Returns (df, source_label, text_col)
    """
    if st.session_state.get("batch_df") is not None and isinstance(st.session_state.batch_df, pd.DataFrame):
        out = st.session_state.batch_df
        text_col = str(out["_text_col_used"].iloc[0]) if "_text_col_used" in out.columns else None
        return out, "Uploaded File", text_col
    # If baseline missing, still return empty df safely
    return df_errors, "Baseline Sample", "review_raw"

def split_sentences(text: str):
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def sentence_scores(calibrated_model, text: str):
    tfidf, svm = _get_fitted_tfidf_and_svm(calibrated_model)
    if tfidf is None or svm is None or not text.strip():
        return []

    coefs = svm.coef_[0]
    results = []
    for sent in split_sentences(text):
        X = tfidf.transform([sent]).tocoo()
        if X.nnz == 0:
            continue
        contrib = X.data * coefs[X.col]
        net = float(contrib.sum())
        pos = float(contrib[contrib > 0].sum()) if np.any(contrib > 0) else 0.0
        neg = float(contrib[contrib < 0].sum()) if np.any(contrib < 0) else 0.0
        results.append({"sentence": sent, "pos": pos, "neg": neg, "net": net})
    return results

def extract_top_phrases_from_group(calibrated_model, texts, top_n=12):
    tfidf, svm = _get_fitted_tfidf_and_svm(calibrated_model)
    if tfidf is None or svm is None or len(texts) == 0:
        return [], []

    X = tfidf.transform(texts)
    coefs = svm.coef_[0]
    feature_names = tfidf.get_feature_names_out()

    weighted = X.multiply(coefs).sum(axis=0)
    weighted = np.asarray(weighted).ravel()

    present = np.asarray(X.sum(axis=0)).ravel() > 0
    weighted = weighted * present

    is_phrase = np.array([" " in f for f in feature_names])
    weighted_phrase = weighted * is_phrase

    if np.all(weighted_phrase == 0):
        weighted_phrase = weighted  # fallback to unigrams

    pos_idx = np.argsort(weighted_phrase)[-200:][::-1]
    neg_idx = np.argsort(weighted_phrase)[:200]

    pos = [feature_names[i] for i in pos_idx if weighted_phrase[i] > 0]
    neg = [feature_names[i] for i in neg_idx if weighted_phrase[i] < 0]

    pos = business_filter_terms(pos)
    neg = business_filter_terms(neg)

    return pos[:top_n], neg[:top_n]

def top_keywords(df, text_col, n=12):
    if df.empty or text_col not in df.columns:
        return []
    text = " ".join(df[text_col].astype(str).tolist())
    tokens = [t for t in text.split() if len(t) > 2]
    return Counter(tokens).most_common(n)

def summarize_bucket_insights(subset_df, k=8):
    kws = top_keywords(subset_df, "review_clean", n=40)
    kws = [(w, c) for (w, c) in kws if w.lower() not in GENERIC_TERMS and len(w) > 2]
    top = [w for w, _ in kws[:k]]
    if not top:
        return "Most reviews in this category share similar wording and tone."
    return f"Most reviews here mention: {', '.join(top[:5])}" + (f" (and also {', '.join(top[5:])})." if len(top) > 5 else ".")


# Helpers: upload anything + auto column guessing
def load_any_table(uploaded_file) -> pd.DataFrame:
    """
    Robust loader for CSV, TSV and XLSX files.
    """

    name = uploaded_file.name.lower()

    # =========================
    # CSV
    # =========================
    if name.endswith(".csv"):

        raw_data = uploaded_file.getvalue()

        # Remove NULL bytes
        raw_data = raw_data.replace(b"\x00", b"")

        encodings = [
            "utf-8",
            "utf-8-sig",
            "latin-1",
            "cp1252"
        ]

        for encoding in encodings:

            try:
                text_data = raw_data.decode(encoding)

                # Python engine is more tolerant
                df = pd.read_csv(
                    io.StringIO(text_data),
                    engine="python",
                    on_bad_lines="skip"
                )

                if not df.empty:
                    return df

            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        # Last attempt: auto-detect delimiter
        for encoding in encodings:

            try:
                text_data = raw_data.decode(encoding)

                df = pd.read_csv(
                    io.StringIO(text_data),
                    sep=None,
                    engine="python",
                    on_bad_lines="skip"
                )

                if not df.empty:
                    return df

            except Exception:
                continue

        raise ValueError(
            "Could not parse the CSV file. "
            "Please check the CSV format."
        )

    # =========================
    # TSV
    # =========================
    if name.endswith(".tsv"):

        raw_data = uploaded_file.getvalue()
        raw_data = raw_data.replace(b"\x00", b"")

        try:
            text_data = raw_data.decode("utf-8")
        except UnicodeDecodeError:
            text_data = raw_data.decode("latin-1")

        return pd.read_csv(
            io.StringIO(text_data),
            sep="\t",
            engine="python",
            on_bad_lines="skip"
        )

    # =========================
    # XLSX
    # =========================
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    raise ValueError(
        "Unsupported file format. "
        "Please upload CSV, TSV, or XLSX."
    )
def reason_counts(df):
    counts = {
        "Needs manual review (uncertain)": int(df["_uncertain"].sum()),
        "Mixed sentiment cases": int(df["_reasons"].apply(lambda xs: "Mixed feelings" in xs).sum()),
        "Negation cases": int(df["_reasons"].apply(lambda xs: "Negation (not/never)" in xs).sum()),
        "Very short reviews": int(df["_reasons"].apply(lambda xs: "Too short / low context" in xs).sum()),
        "Emphasis / tone cases": int(df["_reasons"].apply(lambda xs: "Emphasis / tone (caps/punct)" in xs).sum()),
    }
    return counts

def guess_text_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    preferred = ["review", "text", "content", "comment", "message", "feedback", "body", "sentence"]
    cols_lower = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p in cols_lower:
            return cols_lower[p]

    best_col, best_score = None, -1
    for c in df.columns:
        numeric_ratio = pd.to_numeric(df[c], errors="coerce").notna().mean()
        if numeric_ratio > 0.85:
            continue
        s = df[c].astype(str)
        score = float(s.str.len().mean())
        if score > best_score:
            best_score = score
            best_col = c
    return best_col

def guess_rating_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    preferred = ["rating", "stars", "star", "score", "overall", "review_score"]
    cols_lower = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p in cols_lower:
            return cols_lower[p]
    return None

def guess_product_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    preferred = ["product", "product_name", "item", "title", "asin", "sku", "category", "brand"]
    cols_lower = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p in cols_lower:
            return cols_lower[p]
    return None

def guess_date_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    preferred = ["date", "timestamp", "time", "review_date", "created_at"]
    cols_lower = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p in cols_lower:
            return cols_lower[p]
    return None

def safe_textcol_from_batch(df):
    if df is None or df.empty:
        return None
    if "_text_col_used" in df.columns:
        return str(df["_text_col_used"].iloc[0])
    for c in ["review", "text", "content", "comment", "message", "feedback", "body", "sentence"]:
        if c in df.columns:
            return c
    return df.columns[0]


# Charts (compact + interactive)
def _bucket_colors():
    return {
        "Strongly Negative 😡": "#E83E7B",
        "Negative 🙁": "#FF963F",
        "Mixed 😐": "#FFC21C",
        "Positive 🙂": "#43B978",
        "Strongly Positive 😍": "#22B8A8",
    }


def donut_bucket_distribution(df, bucket_col="bucket"):
    if df.empty or bucket_col not in df.columns:
        return px.pie(
            pd.DataFrame({"Bucket": [], "Count": []}),
            names="Bucket", values="Count", hole=0.68, height=300
        )

    counts = (
        df[bucket_col]
        .value_counts()
        .reindex(BUCKET_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    counts.columns = ["Bucket", "Count"]
    colors = _bucket_colors()

    fig = px.pie(
        counts,
        names="Bucket",
        values="Count",
        hole=0.68,
        height=300,
        color="Bucket",
        color_discrete_map=colors,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        marker=dict(line=dict(color="rgba(255,255,255,.9)", width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:,} reviews<br>%{percent}<extra></extra>",
    )
    fig.update_layout(
        margin=dict(t=5, b=5, l=5, r=5),
        legend_title_text="",
        legend=dict(font=dict(size=11)),
    )
    return style_plotly_fig(fig)


def bar_bucket_distribution(df, bucket_col="bucket"):
    if df.empty or bucket_col not in df.columns:
        return px.bar(
            pd.DataFrame({"Bucket": [], "Count": []}),
            x="Bucket", y="Count", height=300
        )

    counts = (
        df[bucket_col]
        .value_counts()
        .reindex(BUCKET_ORDER)
        .fillna(0)
        .astype(int)
    )
    chart_df = pd.DataFrame({"Bucket": counts.index, "Count": counts.values})
    colors = _bucket_colors()

    fig = px.bar(
        chart_df,
        x="Bucket",
        y="Count",
        height=300,
        color="Bucket",
        color_discrete_map=colors,
        text="Count",
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>%{y:,} reviews<extra></extra>",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=45, l=10, r=10),
        xaxis_title=None,
        yaxis_title=None,
    )
    return style_plotly_fig(fig)


def confidence_hist(df, proba_col="proba_pos"):
    if df.empty or proba_col not in df.columns:
        return px.histogram(
            pd.DataFrame({proba_col: []}),
            x=proba_col, nbins=30, height=300
        )

    fig = px.histogram(
        df,
        x=proba_col,
        nbins=30,
        height=300,
        color_discrete_sequence=["#8B5CF6"],
    )
    fig.update_traces(
        marker=dict(
            color="#8B5CF6",
            line=dict(color="rgba(255,255,255,.45)", width=0.5)
        ),
        hovertemplate="P(Positive): %{x:.2f}<br>Reviews: %{y:,}<extra></extra>",
    )
    fig.update_layout(
        margin=dict(t=10, b=35, l=10, r=10),
        xaxis_title="P(Positive)",
        yaxis_title=None,
    )
    return style_plotly_fig(fig)


def kw_bar(kws, height=320, title=None):
    if not kws:
        return None
    dfk = pd.DataFrame(kws, columns=["Keyword", "Mentions"])
    fig = px.bar(dfk, x="Mentions", y="Keyword", orientation="h", height=height)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title=None, xaxis_title=None, title=title)
    return style_plotly_fig(fig)

# NAVIGATION STATE
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "selected_bucket" not in st.session_state:
    st.session_state.selected_bucket = "Mixed 😐"
if "batch_df" not in st.session_state:
    st.session_state.batch_df = None

pages = [
    "Overview",
    "Category Details",
    "Batch Results",
    "Business Insights",
    "Tricky Reviews",
    "Trust & Reliability",
]
st.sidebar.markdown("""
<div class="rs-side-brand">
    <div class="rs-side-logo">•••</div>
    <div class="rs-side-brand-name">Review<span>Sense</span></div>
</div>
<div class="rs-side-caption">AI Review Intelligence</div>
<div class="rs-nav-label">Navigation</div>
""", unsafe_allow_html=True)

st.session_state.page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.page))

st.sidebar.markdown("""
<div class="rs-side-info">
    <div class="rs-side-info-title">🧠 Explanation Mode</div>
    <div class="rs-side-info-text">Switch between simple customer-friendly explanations and technical model insights.</div>
</div>
<div class="rs-side-info">
    <div class="rs-side-info-title">✨ About ReviewSense</div>
    <div class="rs-side-info-text">Understand customer reviews with AI-powered sentiment insights 💕</div>
</div>
""", unsafe_allow_html=True)

# PAGE: OVERVIEW
if st.session_state.page == "Overview":
    st.markdown('<div class="rs-section"><div class="rs-section-icon">📊</div><span class="rs-section-title">Overview</span></div>', unsafe_allow_html=True)

    if df_errors.empty:
        st.info(
            "Baseline report not found (misclassified.csv). "
            "Upload a file in Quick Analyze to test the model, or re-run training to generate reports."
        )

    total = int(len(df_errors)) if not df_errors.empty else 0
    pos_rate = float((df_errors["true_label"] == 1).mean()) if (not df_errors.empty and "true_label" in df_errors.columns) else 0.0
    neg_rate = 1 - pos_rate if total else 0.0
    best_f1 = float(df_compare.iloc[0]["F1"]) if (not df_compare.empty and "F1" in df_compare.columns) else None

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='rs-kpi-card'><div class='rs-kpi-icon pink'>💬</div><div><div class='rs-kpi-label'>Total Reviews</div>"
            f"<div class='rs-kpi-value'>{total:,}</div><div class='rs-kpi-sub'>Baseline evaluation set</div></div></div>",
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"<div class='rs-kpi-card'><div class='rs-kpi-icon green'>☺</div><div><div class='rs-kpi-label'>Positive Rate</div>"
            f"<div class='rs-kpi-value green'>{pos_rate*100:.1f}%</div><div class='rs-kpi-sub'>From true labels</div></div></div>",
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"<div class='rs-kpi-card'><div class='rs-kpi-icon red'>☹</div><div><div class='rs-kpi-label'>Negative Rate</div>"
            f"<div class='rs-kpi-value red'>{neg_rate*100:.1f}%</div><div class='rs-kpi-sub'>From true labels</div></div></div>",
            unsafe_allow_html=True
        )
    with k4:
        label = explain("AI Reliability", "Best F1 Score")
        val = "High" if st.session_state.mode == "Simple Language" else (f"{best_f1:.4f}" if best_f1 is not None else "N/A")
        sub = explain("Consistency of predictions", "Top-ranked model")
        st.markdown(
            f"<div class='rs-kpi-card'><div class='rs-kpi-icon purple'>✓</div><div><div class='rs-kpi-label'>{label}</div>"
            f"<div class='rs-kpi-value purple'>{val}</div><div class='rs-kpi-sub'>{sub}</div></div></div>",
            unsafe_allow_html=True
        )

    st.markdown('<div class="rs-info-bar">⚡ <b>Quick Analyze</b> — type a review or upload a file to get instant sentiment insights.</div>', unsafe_allow_html=True)

    st.markdown('<div class="rs-section"><div class="rs-section-icon">◔</div><span class="rs-section-title">Sentiment Breakdown</span><span class="rs-section-sub">5 Levels</span></div>', unsafe_allow_html=True)
    st.caption(explain(
        "Click any category to explore examples and understand what drives it.",
        "Buckets are based on calibrated probability ranges."
    ))

    if not df_errors.empty and "bucket" in df_errors.columns:
        bucket_counts = df_errors["bucket"].value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
    else:
        bucket_counts = pd.Series({b: 0 for b in BUCKET_ORDER})

    card_cols = st.columns(5)
    overview_classes = [
        ("strong-neg", "Strongly Negative 😡"),
        ("neg", "Negative 🙁"),
        ("mixed", "Mixed 😐"),
        ("pos", "Positive 🙂"),
        ("strong-pos", "Strongly Positive 😍"),
    ]

    for i, bucket_name in enumerate(BUCKET_ORDER):
        with card_cols[i]:
            count = int(bucket_counts.get(bucket_name, 0))
            pct = (count / total) * 100 if total else 0
            cls, label = overview_classes[i]
            st.markdown(render_sentiment_card_html(cls, label, count, pct), unsafe_allow_html=True)

            if st.button("Explore →", key=f"overview_sentiment_{i}", use_container_width=True, disabled=(total == 0)):
                st.session_state.selected_bucket = bucket_name
                st.session_state.page = "Category Details"
                st.rerun()

    c1, c2, c3 = st.columns([1.1, 1.2, 1.2])

    with c1:
        st.markdown("<div class='rs-chart-head'><div class='rs-chart-icon'>◔</div><div class='rs-chart-title'>Distribution</div></div>", unsafe_allow_html=True)
        st.plotly_chart(donut_bucket_distribution(df_errors), use_container_width=True)

    with c2:
        st.markdown("<div class='rs-chart-head'><div class='rs-chart-icon'>▥</div><div class='rs-chart-title'>Counts by Category</div></div>", unsafe_allow_html=True)
        st.plotly_chart(bar_bucket_distribution(df_errors), use_container_width=True)

    with c3:
        st.markdown("<div class='rs-chart-head'><div class='rs-chart-icon'>♧</div><div class='rs-chart-title'>Confidence Spread</div></div>", unsafe_allow_html=True)
        st.plotly_chart(confidence_hist(df_errors), use_container_width=True)

    # Quick Analyze
    st.markdown("""
    <div class="rs-quick">
        <div class="rs-quick-title">⚡ Quick Analyze a Review</div>
        <div class="rs-quick-sub">Type a customer review below to get instant sentiment prediction and explanation.</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(explain(
        "Analyze a single review or upload a file to get instant insights.",
        "Supports CSV / TSV / XLSX with automatic column detection."
    ))

    qa1, qa2 = st.columns([1.2, 1])

    with qa1:
        review_text = st.text_area("Single review", height=120, placeholder="Paste any customer review here…")

    with qa2:
        st.markdown(
            """
            <div class="rs-upload-card">
                <div class="rs-upload-head">
                    <div class="rs-upload-icon">☁️</div>
                    <div>
                        <div class="rs-upload-title">Upload your review dataset</div>
                    </div>
                </div>
                <div class="rs-upload-sub">Drop a file here or browse from your computer</div>
                <div class="rs-upload-formats">
                    <span class="rs-file-chip">CSV</span>
                    <span class="rs-file-chip">TSV</span>
                    <span class="rs-file-chip">XLSX</span>
                    <span class="rs-file-chip">Up to 200 MB</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        uploaded = st.file_uploader(
            "Upload review dataset",
            type=["csv", "tsv", "xlsx"],
            label_visibility="collapsed",
            help="Upload a CSV, TSV, or XLSX containing customer reviews."
        )
        if uploaded is not None:
            st.markdown(
                f'<div class="rs-upload-success">✓ {uploaded.name} ready for analysis</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        threshold = st.slider(explain("Sensitivity", "Decision threshold"), 0.1, 0.9, 0.5, 0.05)

    run = st.button("Run Analysis", use_container_width=True)

    if run:
        if review_text.strip():
            p = float(model.predict_proba([review_text])[0][1])
            bucket = bucketize(p)
            verdict = "Positive" if p >= threshold else "Negative"

            st.success(explain(
                f"Result: {bucket} — {bucket_explainer(bucket)}",
                f"P(Positive)={p:.4f} | threshold={threshold:.2f} | pred={verdict}"
            ))

            pos_terms, neg_terms = explain_review_terms(model, review_text, top_k_each=12)
            st.markdown("**Highlighted cues (green helps / red hurts):**")
            st.markdown(highlight_terms_both(review_text, pos_terms, neg_terms), unsafe_allow_html=True)

            st.session_state.selected_bucket = bucket
            st.session_state.page = "Category Details"
            st.rerun()

        elif uploaded is not None:
            try:
                df_up = load_any_table(uploaded)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.stop()

            if df_up.empty:
                st.warning("Uploaded file is empty.")
                st.stop()

            st.markdown("#### File preview")
            st.dataframe(df_up.head(10), use_container_width=True)

            guessed_text = guess_text_column(df_up)
            guessed_product = guess_product_column(df_up)
            guessed_rating = guess_rating_column(df_up)
            guessed_date = guess_date_column(df_up)

            st.markdown("#### Column mapping (auto-detected)")
            cols = list(df_up.columns)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                text_col = st.selectbox("Text column", options=cols,
                                        index=cols.index(guessed_text) if guessed_text in cols else 0)
            with m2:
                product_col = st.selectbox("Product (optional)", ["(none)"] + cols)
            with m3:
                rating_col = st.selectbox("Rating (optional)", ["(none)"] + cols)
            with m4:
                date_col = st.selectbox("Date (optional)", ["(none)"] + cols)

            texts = df_up[text_col].astype(str).fillna("").tolist()
            proba = model.predict_proba(texts)[:, 1]

            out = df_up.copy()
            out["_text_col_used"] = text_col
            out["proba_pos"] = proba
            out["bucket"] = [bucketize(pv) for pv in proba]
            out["prediction"] = np.where(out["proba_pos"] >= threshold, "Positive", "Negative")
            out["confidence"] = np.round(np.abs(out["proba_pos"] - 0.5) * 2, 4)

            if product_col != "(none)":
                out["_product"] = out[product_col].astype(str)
            if rating_col != "(none)":
                out["_rating"] = pd.to_numeric(out[rating_col], errors="coerce")
            if date_col != "(none)":
                out["_date"] = pd.to_datetime(out[date_col], errors="coerce")

            st.session_state.batch_df = out
            st.session_state.page = "Batch Results"
            st.rerun()
        else:
            st.warning("Type a review OR upload a file to analyze.")


# PAGE: CATEGORY DETAILS
elif st.session_state.page == "Category Details":
    if df_errors.empty:
        st.info("Baseline reports not available. Use Overview → Quick Analyze upload to explore your dataset.")
        st.stop()

    bucket = st.selectbox("Choose a category", options=BUCKET_ORDER,
                          index=BUCKET_ORDER.index(st.session_state.selected_bucket))
    st.session_state.selected_bucket = bucket
    highlight_level = st.select_slider("Highlight strength", options=["Low", "Medium", "High"], value="Medium")
    k_each = get_highlight_k(highlight_level)

    st.subheader(f"🔎 Category Details — {bucket}")
    st.write(explain(bucket_explainer(bucket), f"Bucket: {bucket}"))

    subset = df_errors[df_errors["bucket"] == bucket].copy()
    st.info(summarize_bucket_insights(subset))
    total = len(df_errors)

    st.caption(f"{len(subset):,} reviews ({(len(subset)/total*100 if total else 0):.1f}%) in this category (baseline subset).")

    a, b = st.columns([1.25, 1.0])

    with a:
        st.markdown("#### Examples (most confident)")
        show = subset.sort_values("confidence", ascending=False).head(12)

        for _, r in show.iterrows():
            txt = str(r.get("review_raw", ""))
            p = float(r.get("proba_pos", 0.5))

            pos_terms, neg_terms = explain_review_terms(model, txt, top_k_each=k_each)
            pos_terms = filter_terms(pos_terms)
            neg_terms = filter_terms(neg_terms)

            mixed_badge = ""
            if pos_terms and neg_terms:
                mixed_badge = "<span class='rs-chip'>⚠ Mixed signals</span>"

            chip_terms = (neg_terms[:4] + pos_terms[:4])
            chips_html = "".join([f"<span class='rs-chip'>{t}</span>" for t in chip_terms])

            sent_info = sentence_scores(model, txt)
            best_pos = max(sent_info, key=lambda x: x["net"], default=None)
            best_neg = min(sent_info, key=lambda x: x["net"], default=None)

            sentence_html = ""
            if best_pos and best_neg and best_pos["sentence"] != best_neg["sentence"]:
                sentence_html = (
                    "<div class='rs-sub' style='margin-top:10px;'>"
                    f"<b>{explain('Most positive sentence:', 'Top positive sentence:')}</b> {best_pos['sentence']}<br/>"
                    f"<b>{explain('Most negative sentence:', 'Top negative sentence:')}</b> {best_neg['sentence']}"
                    "</div>"
                )

            card_html = (
                "<div class='rs-card'>"
                f"<div class='rs-sub'>{explain('AI score', 'P(Positive)')}: <b>{p:.3f}</b> {mixed_badge}</div>"
                f"<div style='margin-top:8px'>{highlight_terms_both(txt, pos_terms, neg_terms)}</div>"
                f"<div class='rs-sub' style='margin-top:10px;'>"
                f"{explain('What influenced it:', 'Top contributing terms:')} "
                f"{chips_html}"
                "</div>"
                f"{sentence_html}"
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)

    with b:
        st.markdown("#### Common Themes (keywords)")
        kws = top_keywords(subset, "review_clean", n=15)
        fig = kw_bar(kws, height=340)
        if fig is None:
            st.info("Not enough text to extract keywords.")
        else:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### What this likely means")
        st.write(explain(
            "These keywords usually hint at what customers care about in this group.",
            "Keyword counts from cleaned text in this subset."
        ))


# PAGE: BATCH RESULTS
elif st.session_state.page == "Batch Results":
    st.markdown(
        '<div class="rs-section"><div class="rs-section-icon">📦</div>'
        '<span class="rs-section-title">Batch Results</span>'
        '<span class="rs-section-sub">Uploaded file analysis</span></div>',
        unsafe_allow_html=True
    )

    out = st.session_state.batch_df
    if out is None or len(out) == 0:
        st.info("No batch results yet. Go to Overview → upload a file → Run Analysis.")
        st.stop()

    text_col_used = safe_textcol_from_batch(out)
    if text_col_used is None or text_col_used not in out.columns:
        st.error("Text column missing in batch results. Upload again and confirm mapping.")
        st.stop()

    total = len(out)

    # Exact counts from the model's 5-level prediction buckets.
    bucket_counts = (
        out["bucket"].value_counts()
        .reindex(BUCKET_ORDER)
        .fillna(0)
        .astype(int)
    )

    strongly_neg = int(bucket_counts["Strongly Negative 😡"])
    negative = int(bucket_counts["Negative 🙁"])
    mixed = int(bucket_counts["Mixed 😐"])
    positive = int(bucket_counts["Positive 🙂"])
    strongly_pos = int(bucket_counts["Strongly Positive 😍"])

    neg_count = strongly_neg + negative
    pos_count = positive + strongly_pos

    pct_pos = (pos_count / total * 100) if total else 0
    pct_neg = (neg_count / total * 100) if total else 0
    pct_mix = (mixed / total * 100) if total else 0
    avg_conf = float(out["confidence"].mean()) if "confidence" in out.columns else 0.0

    # High-confidence examples are used only for the review showcase.
    high_risk = out[
        (out["bucket"].isin(["Strongly Negative 😡", "Negative 🙁"]))
        & (out["confidence"] >= 0.70)
    ].copy()

    high_praise = out[
        (out["bucket"].isin(["Strongly Positive 😍", "Positive 🙂"]))
        & (out["confidence"] >= 0.70)
    ].copy()

    # ======================================================
    # 1. EXECUTIVE SUMMARY — exact numbers
    # ======================================================
    k1, k2, k3, k4 = st.columns(4)

    kpi_data = [
        ("💬", "Total Reviews", f"{total:,}", "Reviews analyzed", "pink"),
        ("☺", "Positive Reviews", f"{pos_count:,}", f"{pct_pos:.1f}% of all reviews", "green"),
        ("☹", "Negative Reviews", f"{neg_count:,}", f"{pct_neg:.1f}% of all reviews", "red"),
        ("✓", "Mixed Reviews", f"{mixed:,}", f"{pct_mix:.1f}% of all reviews", "purple"),
    ]

    for col, (icon, label, value, sub, tone) in zip([k1, k2, k3, k4], kpi_data):
        with col:
            st.markdown(
                f"""
                <div class="rs-kpi-card">
                    <div class="rs-kpi-icon {tone}">{icon}</div>
                    <div>
                        <div class="rs-kpi-label">{label}</div>
                        <div class="rs-kpi-value">{value}</div>
                        <div class="rs-kpi-sub">{sub}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        f"""
        <div class="rs-batch-summary">
            <b>📊 Dataset Summary</b>
            &nbsp; {total:,} reviews analyzed
            &nbsp; • &nbsp; <span style="color:#18743b"><b>{pos_count:,}</b> positive ({pct_pos:.1f}%)</span>
            &nbsp; • &nbsp; <span style="color:#b52c45"><b>{neg_count:,}</b> negative ({pct_neg:.1f}%)</span>
            &nbsp; • &nbsp; <b>{mixed:,}</b> mixed ({pct_mix:.1f}%)
            &nbsp; • &nbsp; Average confidence <b>{avg_conf:.2f}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ======================================================
    # 2. 5-LEVEL SENTIMENT CARDS
    # ======================================================
    st.markdown(
        '<div class="rs-section"><div class="rs-section-icon">📊</div>'
        '<span class="rs-section-title">Sentiment Breakdown</span>'
        '<span class="rs-section-sub">5 prediction levels</span></div>',
        unsafe_allow_html=True
    )

    sentiment_cards = [
        ("strong-neg", "Strongly Negative 😡", strongly_neg),
        ("neg", "Negative 🙁", negative),
        ("mixed", "Mixed 😐", mixed),
        ("pos", "Positive 🙂", positive),
        ("strong-pos", "Strongly Positive 😍", strongly_pos),
    ]

    cols = st.columns(5)
    for col, (cls, label, count) in zip(cols, sentiment_cards):
        pct = count / total * 100 if total else 0
        with col:
            st.markdown(render_sentiment_card_html(cls, label, count, pct), unsafe_allow_html=True)

    # ======================================================
    # 3. COLOURED CHARTS
    # ======================================================
    c1, c2, c3 = st.columns([1.05, 1.12, 1.12])

    with c1:
        st.markdown(
            '<div class="rs-chart-head"><div class="rs-chart-icon">◔</div>'
            '<div class="rs-chart-title">Distribution</div></div>',
            unsafe_allow_html=True
        )
        st.plotly_chart(
            donut_bucket_distribution(out, bucket_col="bucket"),
            use_container_width=True,
            key="batch_donut"
        )

    with c2:
        st.markdown(
            '<div class="rs-chart-head"><div class="rs-chart-icon">▥</div>'
            '<div class="rs-chart-title">Counts by Category</div></div>',
            unsafe_allow_html=True
        )
        st.plotly_chart(
            bar_bucket_distribution(out, bucket_col="bucket"),
            use_container_width=True,
            key="batch_bar"
        )

    with c3:
        st.markdown(
            '<div class="rs-chart-head"><div class="rs-chart-icon">♧</div>'
            '<div class="rs-chart-title">Confidence Spread</div></div>',
            unsafe_allow_html=True
        )
        st.plotly_chart(
            confidence_hist(out, proba_col="proba_pos"),
            use_container_width=True,
            key="batch_confidence"
        )

    st.markdown('<div class="rs-batch-divider"></div>', unsafe_allow_html=True)

    # ======================================================
    # 4. POSITIVE / NEGATIVE REVIEW SHOWCASE
    # ======================================================
    st.markdown(
        '<div class="rs-section"><div class="rs-section-icon">💬</div>'
        '<span class="rs-section-title">Customer Reviews</span>'
        '<span class="rs-section-sub">High-confidence examples from your uploaded dataset</span></div>',
        unsafe_allow_html=True
    )

    review_left, review_right = st.columns(2)

    with review_left:
        st.markdown(
            '<div class="rs-review-panel positive">'
            '<div class="rs-review-title">🟢 Positive Reviews '
            '<span class="badge green">CUSTOMERS LOVE IT</span></div>',
            unsafe_allow_html=True
        )

        positive_examples = (
            high_praise.sort_values("proba_pos", ascending=False).head(3)
            if not high_praise.empty else
            out[out["bucket"].isin(["Strongly Positive 😍", "Positive 🙂"])]
            .sort_values("proba_pos", ascending=False).head(3)
        )

        if positive_examples.empty:
            st.info("No positive reviews available.")
        else:
            for _, row in positive_examples.iterrows():
                review_text = str(row[text_col_used]).strip()
                review_text = review_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                score = float(row["proba_pos"])
                st.markdown(
                    f'<div class="rs-review-item">'
                    f'<div class="rs-review-score">Positive score {score:.2f}</div>'
                    f'{review_text[:420]}'
                    f'{"..." if len(review_text) > 420 else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with review_right:
        st.markdown(
            '<div class="rs-review-panel negative">'
            '<div class="rs-review-title">🔴 Negative Reviews '
            '<span class="badge red">NEEDS ATTENTION</span></div>',
            unsafe_allow_html=True
        )

        negative_examples = (
            high_risk.sort_values("proba_pos", ascending=True).head(3)
            if not high_risk.empty else
            out[out["bucket"].isin(["Strongly Negative 😡", "Negative 🙁"])]
            .sort_values("proba_pos", ascending=True).head(3)
        )

        if negative_examples.empty:
            st.info("No negative reviews available.")
        else:
            for _, row in negative_examples.iterrows():
                review_text = str(row[text_col_used]).strip()
                review_text = review_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                score = float(row["proba_pos"])
                st.markdown(
                    f'<div class="rs-review-item">'
                    f'<div class="rs-review-score">Positive score {score:.2f} • negative signal</div>'
                    f'{review_text[:420]}'
                    f'{"..." if len(review_text) > 420 else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # 5. TOP THEMES — COLOURED
    # ======================================================
    st.markdown(
        '<div class="rs-section"><div class="rs-section-icon">🎯</div>'
        '<span class="rs-section-title">What Customers Are Saying</span></div>',
        unsafe_allow_html=True
    )

    neg_texts = out[out["bucket"].isin(["Strongly Negative 😡", "Negative 🙁"])][text_col_used].astype(str).tolist()
    pos_texts = out[out["bucket"].isin(["Strongly Positive 😍", "Positive 🙂"])][text_col_used].astype(str).tolist()

    top_pos_phrases, _ = extract_top_phrases_from_group(model, pos_texts, top_n=10)
    _, top_neg_phrases = extract_top_phrases_from_group(model, neg_texts, top_n=10)

    t1, t2 = st.columns(2)

    with t1:
        st.markdown(
            '<div class="rs-theme-panel">'
            '<div class="rs-theme-title">🟢 Top Positive Themes</div>',
            unsafe_allow_html=True
        )
        if top_pos_phrases:
            chips = "".join(
                f'<span class="rs-theme-chip green">{p}</span>'
                for p in top_pos_phrases[:10]
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("Not enough positive phrase signals.")
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        st.markdown(
            '<div class="rs-theme-panel">'
            '<div class="rs-theme-title">🔴 Top Complaint Themes</div>',
            unsafe_allow_html=True
        )
        if top_neg_phrases:
            chips = "".join(
                f'<span class="rs-theme-chip red">{p}</span>'
                for p in top_neg_phrases[:10]
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("Not enough negative phrase signals.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="rs-batch-divider"></div>', unsafe_allow_html=True)

    # ======================================================
    # 6. DRILLDOWN
    # ======================================================
    st.markdown(
        '<div class="rs-section"><div class="rs-section-icon">🔎</div>'
        '<span class="rs-section-title">Drilldown by Category</span></div>',
        unsafe_allow_html=True
    )
    st.caption("Pick a category to see examples, themes, and what it means.")

    d1, d2, d3 = st.columns([1.1, 1.0, 1.0])
    with d1:
        pick_bucket = st.selectbox("Category", ["All"] + BUCKET_ORDER, index=0)
    with d2:
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.35, 0.05)
    with d3:
        keyword = st.text_input("Search keyword (optional)")

    view = out.copy()
    if pick_bucket != "All":
        view = view[view["bucket"] == pick_bucket]
    view = view[view["confidence"] >= min_conf]
    if keyword.strip():
        view = view[view[text_col_used].astype(str).str.contains(keyword, case=False, na=False)]

    st.markdown(
        f"<div class='rs-card'><div class='rs-title'>Filtered results</div>"
        f"<div class='rs-kpi'>{len(view):,}</div>"
        f"<div class='rs-sub'>reviews match your filters</div></div>",
        unsafe_allow_html=True
    )

    if pick_bucket != "All" and len(view) > 0:
        bucket_texts = view[text_col_used].astype(str).tolist()
        pos_phr, neg_phr = extract_top_phrases_from_group(model, bucket_texts, top_n=10)

        st.markdown("### 📌 Key Themes in this Category")
        chips = "".join(
            [f"<span class='rs-theme-chip red'>{p}</span>" for p in neg_phr[:5]]
            + [f"<span class='rs-theme-chip green'>{p}</span>" for p in pos_phr[:5]]
        )
        st.markdown(f"<div class='rs-theme-panel'>{chips}</div>", unsafe_allow_html=True)



# PAGE: BUSINESS INSIGHTS
elif st.session_state.page == "Business Insights":
    st.subheader("💡 Business Insights (Executive View)")

    active_df, source_label, text_col = get_active_insights_df()
    st.caption(f"Source: **{source_label}** — insights update automatically when you upload a file.")

    if active_df.empty:
        st.info("No data available yet. Upload a CSV/TSV/XLSX in Overview → Quick Analyze.")
        st.stop()

    if "bucket" not in active_df.columns:
        if "proba_pos" in active_df.columns:
            active_df = active_df.copy()
            active_df["bucket"] = active_df["proba_pos"].apply(bucketize)
        else:
            st.error("This dataset doesn't contain sentiment probabilities/buckets yet.")
            st.stop()

    if text_col is None or text_col not in active_df.columns:
        st.error("Text column not found for insights. Upload again and confirm text column mapping.")
        st.stop()

    if "review_clean" not in active_df.columns:
        active_df = active_df.copy()
        active_df["review_clean"] = (
            active_df[text_col]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    neg_group = active_df[active_df["bucket"].isin(["Strongly Negative 😡", "Negative 🙁"])].copy()
    pos_group = active_df[active_df["bucket"].isin(["Strongly Positive 😍", "Positive 🙂"])].copy()
    mix_group = active_df[active_df["bucket"] == "Mixed 😐"].copy()

    pos_texts = pos_group[text_col].astype(str).tolist()
    neg_texts = neg_group[text_col].astype(str).tolist()

    top_pos_phrases, _ = extract_top_phrases_from_group(model, pos_texts, top_n=10)
    _, top_neg_phrases = extract_top_phrases_from_group(model, neg_texts, top_n=10)

    total = len(active_df)
    pct_pos = (len(pos_group) / total * 100) if total else 0
    pct_neg = (len(neg_group) / total * 100) if total else 0
    pct_mix = (len(mix_group) / total * 100) if total else 0
    focus_index = (pct_neg + 0.5 * pct_mix)

    k1, k2, k3, k4 = st.columns(4)

    def bi_review_card(title, icon, count, pct, kind, subtitle):
        return f"""
        <div class='rs-bi-review-card {kind}' style='--bi-pct:{pct:.1f}%;'>
            <div class='rs-bi-review-head'>
                <div class='rs-bi-review-title'>{title}</div>
                <div class='rs-bi-review-icon'>{icon}</div>
            </div>
            <div class='rs-bi-review-stat'>
                <span class='rs-bi-review-count'>{count:,}</span>
                <span class='rs-bi-review-pct'>{pct:.1f}%</span>
            </div>
            <div class='rs-bi-review-sub'>{subtitle}</div>
            <div class='rs-bi-review-track'><div class='rs-bi-review-fill'></div></div>
            <div class='rs-bi-review-foot'>
                <span>{count:,} reviews</span>
                <span>{pct:.1f}% of dataset</span>
            </div>
        </div>
        """

    with k1:
        st.markdown(bi_review_card("Positive Reviews", "😊", len(pos_group), pct_pos, "positive", "Satisfied customers"), unsafe_allow_html=True)
    with k2:
        st.markdown(bi_review_card("Negative Reviews", "😟", len(neg_group), pct_neg, "negative", "Customer pain points"), unsafe_allow_html=True)
    with k3:
        st.markdown(bi_review_card("Mixed Reviews", "😐", len(mix_group), pct_mix, "mixed", "Pros + cons in one review"), unsafe_allow_html=True)
    with k4:
        focus_pct = min(max(focus_index, 0), 100)
        st.markdown(bi_review_card("Focus Index", "🎯", int(round(focus_index)), focus_pct, "mixed", "Higher score = fix priority"), unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1.15, 1.0])

    with left:
        st.markdown("### ✅ What Customers Appreciate")
        if not top_pos_phrases:
            st.info("Not enough signal to extract strong positive themes.")
        else:
            chips = "".join([f"<span class='rs-bi-theme-chip'>{p}</span>" for p in top_pos_phrases[:8]])
            st.markdown(
                f"<div class='rs-bi-theme-card positive'>"
                f"<div class='rs-bi-theme-label'>Top praise themes</div>"
                f"<div class='rs-bi-theme-chips'>{chips}</div></div>",
                unsafe_allow_html=True
            )

        if len(pos_group) > 0:
            sample_pos = pos_group.sort_values("confidence", ascending=False).head(1) if "confidence" in pos_group.columns else pos_group.head(1)
            row = sample_pos.iloc[0]
            quote = str(row.get(text_col, "")).strip()[:320]
            st.markdown(
                f"<div class='rs-bi-review-card positive' style='--bi-pct:{pct_pos:.1f}%; margin-top:12px;'>"
                f"<div class='rs-bi-review-head'><div class='rs-bi-review-title'>⭐ Positive review highlight</div><div class='rs-bi-review-icon'>✓</div></div>"
                f"<div class='rs-bi-quote' style='--bi-color:#35D07F'><strong>Customer:</strong> “{quote}”</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("### ⚠️ What’s Driving Complaints")
        if not top_neg_phrases:
            st.info("Not enough signal to extract strong complaint themes.")
        else:
            chips = "".join([f"<span class='rs-bi-theme-chip'>{p}</span>" for p in top_neg_phrases[:8]])
            st.markdown(
                f"<div class='rs-bi-theme-card negative'>"
                f"<div class='rs-bi-theme-label'>Top complaint themes</div>"
                f"<div class='rs-bi-theme-chips'>{chips}</div></div>",
                unsafe_allow_html=True
            )

        if len(neg_group) > 0:
            sample_neg = neg_group.sort_values("confidence", ascending=False).head(1) if "confidence" in neg_group.columns else neg_group.head(1)
            row = sample_neg.iloc[0]
            quote = str(row.get(text_col, "")).strip()[:320]
            st.markdown(
                f"<div class='rs-bi-review-card negative' style='--bi-pct:{pct_neg:.1f}%; margin-top:12px;'>"
                f"<div class='rs-bi-review-head'><div class='rs-bi-review-title'>🚨 Negative review highlight</div><div class='rs-bi-review-icon'>!</div></div>"
                f"<div class='rs-bi-quote' style='--bi-color:#FF5570'><strong>Customer:</strong> “{quote}”</div></div>",
                unsafe_allow_html=True
            )

    with right:
        st.markdown("### 🎯 Recommended Actions")
        action_items = []
        for p in top_neg_phrases[:6]:
            if any(x in p for x in ["not", "poor", "waste", "broken", "refund", "return", "late"]):
                action_items.append(f"Fix urgently: **{p}**")
            else:
                action_items.append(f"Improve: **{p}**")

        if not action_items:
            action_items = ["Monitor feedback weekly and validate top complaint drivers."]

        st.markdown("<div class='rs-card'>", unsafe_allow_html=True)
        for i, a in enumerate(action_items, 1):
            st.markdown(f"{i}. {a}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 🧾 Proof (Customer Quotes)")
        quote_df = neg_group.copy()
        if "confidence" in quote_df.columns:
            quote_df = quote_df.sort_values("confidence", ascending=False)
        quote_df = quote_df.head(2)
        for _, row in quote_df.iterrows():
            quote = str(row.get(text_col, ""))[:260].strip()
            st.markdown(f"<div class='rs-card'><div class='rs-sub'>Customer said</div>“{quote}...”</div>", unsafe_allow_html=True)

    st.markdown("---")

    pos_line = ", ".join(top_pos_phrases[:4]) if top_pos_phrases else "strong experience"
    neg_line = ", ".join(top_neg_phrases[:4]) if top_neg_phrases else "recurring issues"

    st.markdown("### 📌 Shareable Summary")
    st.success(
        explain(
            f"Customers praise {pos_line}. Biggest complaints are about {neg_line}. Fix complaints first to reduce churn.",
            f"Praise drivers: {pos_line} | Complaint drivers: {neg_line}"
        )
    )

    if source_label == "Baseline Sample":
        st.info("Upload a CSV/TSV/XLSX in Overview to get Business Insights for your own dataset.")


# PAGE: TRICKY REVIEWS
elif st.session_state.page == "Tricky Reviews":
    st.subheader("🧪 Tricky Reviews (AI Limitations)")
    st.caption(explain(
        "Some reviews are naturally hard: mixed feelings, unclear wording, very short text, or tricky negation.",
        "These cases help you understand where the model can be uncertain or misread intent."
    ))

    active_df, source_label, text_col = get_active_insights_df()
    st.caption(f"Source: **{source_label}**")

    if active_df.empty:
        st.info("No data available yet. Upload a CSV/TSV/XLSX in Overview → Quick Analyze.")
        st.stop()

    if text_col is None or text_col not in active_df.columns:
        st.error("Text column not found for this dataset.")
        st.stop()

    df = active_df.copy()

    if "proba_pos" not in df.columns:
        st.error("This dataset does not contain proba_pos. Run predictions first (Overview → Upload → Run Analysis).")
        st.stop()
    if "bucket" not in df.columns:
        df["bucket"] = df["proba_pos"].apply(bucketize)
    if "confidence" not in df.columns:
        df["confidence"] = np.round(np.abs(df["proba_pos"] - 0.5) * 2, 4)

    def detect_reasons(text: str):
        t = (text or "").lower()
        reasons = []
        if any(w in t for w in [" but ", " however ", " although ", " though ", " yet "]):
            reasons.append("Mixed feelings")
        if any(w in t for w in [" not ", " never ", " no ", "n't "]):
            reasons.append("Negation (not/never)")
        if len(t.split()) <= 5:
            reasons.append("Too short / low context")
        if "!!" in t or "??" in t or (sum(ch.isupper() for ch in (text or "")) > 10):
            reasons.append("Emphasis / tone (caps/punct)")
        if not reasons:
            reasons.append("Unclear / subtle wording")
        return reasons

    df["_reasons"] = df[text_col].astype(str).apply(detect_reasons)
    df["_reason_main"] = df["_reasons"].apply(lambda xs: xs[0] if xs else "Unclear / subtle wording")
    df["_uncertain"] = (df["proba_pos"].between(0.45, 0.65)) | (df["confidence"] < 0.35)

    counts = reason_counts(df)

    st.markdown("### 📌 Tricky Review Categories")
    st.caption("See which review patterns need the most attention. Each card visualizes its share of the uploaded dataset.")

    if "tricky_focus" not in st.session_state:
        st.session_state.tricky_focus = "All tricky reviews"

    # Build category data first so the UI and chart always use the exact same counts.
    tricky_items = [
        ("Needs Human Review", "Needs manual review (uncertain)", "#FF4D6D", "🧠"),
        ("Mixed Feelings", "Mixed sentiment cases", "#FFB020", "😐"),
        ("Confusing Wording", "Negation cases", "#9B6CFF", "❓"),
        ("Too Little Detail", "Very short reviews", "#3DA9FC", "📝"),
        ("Strong Tone / Emphasis", "Emphasis / tone cases", "#20C7B5", "🔥"),
    ]

    total_all = len(df)
    max_count = max([counts.get(key, 0) for _, key, _, _ in tricky_items] + [1])

    # Visual category cards. The card itself is visual; the small button underneath
    # remains the actual Streamlit interaction so filtering still works reliably.
    card_cols = st.columns(5)

    for col, (label, key, color, icon) in zip(card_cols, tricky_items):
        count = int(counts.get(key, 0))
        share = (count / total_all * 100) if total_all else 0
        visual_width = (count / max_count * 100) if max_count else 0
        active = st.session_state.tricky_focus == key

        border = color if active else "rgba(255,255,255,0.14)"
        glow = f"0 0 22px {color}35" if active else "0 8px 22px rgba(0,0,0,.14)"
        bg = f"linear-gradient(145deg, {color}18, rgba(255,255,255,.045))"

        card = (
            f'<div style="background:{bg};border:1px solid {border};'
            f'border-top:3px solid {color};border-radius:16px;padding:14px 14px 13px;'
            f'min-height:172px;box-shadow:{glow};">'
            f'<div style="font-size:13px;font-weight:800;color:#E8EBF8;min-height:38px;line-height:1.35;">'
            f'{icon}&nbsp; {label}</div>'
            f'<div style="display:flex;align-items:flex-end;justify-content:space-between;margin-top:8px;">'
            f'<div style="font-size:25px;font-weight:900;color:{color};line-height:1;">{count:,}</div>'
            f'<div style="font-size:12px;font-weight:800;color:#AEB6D4;">{share:.1f}%</div>'
            f'</div>'
            f'<div style="height:8px;margin-top:15px;border-radius:99px;background:rgba(255,255,255,.09);overflow:hidden;">'
            f'<div style="width:{visual_width:.2f}%;height:100%;border-radius:99px;'
            f'background:linear-gradient(90deg,{color},{color}CC);box-shadow:0 0 12px {color}70;"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;margin-top:7px;font-size:10px;color:#7F89A8;">'
            f'<span>review volume</span><span style="color:#C9D0E8;font-weight:700;">{share:.1f}% of dataset</span>'
            f'</div>'
            f'</div>'
        )
        with col:
            st.markdown(card, unsafe_allow_html=True)
            if st.button(
                "Selected" if active else "View reviews",
                key=f"tricky_{key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.tricky_focus = key
                st.rerun()

    if st.button("↻ Show All Tricky Reviews", use_container_width=True, key="tricky_show_all"):
        st.session_state.tricky_focus = "All tricky reviews"
        st.rerun()

    focus = st.session_state.tricky_focus
    FRIENDLY_FOCUS = {
        "All tricky reviews": "All tricky reviews",
        "Needs manual review (uncertain)": "Needs Human Review",
        "Mixed sentiment cases": "Mixed Feelings",
        "Negation cases": "Confusing Wording",
        "Very short reviews": "Too Little Detail",
        "Emphasis / tone cases": "Strong Tone / Emphasis",
    }
    focus_label = FRIENDLY_FOCUS.get(focus, focus)

    chart_df = pd.DataFrame({
        "Category": [x[0] for x in tricky_items],
        "Count": [counts.get(x[1], 0) for x in tricky_items],
        "Color": [x[2] for x in tricky_items],
    })

    fig = px.bar(
        chart_df,
        x="Count",
        y="Category",
        orientation="h",
        text="Count",
        color="Category",
        color_discrete_sequence=[x[2] for x in tricky_items],
        height=285,
    )
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=15, b=10, l=10, r=55),
        xaxis_title=None,
        yaxis_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E9ECF8"),
        bargap=0.28,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        tickfont=dict(color="#8F98B5"),
    )
    fig.update_yaxes(
        showgrid=False,
        tickfont=dict(color="#C8CEE2", size=11),
        categoryorder="array",
        categoryarray=[x[0] for x in tricky_items],
    )

    st.markdown(
        '<div style="margin-top:16px;margin-bottom:4px;font-size:14px;font-weight:800;color:#E9ECF8;">📊 Category volume</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c2, c3 = st.columns([1.2, 1.2])
    with c2:
        max_rows = st.slider("How many examples to show", 5, 30, 12, 1)
    with c3:
        highlight_level = st.select_slider("Highlight strength", ["Low", "Medium", "High"], value="Medium")
        k_each = get_highlight_k(highlight_level)

    view = df.copy()

    if focus == "Needs manual review (uncertain)":
        view = view[view["_uncertain"]]
    elif focus == "Mixed sentiment cases":
        view = view[view["_reasons"].apply(lambda xs: "Mixed feelings" in xs)]
    elif focus == "Negation cases":
        view = view[view["_reasons"].apply(lambda xs: "Negation (not/never)" in xs)]
    elif focus == "Very short reviews":
        view = view[view["_reasons"].apply(lambda xs: "Too short / low context" in xs)]
    elif focus == "Emphasis / tone cases":
        view = view[view["_reasons"].apply(lambda xs: "Emphasis / tone (caps/punct)" in xs)]

    view["_dist_to_mid"] = (view["proba_pos"] - 0.5).abs()
    view = view.sort_values(["_dist_to_mid"], ascending=True).head(max_rows)

    tmp = df.copy()
    if focus == "Needs manual review (uncertain)":
        tmp = tmp[tmp["_uncertain"]]
    elif focus == "Mixed sentiment cases":
        tmp = tmp[tmp["_reasons"].apply(lambda xs: "Mixed feelings" in xs)]
    elif focus == "Negation cases":
        tmp = tmp[tmp["_reasons"].apply(lambda xs: "Negation (not/never)" in xs)]
    elif focus == "Very short reviews":
        tmp = tmp[tmp["_reasons"].apply(lambda xs: "Too short / low context" in xs)]
    elif focus == "Emphasis / tone cases":
        tmp = tmp[tmp["_reasons"].apply(lambda xs: "Emphasis / tone (caps/punct)" in xs)]

    focus_total = len(tmp)

    total_all = len(df)
    showing = len(view)

    st.markdown(
        f"""
        <div class="rs-card" style="padding:10px 12px;">
          <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
            <span class="rs-chip">Viewing: <b>{focus_label}</b></span>
            <span class="rs-chip">Category size: <b>{focus_total:,}</b></span>
            <span class="rs-chip">Showing: <b>{showing:,}</b></span>
            <span class="rs-chip">Dataset: <b>{total_all:,}</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    for _, r in view.iterrows():
        txt = str(r[text_col])
        p = float(r["proba_pos"])
        bucket = r.get("bucket", bucketize(p))
        reasons = r.get("_reasons", ["Unclear / subtle wording"])

        pos_terms, neg_terms = explain_review_terms(model, txt, top_k_each=k_each)
        pos_terms = filter_terms(pos_terms)
        neg_terms = filter_terms(neg_terms)

        sent_info = sentence_scores(model, txt)
        best_pos = max(sent_info, key=lambda x: x["net"], default=None)
        best_neg = min(sent_info, key=lambda x: x["net"], default=None)

        reason_chips = "".join([f"<span class='rs-chip'>{rs}</span>" for rs in reasons])

        sentence_html = ""
        if best_pos and best_neg and best_pos["sentence"] != best_neg["sentence"]:
            sentence_html = (
                "<div class='rs-sub' style='margin-top:10px;'>"
                f"<b>{explain('Positive part:', 'Top positive sentence:')}</b> {best_pos['sentence']}<br/>"
                f"<b>{explain('Negative part:', 'Top negative sentence:')}</b> {best_neg['sentence']}"
                "</div>"
            )

        card_html = (
            "<div class='rs-card'>"
            f"<div class='rs-sub'>{explain('AI score', 'P(Positive)')}: <b>{p:.3f}</b> | {bucket}</div>"
            f"<div style='margin-top:6px'>{reason_chips}</div>"
            f"<div style='margin-top:10px'>{highlight_terms_both(txt, pos_terms, neg_terms)}</div>"
            f"{sentence_html}"
            "</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✅ How to use this safely")
    st.write(explain(
        "- If the review looks mixed (both green & red), treat it as **Mixed** and read it manually.\n"
        "- If the review is very short, the AI has less context — confidence can be low.\n"
        "- Negation (like “not bad”) can flip meaning; always double-check if the result feels wrong.",
        "- Mixed reviews: both positive and negative signals appear.\n"
        "- Short text reduces feature signal → less reliable.\n"
        "- Negation patterns can reverse polarity; review manually if near 0.5."
    ))


# PAGE: TRUST & RELIABILITY
elif st.session_state.page in ["Model Trust", "Trust & Reliability"]:
    st.subheader("🛡️ Trust Dashboard")
    st.caption("Executive view: how reliable the scores are and where risk is concentrated.")

    active_df, source_label, text_col = get_active_insights_df()
    df = active_df.copy()
    st.caption(f"Source: **{source_label}**")

    if df.empty:
        st.info("No data available yet. Upload a CSV/TSV/XLSX in Overview → Quick Analyze.")
        st.stop()

    if "proba_pos" not in df.columns:
        st.error("No probability scores found. Upload a file and run analysis first.")
        st.stop()

    if "confidence" not in df.columns:
        df["confidence"] = np.round(np.abs(df["proba_pos"] - 0.5) * 2, 4)

    st.markdown("### ✅ Score Zones (Safe vs Needs Review)")
    st.caption("See where reviews are safe to approve, need human review, or should be escalated.")

    zones = [
        ("Auto-escalate (Strong negative)", 0.00, 0.25),
        ("Likely negative",                0.25, 0.45),
        ("Needs review (Mixed/uncertain)", 0.45, 0.65),
        ("Likely positive",                0.65, 0.85),
        ("Auto-approve (Strong positive)", 0.85, 1.01),
    ]

    def zone_name(p: float) -> str:
        for name, lo, hi in zones:
            if lo <= p < hi:
                return name
        return "Needs review (Mixed/uncertain)"

    df["_zone"] = df["proba_pos"].apply(zone_name)
    zorder = [z[0] for z in zones]
    zcounts = df["_zone"].value_counts().reindex(zorder).fillna(0).astype(int)
    total = max(len(df), 1)

    if "trust_zone_focus" not in st.session_state:
        st.session_state.trust_zone_focus = None

    zone_ui = [
        ("Auto-approve", "Auto-approve (Strong positive)", "🟢", "Strong positive", "#35D07F"),
        ("Needs review", "Needs review (Mixed/uncertain)", "🟡", "Mixed / uncertain", "#FFC83D"),
        ("Auto-escalate", "Auto-escalate (Strong negative)", "🔴", "Strong negative", "#FF5572"),
        ("Likely positive", "Likely positive", "🔵", "Positive range", "#58A6FF"),
    ]

    cols = st.columns(4)
    for col, (label, key, emoji, subtitle, color) in zip(cols, zone_ui):
        count = int(zcounts.get(key, 0))
        pct = count / total * 100
        active = st.session_state.trust_zone_focus == key
        border = color if active else "rgba(255,255,255,.13)"
        glow = f"0 0 24px {color}33" if active else "0 8px 22px rgba(0,0,0,.16)"

        with col:
            card_html = f'''<div style="background:linear-gradient(145deg,rgba(255,255,255,.075),rgba(255,255,255,.025));border:1px solid {border};border-top:3px solid {color};border-radius:18px;padding:17px 16px 15px;min-height:178px;box-shadow:{glow};">
<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;"><div style="font-weight:800;color:#EEF1FF;font-size:14px;">{emoji} {label}</div><div style="font-size:10px;color:#8F98B8;">{pct:.1f}%</div></div>
<div style="margin-top:10px;color:{color};font-size:30px;font-weight:900;line-height:1;">{count:,}</div>
<div style="margin-top:4px;color:#8F98B8;font-size:11px;">{subtitle}</div>
<div style="height:8px;background:rgba(255,255,255,.09);border-radius:99px;overflow:hidden;margin-top:17px;"><div style="height:100%;width:{min(pct,100):.2f}%;background:linear-gradient(90deg,{color},{color}AA);border-radius:99px;box-shadow:0 0 10px {color}66;"></div></div>
<div style="display:flex;justify-content:space-between;margin-top:7px;color:#7F89AA;font-size:10px;"><span>Share of reviews</span><b style="color:#DCE1F4;">{pct:.1f}%</b></div>
</div>'''
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    button_specs = [
        (b1, "🔎 View Auto-approve", "Auto-approve (Strong positive)"),
        (b2, "🔎 View Needs review", "Needs review (Mixed/uncertain)"),
        (b3, "🔎 View Auto-escalate", "Auto-escalate (Strong negative)"),
    ]
    for col, label, key in button_specs:
        with col:
            if st.button(label, use_container_width=True, type="primary" if st.session_state.trust_zone_focus == key else "secondary"):
                st.session_state.trust_zone_focus = key
                st.rerun()
    with b4:
        if st.button("Clear selection", use_container_width=True):
            st.session_state.trust_zone_focus = None
            st.rerun()

    needs_review = int(zcounts["Needs review (Mixed/uncertain)"])
    auto_ok = int(zcounts["Auto-approve (Strong positive)"])
    auto_bad = int(zcounts["Auto-escalate (Strong negative)"])
    likely_pos = int(zcounts["Likely positive"])
    likely_neg = int(zcounts["Likely negative"])

    parts = [
        ("Auto-escalate", auto_bad, "#FF5572"),
        ("Likely negative", likely_neg, "#FF8B5C"),
        ("Needs review", needs_review, "#FFC83D"),
        ("Likely positive", likely_pos, "#58A6FF"),
        ("Auto-approve", auto_ok, "#35D07F"),
    ]
    bar_html = "".join(f'''<div title="{name}: {count:,} ({count/total*100:.1f}%)" style="width:{count/total*100:.3f}%;height:100%;background:{color};"></div>''' for name, count, color in parts if count > 0)
    legend_html = "".join(f'''<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;color:#AAB2CF;font-size:11px;"><i style="width:9px;height:9px;border-radius:3px;background:{color};display:inline-block;"></i>{name} <b style="color:#E8ECFA;">{count:,}</b></span>''' for name, count, color in parts)

    risk_html = f'''<div style="margin-top:18px;padding:16px 18px;border:1px solid rgba(255,255,255,.10);border-radius:18px;background:rgba(255,255,255,.035);">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:11px;"><span style="font-weight:800;color:#E9EDFF;font-size:13px;">Risk distribution</span><span style="color:#7F89AA;font-size:11px;">{total:,} reviews</span></div>
<div style="height:18px;width:100%;display:flex;overflow:hidden;border-radius:99px;background:rgba(255,255,255,.08);">{bar_html}</div>
<div style="margin-top:12px;line-height:2;">{legend_html}</div>
</div>'''
    st.markdown(risk_html, unsafe_allow_html=True)

    st.markdown(f'''<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;"><span class="rs-chip">Total: <b>{total:,}</b></span><span class="rs-chip">Auto-approve: <b>{auto_ok:,}</b></span><span class="rs-chip">Auto-escalate: <b>{auto_bad:,}</b></span><span class="rs-chip">Needs review: <b>{needs_review:,}</b></span></div>''', unsafe_allow_html=True)

    zone_focus = st.session_state.trust_zone_focus

    def tokenize_for_summary(text: str):
        toks = re.findall(r"[a-zA-Z']+", (text or "").lower())
        toks = [t for t in toks if len(t) >= 3]
        return toks

    SUMMARY_STOP = set([
        "the", "and", "for", "with", "this", "that", "have", "had", "was", "were", "are",
        "but", "not", "you", "your", "they", "them", "its", "it's", "very", "just",
        "good", "great", "nice", "bad", "book", "movie", "product", "one", "would",
        "also", "really", "much", "get", "like", "love", "time", "read"
    ])

    def one_sentence_summary(drill_df: pd.DataFrame, text_col: str) -> str:
        if drill_df.empty or text_col not in drill_df.columns:
            return "No reviews found for this segment."
        sample_text = " ".join(drill_df[text_col].astype(str).head(200).tolist())
        toks = [t for t in tokenize_for_summary(sample_text) if t not in SUMMARY_STOP]
        if not toks:
            return "Themes are diverse here — consider reviewing a few examples manually."
        top = [w for w, _ in Counter(toks).most_common(5)]
        return f"Common themes in this segment: {', '.join(top)}."

    if zone_focus:
        st.markdown("---")
        st.markdown(f"### 🔎 Segment Drilldown — {zone_focus}")

        if text_col is None or text_col not in df.columns:
            st.info("Text column not available for drilldown display.")
        else:
            drill = df[df["_zone"] == zone_focus].copy()

            d1, d2, d3 = st.columns([1.4, 1.0, 1.0])
            with d1:
                search_kw = st.text_input("Search within this segment (optional)")
            with d2:
                show_n = st.slider("Show reviews", 5, 50, 15, 1)
            with d3:
                highlight_level = st.select_slider("Highlight strength", ["Low", "Medium", "High"], value="Medium")
                k_each = get_highlight_k(highlight_level)

            if search_kw.strip():
                drill = drill[drill[text_col].astype(str).str.contains(search_kw, case=False, na=False)]

            st.markdown(
                f"""
                <div class="rs-card" style="padding:10px 12px;">
                  <div class="rs-sub"><b>Summary:</b> {one_sentence_summary(drill, text_col)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if zone_focus == "Auto-approve (Strong positive)":
                drill = drill.sort_values("proba_pos", ascending=False)
            elif zone_focus == "Auto-escalate (Strong negative)":
                drill = drill.sort_values("proba_pos", ascending=True)
            else:
                drill["_dist_to_mid"] = (drill["proba_pos"] - 0.5).abs()
                drill = drill.sort_values("_dist_to_mid", ascending=True)

            drill_show = drill.head(show_n)

            st.markdown(
                f"""
                <div class="rs-card" style="padding:10px 12px;">
                  <div style="display:flex; gap:10px; flex-wrap:wrap;">
                    <span class="rs-chip">Segment size: <b>{len(drill):,}</b></span>
                    <span class="rs-chip">Showing: <b>{len(drill_show):,}</b></span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            for _, r in drill_show.iterrows():
                txt = str(r.get(text_col, ""))
                p = float(r.get("proba_pos", 0.5))

                pos_terms, neg_terms = explain_review_terms(model, txt, top_k_each=k_each)
                pos_terms = filter_terms(pos_terms)
                neg_terms = filter_terms(neg_terms)

                st.markdown(
                    f"<div class='rs-card'>"
                    f"<div class='rs-sub'>Score: <b>{p:.3f}</b></div>"
                    f"<div style='margin-top:8px'>{highlight_terms_both(txt, pos_terms, neg_terms)}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            safe_name = zone_focus.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            st.download_button(
                "Download this segment (CSV)",
                drill.to_csv(index=False).encode("utf-8"),
                file_name=f"reviewsense_segment_{safe_name}.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")

    st.markdown("### 📊 Risk & Confidence (Executive)")

    risk_pct = float(((df["proba_pos"] < 0.45).mean()) * 100)
    needs_review_pct = float((df["proba_pos"].between(0.45, 0.65).mean()) * 100)
    low_conf_pct = float(((df["confidence"] < 0.35).mean()) * 100)

    st.markdown(
        f"""
        <div class="rs-card" style="padding:10px 12px;">
          <div style="display:flex; gap:10px; flex-wrap:wrap;">
            <span class="rs-chip">Negative risk: <b>{risk_pct:.1f}%</b></span>
            <span class="rs-chip">Needs review: <b>{needs_review_pct:.1f}%</b></span>
            <span class="rs-chip">Low confidence: <b>{low_conf_pct:.1f}%</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    def level_from_pct(p, low=20, high=40):
        if p < low:
            return ("LOW", "🟢")
        if p < high:
            return ("MEDIUM", "🟡")
        return ("HIGH", "🔴")

    risk_lvl, risk_emoji = level_from_pct(risk_pct, low=20, high=40)
    review_lvl, review_emoji = level_from_pct(needs_review_pct, low=20, high=35)
    conf_lvl, conf_emoji = level_from_pct(low_conf_pct, low=25, high=40)

    st.caption(
        explain(
            f"About {risk_pct:.0f}% of customers show negative sentiment. "
            f"{needs_review_pct:.0f}% of reviews are unclear and need human review. "
            f"The model is unsure about {low_conf_pct:.0f}% of cases.",
            f"{risk_pct:.1f}% negative sentiment, "
            f"{needs_review_pct:.1f}% manual-review zone, "
            f"{low_conf_pct:.1f}% low-confidence predictions."
        )
    )

    if risk_lvl == "HIGH" or conf_lvl == "HIGH":
        rec = explain(
            "Recommendation: treat this dataset as high-risk. Route critical cases to humans first.",
            "Recommendation: high risk or low confidence. Use human review for key decisions."
        )
    elif review_lvl == "HIGH":
        rec = explain(
            "Recommendation: many reviews are mixed. Use a human-in-the-loop workflow.",
            "Recommendation: large mixed zone. Apply manual review for borderline scores."
        )
    else:
        rec = explain(
            "Recommendation: safe to automate most workflows with spot-checking.",
            "Recommendation: acceptable risk. Automate with periodic QA audits."
        )

    st.markdown(
        f"""
        <div class="rs-card" style="padding:12px 14px;">
          <div style="display:flex; gap:14px; flex-wrap:wrap; align-items:center;">
            <span class="rs-chip">{risk_emoji} Operational Risk: <b>{risk_lvl}</b></span>
            <span class="rs-chip">{review_emoji} Human Review Load: <b>{review_lvl}</b></span>
            <span class="rs-chip">{conf_emoji} Model Confidence: <b>{conf_lvl}</b></span>
          </div>
          <div class="rs-sub" style="margin-top:8px;"><b>{rec}</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ✅ Recommended Usage Rules")
    st.markdown(
        """
        <div class="rs-card">
          <div class="rs-sub">
            🟢 <b>Auto-approve</b>: score ≥ 0.85<br/>
            🔴 <b>Auto-escalate</b>: score ≤ 0.25<br/>
            🟡 <b>Manual review</b>: score 0.45–0.65 or confidence &lt; 0.35
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Technical details (optional)"):
        if not df_compare.empty:
            st.dataframe(df_compare.head(10), use_container_width=True)₹
