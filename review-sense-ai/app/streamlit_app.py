
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import io
from pathlib import Path
import plotly.express as px
from collections import Counter

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model_calibrated.joblib"
REPORTS_PATH = PROJECT_ROOT / "outputs" / "reports"


# PAGE CONFIG
st.set_page_config(page_title="ReviewSense", page_icon="💬", layout="wide")


# STYLE
st.markdown(
    """
    <style>
      .rs-card {
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
      }
      .rs-kpi {
        font-size: 28px;
        font-weight: 750;
        line-height: 1.05;
        margin-top: 6px;
      }
      .rs-sub {
        font-size: 12px;
        opacity: 0.78;
        margin-top: 3px;
      }
      .rs-title {
        font-size: 14px;
        font-weight: 700;
        opacity: 0.95;
      }
      .rs-chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        font-size: 12px;
        margin-right: 6px;
        margin-bottom: 6px;
      }
      .hl{
        display:inline-block;
        padding: 1px 6px;         /* smaller padding */
        border-radius: 8px;
        line-height: 1.4;
        margin: 0 1px;
        font-weight: 600;
        color: #ffffff !important; /* readable in dark theme */
        border: 1px solid rgba(255,255,255,0.18);
      }
      .hl-pos{
        background: rgba(34,197,94,0.55);  /* green, not too bright */
      }
      .hl-neg{
        background: rgba(239,68,68,0.55);  /* red, not too bright */
      }
    
    </style>
    """,
    unsafe_allow_html=True
)


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
st.title("💬 ReviewSense")
st.caption("A customer-ready review intelligence dashboard")

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
def donut_bucket_distribution(df, bucket_col="bucket"):
    if df.empty or bucket_col not in df.columns:
        return px.pie(pd.DataFrame({"Bucket": [], "Count": []}), names="Bucket", values="Count", hole=0.68, height=300)
    counts = df[bucket_col].value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int).reset_index()
    counts.columns = ["Bucket", "Count"]
    fig = px.pie(counts, names="Bucket", values="Count", hole=0.68, height=300)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
    return fig

def bar_bucket_distribution(df, bucket_col="bucket"):
    if df.empty or bucket_col not in df.columns:
        return px.bar(pd.DataFrame({"Bucket": [], "Count": []}), x="Bucket", y="Count", height=270)
    counts = df[bucket_col].value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
    chart_df = pd.DataFrame({"Bucket": counts.index, "Count": counts.values})
    fig = px.bar(chart_df, x="Bucket", y="Count", height=270)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title=None, yaxis_title=None)
    return fig

def confidence_hist(df, proba_col="proba_pos"):
    if df.empty or proba_col not in df.columns:
        return px.histogram(pd.DataFrame({proba_col: []}), x=proba_col, nbins=30, height=270)
    fig = px.histogram(df, x=proba_col, nbins=30, height=270)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="P(Positive)", yaxis_title=None)
    return fig

def kw_bar(kws, height=320, title=None):
    if not kws:
        return None
    dfk = pd.DataFrame(kws, columns=["Keyword", "Mentions"])
    fig = px.bar(dfk, x="Mentions", y="Keyword", orientation="h", height=height)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title=None, xaxis_title=None, title=title)
    return fig

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
st.sidebar.title("🧭 Navigation")
st.session_state.page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.page))

# PAGE: OVERVIEW
if st.session_state.page == "Overview":
    st.subheader("📊 Overview")

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
            f"<div class='rs-card'><div class='rs-title'>Total Reviews</div>"
            f"<div class='rs-kpi'>{total:,}</div>"
            f"<div class='rs-sub'>Baseline evaluation set</div></div>",
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"<div class='rs-card'><div class='rs-title'>Positive Rate</div>"
            f"<div class='rs-kpi'>{pos_rate*100:.1f}%</div>"
            f"<div class='rs-sub'>From true labels</div></div>",
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"<div class='rs-card'><div class='rs-title'>Negative Rate</div>"
            f"<div class='rs-kpi'>{neg_rate*100:.1f}%</div>"
            f"<div class='rs-sub'>From true labels</div></div>",
            unsafe_allow_html=True
        )
    with k4:
        label = explain("AI Reliability", "Best F1 Score")
        val = "High" if st.session_state.mode == "Simple Language" else (f"{best_f1:.4f}" if best_f1 is not None else "N/A")
        sub = explain("Consistency of predictions", "Top-ranked model")
        st.markdown(
            f"<div class='rs-card'><div class='rs-title'>{label}</div>"
            f"<div class='rs-kpi'>{val}</div>"
            f"<div class='rs-sub'>{sub}</div></div>",
            unsafe_allow_html=True
        )

    st.caption("Scroll down for ⚡ Quick Analyze (type a review or upload a file).")

    st.markdown("### Sentiment Breakdown (5 Levels)")
    st.caption(explain(
        "Click any category to explore examples and understand what drives it.",
        "Buckets are based on calibrated probability ranges."
    ))

    if not df_errors.empty and "bucket" in df_errors.columns:
        bucket_counts = df_errors["bucket"].value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
    else:
        bucket_counts = pd.Series({b: 0 for b in BUCKET_ORDER})

    card_cols = st.columns(5)
    for i, bucket_name in enumerate(BUCKET_ORDER):
        with card_cols[i]:
            count = int(bucket_counts.get(bucket_name, 0))
            pct = (count / total) * 100 if total else 0
            if st.button(f"{bucket_name}\n\n{count:,} ({pct:.1f}%)", use_container_width=True, disabled=(total == 0)):
                st.session_state.selected_bucket = bucket_name
                st.session_state.page = "Category Details"
                st.rerun()

    c1, c2, c3 = st.columns([1.1, 1.2, 1.2])

    with c1:
        st.markdown("<div class='rs-card'><div class='rs-title'>Distribution</div></div>", unsafe_allow_html=True)
        st.plotly_chart(donut_bucket_distribution(df_errors), use_container_width=True)

    with c2:
        st.markdown("<div class='rs-card'><div class='rs-title'>Counts by Category</div></div>", unsafe_allow_html=True)
        st.plotly_chart(bar_bucket_distribution(df_errors), use_container_width=True)

    with c3:
        st.markdown("<div class='rs-card'><div class='rs-title'>Confidence Spread</div></div>", unsafe_allow_html=True)
        st.plotly_chart(confidence_hist(df_errors), use_container_width=True)

    # Quick Analyze
    st.markdown("---")
    st.markdown("### ⚡ Quick Analyze (Type anything or Upload anything)")
    st.caption(explain(
        "Analyze a single review or upload a file to get instant insights.",
        "Supports CSV / TSV / XLSX with automatic column detection."
    ))

    qa1, qa2 = st.columns([1.2, 1])

    with qa1:
        review_text = st.text_area("Single review", height=120, placeholder="Paste any customer review here…")

    with qa2:
        uploaded = st.file_uploader("Upload file (CSV / TSV / XLSX)", type=["csv", "tsv", "xlsx"])
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
    st.subheader("📦 Batch Results (Uploaded File)")

    out = st.session_state.batch_df
    if out is None or len(out) == 0:
        st.info("No batch results yet. Go to Overview → upload a file → Run Analysis.")
        st.stop()

    text_col_used = safe_textcol_from_batch(out)
    if text_col_used is None or text_col_used not in out.columns:
        st.error("Text column missing in batch results. Upload again and confirm mapping.")
        st.stop()

    total = len(out)
    pct_pos = float((out["bucket"].isin(["Strongly Positive 😍", "Positive 🙂"])).mean()) * 100
    pct_neg = float((out["bucket"].isin(["Strongly Negative 😡", "Negative 🙁"])).mean()) * 100
    pct_mix = float((out["bucket"] == "Mixed 😐").mean()) * 100

    avg_conf = float(out["confidence"].mean())
    high_risk = out[(out["bucket"] == "Strongly Negative 😡") & (out["confidence"] >= 0.7)]
    high_praise = out[(out["bucket"] == "Strongly Positive 😍") & (out["confidence"] >= 0.7)]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='rs-card'><div class='rs-title'>Rows analyzed</div>"
            f"<div class='rs-kpi'>{total:,}</div><div class='rs-sub'>Uploaded dataset</div></div>",
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"<div class='rs-card'><div class='rs-title'>Positive</div>"
            f"<div class='rs-kpi'>{pct_pos:.1f}%</div><div class='rs-sub'>Satisfied customers</div></div>",
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"<div class='rs-card'><div class='rs-title'>Negative</div>"
            f"<div class='rs-kpi'>{pct_neg:.1f}%</div><div class='rs-sub'>Customer pain points</div></div>",
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"<div class='rs-card'><div class='rs-title'>{explain('Avg confidence', 'Avg confidence (0..1)')}</div>"
            f"<div class='rs-kpi'>{avg_conf:.2f}</div><div class='rs-sub'>Higher = more reliable</div></div>",
            unsafe_allow_html=True
        )

    st.success(
        explain(
            f"Summary: {pct_pos:.1f}% positive, {pct_neg:.1f}% negative, {pct_mix:.1f}% mixed. "
            f"High-risk unhappy reviews: {len(high_risk):,}.",
            f"pos={pct_pos:.1f}% | neg={pct_neg:.1f}% | mixed={pct_mix:.1f}% | high_risk={len(high_risk)}"
        )
    )

    st.markdown("---")

    neg_texts = out[out["bucket"].isin(["Strongly Negative 😡", "Negative 🙁"])][text_col_used].astype(str).tolist()
    pos_texts = out[out["bucket"].isin(["Strongly Positive 😍", "Positive 🙂"])][text_col_used].astype(str).tolist()

    top_pos_phrases, _ = extract_top_phrases_from_group(model, pos_texts, top_n=10)
    _, top_neg_phrases = extract_top_phrases_from_group(model, neg_texts, top_n=10)

    cL, cR = st.columns(2)
    with cL:
        st.markdown("### ✅ Top Wins (what customers like)")
        if top_pos_phrases:
            chips = "".join([f"<span class='rs-chip'>{p}</span>" for p in top_pos_phrases[:8]])
            st.markdown(f"<div class='rs-card'><div class='rs-sub'>Top praise themes</div>{chips}</div>", unsafe_allow_html=True)
        else:
            st.info("Not enough strong positive phrase signals.")

        st.markdown("#### Example praise (high confidence)")
        ex = high_praise.head(2)
        for _, r in ex.iterrows():
            txt = str(r[text_col_used])
            p = float(r["proba_pos"])
            pos_terms, neg_terms = explain_review_terms(model, txt, top_k_each=7)
            pos_terms = filter_terms(pos_terms)
            neg_terms = filter_terms(neg_terms)
            st.markdown(
                f"<div class='rs-card'><div class='rs-sub'>AI score: <b>{p:.3f}</b></div>"
                f"<div style='margin-top:8px'>{highlight_terms_both(txt, pos_terms, neg_terms)}</div></div>",
                unsafe_allow_html=True
            )

    with cR:
        st.markdown("### ⚠️ Top Issues (what to fix first)")
        if top_neg_phrases:
            chips = "".join([f"<span class='rs-chip'>{p}</span>" for p in top_neg_phrases[:8]])
            st.markdown(f"<div class='rs-card'><div class='rs-sub'>Top complaint themes</div>{chips}</div>", unsafe_allow_html=True)
        else:
            st.info("Not enough strong negative phrase signals.")

        st.markdown("#### Example complaint (high confidence)")
        ex = high_risk.head(2)
        for _, r in ex.iterrows():
            txt = str(r[text_col_used])
            p = float(r["proba_pos"])
            pos_terms, neg_terms = explain_review_terms(model, txt, top_k_each=7)
            pos_terms = filter_terms(pos_terms)
            neg_terms = filter_terms(neg_terms)
            st.markdown(
                f"<div class='rs-card'><div class='rs-sub'>AI score: <b>{p:.3f}</b></div>"
                f"<div style='margin-top:8px'>{highlight_terms_both(txt, pos_terms, neg_terms)}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    c1, c2, c3 = st.columns([1.1, 1.2, 1.2])
    with c1:
        st.markdown("<div class='rs-card'><div class='rs-title'>Distribution (5 levels)</div></div>", unsafe_allow_html=True)
        st.plotly_chart(donut_bucket_distribution(out, bucket_col="bucket"), use_container_width=True)
    with c2:
        st.markdown("<div class='rs-card'><div class='rs-title'>Counts by Category</div></div>", unsafe_allow_html=True)
        st.plotly_chart(bar_bucket_distribution(out, bucket_col="bucket"), use_container_width=True)
    with c3:
        st.markdown("<div class='rs-card'><div class='rs-title'>Confidence Spread</div></div>", unsafe_allow_html=True)
        st.plotly_chart(confidence_hist(out, proba_col="proba_pos"), use_container_width=True)

    st.markdown("---")

    st.markdown("## 🔎 Drilldown by Category")
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
        f"<div class='rs-sub'>{len(view):,} rows match your filters</div></div>",
        unsafe_allow_html=True
    )

    if pick_bucket != "All" and len(view) > 0:
        bucket_texts = view[text_col_used].astype(str).tolist()
        pos_phr, neg_phr = extract_top_phrases_from_group(model, bucket_texts, top_n=10)

        st.markdown("### 📌 Key Themes in this Category")
        chips = "".join([f"<span class='rs-chip'>{p}</span>" for p in (neg_phr[:5] + pos_phr[:5])])
        st.markdown(f"<div class='rs-card'>{chips}</div>", unsafe_allow_html=True)

        st.markdown("### 🧾 Example Reviews (with highlights)")
        examples = view.sort_values("confidence", ascending=False).head(6)
        for _, r in examples.iterrows():
            txt = str(r[text_col_used])
            p = float(r["proba_pos"])
            pos_terms, neg_terms = explain_review_terms(model, txt, top_k_each=7)
            pos_terms = filter_terms(pos_terms)
            neg_terms = filter_terms(neg_terms)

            st.markdown(
                f"<div class='rs-card'>"
                f"<div class='rs-sub'>AI score: <b>{p:.3f}</b> | {r['bucket']}</div>"
                f"<div style='margin-top:8px'>{highlight_terms_both(txt, pos_terms, neg_terms)}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.markdown("## 📋 Results Table (download-ready)")
    st.dataframe(view.head(200), use_container_width=True)

    st.download_button(
        "Download results (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        file_name="reviewsense_results.csv",
        mime="text/csv",
        use_container_width=True
    )


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
    with k1:
        st.markdown(f"<div class='rs-card'><div class='rs-title'>Positive</div><div class='rs-kpi'>{pct_pos:.1f}%</div><div class='rs-sub'>Satisfied customers</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='rs-card'><div class='rs-title'>Negative</div><div class='rs-kpi'>{pct_neg:.1f}%</div><div class='rs-sub'>Pain points present</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='rs-card'><div class='rs-title'>Mixed</div><div class='rs-kpi'>{pct_mix:.1f}%</div><div class='rs-sub'>Pros + cons</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='rs-card'><div class='rs-title'>Focus Index</div><div class='rs-kpi'>{focus_index:.1f}</div><div class='rs-sub'>Higher = fix priority</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1.15, 1.0])

    with left:
        st.markdown("### ✅ What Customers Appreciate")
        if not top_pos_phrases:
            st.info("Not enough signal to extract strong positive themes.")
        else:
            chips = "".join([f"<span class='rs-chip'>{p}</span>" for p in top_pos_phrases[:8]])
            st.markdown(f"<div class='rs-card'><div class='rs-sub'>Top praise themes</div>{chips}</div>", unsafe_allow_html=True)

        st.markdown("### ⚠️ What’s Driving Complaints")
        if not top_neg_phrases:
            st.info("Not enough signal to extract strong complaint themes.")
        else:
            chips = "".join([f"<span class='rs-chip'>{p}</span>" for p in top_neg_phrases[:8]])
            st.markdown(f"<div class='rs-card'><div class='rs-sub'>Top complaint themes</div>{chips}</div>", unsafe_allow_html=True)

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
    st.caption("These categories explain *why* a review may be hard for AI to judge accurately.")

    if "tricky_focus" not in st.session_state:
        st.session_state.tricky_focus = "All tricky reviews"

    def focus_button(label, key):
        active = (st.session_state.tricky_focus == key)
        if st.button(
            f"{label}\n\n{counts.get(key, 0):,}",
            use_container_width=True,
            type=("primary" if active else "secondary")
        ):
            st.session_state.tricky_focus = key
            st.rerun()

    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        focus_button("Needs Human Review", "Needs manual review (uncertain)")
    with b2:
        focus_button("Mixed Feelings", "Mixed sentiment cases")
    with b3:
        focus_button("Confusing Wording", "Negation cases")
    with b4:
        focus_button("Too Little Detail", "Very short reviews")
    with b5:
        focus_button("Strong Tone / Emphasis", "Emphasis / tone cases")

    if st.button("Show All", use_container_width=True):
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
        "Category": ["Needs Human Review", "Mixed Feelings", "Confusing Wording", "Too Little Detail", "Strong Tone / Emphasis"],
        "Count": [
            counts["Needs manual review (uncertain)"],
            counts["Mixed sentiment cases"],
            counts["Negation cases"],
            counts["Very short reviews"],
            counts["Emphasis / tone cases"],
        ]
    })
    fig = px.bar(chart_df, x="Category", y="Count", height=260)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

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
    st.caption("Click a segment to view the exact reviews behind that number.")

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

    total = len(df)
    zdf = pd.DataFrame({"Zone": zcounts.index, "Count": zcounts.values})

    if "trust_zone_focus" not in st.session_state:
        st.session_state.trust_zone_focus = None

    seg1, seg2, seg3, seg4 = st.columns([1, 1, 1, 1])

    def zone_btn(label: str, zone_key: str, count: int, col):
        active = (st.session_state.trust_zone_focus == zone_key)
        with col:
            if st.button(
                f"{label}: {count:,}",
                use_container_width=True,
                type=("primary" if active else "secondary"),
            ):
                st.session_state.trust_zone_focus = zone_key
                st.rerun()

    zone_btn("Auto-approve", "Auto-approve (Strong positive)", int(zcounts["Auto-approve (Strong positive)"]), seg1)
    zone_btn("Needs review", "Needs review (Mixed/uncertain)", int(zcounts["Needs review (Mixed/uncertain)"]), seg2)
    zone_btn("Auto-escalate", "Auto-escalate (Strong negative)", int(zcounts["Auto-escalate (Strong negative)"]), seg3)

    with seg4:
        if st.button("Clear", use_container_width=True):
            st.session_state.trust_zone_focus = None
            st.rerun()

    needs_review = int(zcounts["Needs review (Mixed/uncertain)"])
    auto_ok = int(zcounts["Auto-approve (Strong positive)"])
    auto_bad = int(zcounts["Auto-escalate (Strong negative)"])

    st.markdown(
        f"""
        <div class="rs-card" style="padding:10px 12px;">
          <div style="display:flex; gap:10px; flex-wrap:wrap;">
            <span class="rs-chip">Total: <b>{total:,}</b></span>
            <span class="rs-chip">Auto-approve: <b>{auto_ok:,}</b></span>
            <span class="rs-chip">Auto-escalate: <b>{auto_bad:,}</b></span>
            <span class="rs-chip">Needs review: <b>{needs_review:,}</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    fig_zone = px.pie(zdf, names="Zone", values="Count", hole=0.65, height=320)
    fig_zone.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
    st.plotly_chart(fig_zone, use_container_width=True)

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
            st.dataframe(df_compare.head(10), use_container_width=True)
