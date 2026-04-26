"""
🎨 AI TEXT DETECTOR — DOODLE CARTOON THEME
==========================================
Fixes applied vs previous version:
- Empty label warnings eliminated (all sliders/radios have real labels)
- Threshold slider actually changes verdict
- PDF download button present after PDF upload analysis
- CSV/JSON export removed — only PDF report
- Windowed sentence-level prediction (not isolated sentences)
- inf/NaN guard before scaling
- reindex with fill_value=0 for column safety
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import ast
import spacy
import fitz
from scipy.sparse import hstack
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors as rl_colors
import plotly.graph_objects as go
import json
import io
from datetime import datetime
from huggingface_hub import hf_hub_download
import os

REPO_ID = "bushraasaleem/ai-detector-models"

def download_models():
    files = ["best_model.pkl", "tfidf_vectorizer.pkl", 
             "feature_scaler.pkl", "feature_columns.pkl"]
    for fname in files:
        if not os.path.exists(fname):
            print(f"Downloading {fname} from HuggingFace...")
            hf_hub_download(
                repo_id=REPO_ID, 
                filename=fname, 
                local_dir="."
            )

download_models()  # ← this runs before anything else

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 AI Sniff-O-Meter",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Doodle / Cartoon Theme ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Architects+Daughter&family=Nunito:wght@400;600;700;800;900&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #FFF9F0 !important;
    color: #2D2D2D !important;
    font-family: 'Nunito', sans-serif !important;
}

/* Doodle paper texture background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        repeating-linear-gradient(0deg, transparent, transparent 27px, rgba(180,180,180,0.15) 27px, rgba(180,180,180,0.15) 28px),
        repeating-linear-gradient(90deg, transparent, transparent 27px, rgba(180,180,180,0.08) 27px, rgba(180,180,180,0.08) 28px);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: #FFF3E0 !important;
    border-right: 3px solid #2D2D2D !important;
}

[data-testid="stHeader"] { background: transparent !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── All text visible ── */
p, span, div, li, label {
    color: #2D2D2D !important;
    font-family: 'Nunito', sans-serif !important;
}
h1, h2, h3, h4 {
    color: #1A1A1A !important;
    font-family: 'Architects Daughter', cursive !important;
}

/* ── Doodle card base ── */
.doodle-card {
    background: #FFFFFF;
    border: 3px solid #2D2D2D;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin: 0.8rem 0;
    box-shadow: 5px 5px 0px #2D2D2D;
    position: relative;
}
.doodle-card:hover {
    transform: translate(-2px, -2px);
    box-shadow: 7px 7px 0px #2D2D2D;
    transition: all 0.15s ease;
}

/* ── Masthead ── */
.masthead {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.masthead-emoji {
    font-size: 4rem;
    display: block;
    margin-bottom: 0.3rem;
    animation: wobble 2.5s ease-in-out infinite;
}
@keyframes wobble {
    0%, 100% { transform: rotate(-3deg); }
    50%       { transform: rotate(3deg); }
}
.masthead-title {
    font-family: 'Architects Daughter', cursive !important;
    font-size: 3rem;
    color: #1A1A1A !important;
    margin: 0;
    line-height: 1.1;
    text-shadow: 3px 3px 0px #FFD93D;
}
.masthead-sub {
    font-size: 1rem;
    color: #666 !important;
    margin-top: 0.4rem;
    font-weight: 600;
}

/* ── Section header ── */
.section-head {
    font-family: 'Architects Daughter', cursive !important;
    font-size: 1.3rem;
    color: #1A1A1A !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1.5rem 0 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px dashed #2D2D2D;
}

/* ── Verdict banner ── */
.verdict-wrap {
    border: 3px solid #2D2D2D;
    border-radius: 20px;
    padding: 1.6rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin: 1rem 0;
    box-shadow: 6px 6px 0 #2D2D2D;
    position: relative;
    overflow: hidden;
}
.verdict-ai     { background: #FFE5E5; }
.verdict-human  { background: #E8FFE8; }
.verdict-mixed  { background: #FFFBE5; }

.verdict-icon { font-size: 3.5rem; flex-shrink: 0; }
.verdict-tag {
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 0.1rem;
}
.verdict-title {
    font-family: 'Architects Daughter', cursive !important;
    font-size: 2rem !important;
    font-weight: 900;
    margin: 0;
    line-height: 1.1;
}
.verdict-prob {
    font-size: 0.9rem;
    font-weight: 700;
    opacity: 0.75;
    margin-top: 0.2rem;
}
.verdict-score {
    margin-left: auto;
    text-align: center;
    background: #2D2D2D;
    color: #FFF9F0 !important;
    border-radius: 14px;
    padding: 0.8rem 1.4rem;
    flex-shrink: 0;
}
.verdict-score-num {
    font-family: 'Architects Daughter', cursive;
    font-size: 2.2rem;
    color: #FFD93D !important;
    line-height: 1;
}
.verdict-score-label {
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #aaa !important;
    margin-top: 0.2rem;
}

/* ── Metric chips ── */
.chips-row {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
    margin: 0.8rem 0;
}
.chip {
    background: #fff;
    border: 2.5px solid #2D2D2D;
    border-radius: 12px;
    padding: 0.65rem 1rem;
    flex: 1;
    min-width: 100px;
    text-align: center;
    box-shadow: 3px 3px 0 #2D2D2D;
}
.chip-val {
    font-family: 'Architects Daughter', cursive;
    font-size: 1.5rem;
    font-weight: 900;
    line-height: 1.1;
}
.chip-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #666 !important;
    margin-top: 0.15rem;
}

/* ── Sentence annotation ── */
.sent-annotated { line-height: 2.4; font-size: 1rem; }
.sent-token {
    display: inline;
    border-radius: 4px;
    padding: 2px 1px;
    cursor: default;
}
.sent-ai       { background: #FFB3B3; border-bottom: 3px solid #E53E3E; }
.sent-human    { background: #B3F0B3; border-bottom: 3px solid #276749; }
.sent-uncertain{ background: #FFE9A0; border-bottom: 3px solid #D69E2E; }

/* legend */
.legend {
    display: flex;
    gap: 1.2rem;
    font-size: 0.8rem;
    font-weight: 700;
    margin: 0.5rem 0 0.8rem;
    flex-wrap: wrap;
}
.legend-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 2px solid #2D2D2D;
    margin-right: 4px;
    vertical-align: middle;
}
.dot-ai       { background: #FFB3B3; }
.dot-human    { background: #B3F0B3; }
.dot-uncertain{ background: #FFE9A0; }

/* ── Sentence cards ── */
.sent-card {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    background: #fff;
    border: 2.5px solid #2D2D2D;
    border-radius: 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    box-shadow: 3px 3px 0 #2D2D2D;
    font-size: 0.9rem;
}
.sent-card-ai       { border-left: 5px solid #E53E3E; }
.sent-card-human    { border-left: 5px solid #276749; }
.sent-card-uncertain{ border-left: 5px solid #D69E2E; }

.sent-badge {
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.2rem 0.45rem;
    border-radius: 6px;
    white-space: nowrap;
    flex-shrink: 0;
    border: 2px solid #2D2D2D;
}
.badge-ai       { background: #FFB3B3; color: #7B1E1E !important; }
.badge-human    { background: #B3F0B3; color: #1A4731 !important; }
.badge-uncertain{ background: #FFE9A0; color: #744210 !important; }

.sent-text { flex: 1; line-height: 1.6; color: #2D2D2D !important; }
.sent-pct  { font-weight: 800; font-size: 0.78rem; white-space: nowrap; color: #555 !important; }

/* ── Feature bars ── */
.feat-row { margin: 0.55rem 0; }
.feat-top { display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 700; margin-bottom: 0.2rem; }
.feat-bg { background: #EEE; border: 2px solid #2D2D2D; border-radius: 6px; height: 10px; overflow: hidden; }
.feat-fill { height: 100%; border-radius: 4px; }

/* ── Buttons ── */
.stButton > button {
    background: #FFD93D !important;
    color: #1A1A1A !important;
    border: 3px solid #2D2D2D !important;
    font-family: 'Architects Daughter', cursive !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 12px !important;
    box-shadow: 4px 4px 0 #2D2D2D !important;
    transition: all 0.12s ease !important;
}
.stButton > button:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: 6px 6px 0 #2D2D2D !important;
    background: #FFC107 !important;
}
.stButton > button:active {
    transform: translate(2px, 2px) !important;
    box-shadow: 2px 2px 0 #2D2D2D !important;
}

/* Download button variant */
[data-testid="stDownloadButton"] > button {
    background: #6BCB77 !important;
    color: #1A1A1A !important;
    border: 3px solid #2D2D2D !important;
    font-family: 'Architects Daughter', cursive !important;
    font-size: 0.9rem !important;
    border-radius: 12px !important;
    box-shadow: 4px 4px 0 #2D2D2D !important;
    transition: all 0.12s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    transform: translate(-2px,-2px) !important;
    box-shadow: 6px 6px 0 #2D2D2D !important;
}

/* ── Textarea ── */
.stTextArea textarea {
    background: #fff !important;
    border: 2.5px solid #2D2D2D !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.9rem !important;
    color: #2D2D2D !important;
    padding: 0.8rem !important;
    box-shadow: 3px 3px 0 #2D2D2D !important;
}
.stTextArea textarea:focus {
    border-color: #FF6B6B !important;
    box-shadow: 3px 3px 0 #FF6B6B !important;
    outline: none !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    background: #fff !important;
    border: 2.5px dashed #2D2D2D !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* ── Radio ── */
[data-testid="stRadio"] > label {
    font-family: 'Architects Daughter', cursive !important;
    font-size: 1rem !important;
    color: #1A1A1A !important;
    font-weight: 700 !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label {
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #2D2D2D !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > label {
    font-family: 'Architects Daughter', cursive !important;
    font-size: 0.95rem !important;
    color: #1A1A1A !important;
    font-weight: 700 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #FFD93D !important;
    border: 2px solid #2D2D2D !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.5rem !important;
    border-bottom: 2px solid #2D2D2D !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Architects Daughter', cursive !important;
    font-size: 0.9rem !important;
    color: #666 !important;
    background: #fff !important;
    border: 2px solid #2D2D2D !important;
    border-bottom: none !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 0.4rem 1rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #FFD93D !important;
    color: #1A1A1A !important;
    font-weight: 800 !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: #EEE !important;
    border: 2px solid #2D2D2D !important;
    border-radius: 6px !important;
}
[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #FF6B6B, #FFD93D) !important;
    border-radius: 4px !important;
}

/* ── Success / error ── */
[data-testid="stAlert"] {
    border: 2.5px solid #2D2D2D !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    font-family: 'Nunito', sans-serif !important;
    color: #2D2D2D !important;
    font-weight: 600 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #FFF9F0; }
::-webkit-scrollbar-thumb { background: #FFD93D; border-radius: 3px; border: 2px solid #2D2D2D; }

/* ── Animations ── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.slide-up   { animation: slideUp 0.4s ease forwards; }
.slide-up-1 { animation: slideUp 0.4s 0.1s ease both; }
.slide-up-2 { animation: slideUp 0.4s 0.2s ease both; }
.slide-up-3 { animation: slideUp 0.4s 0.3s ease both; }

@keyframes bounce-dot {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
    40%            { transform: scale(1.2); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_WORDS    = 20
WINDOW_SIZE  = 3

TRANSITION_WORDS = {
    'furthermore','moreover','additionally','consequently','therefore',
    'however','nevertheless','firstly','secondly','thirdly','finally',
    'in conclusion','to summarize','in summary','to conclude','in addition',
    'as a result','on the other hand','that being said','it is worth noting'
}
FORMAL_STARTS = {
    'the','this','these','it','in','as','furthermore','moreover',
    'additionally','there','one','when','while'
}

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_spacy():
    try:
        nlp = spacy.load('en_core_web_sm', disable=['ner','parser'])
        nlp.add_pipe('sentencizer')
        return nlp
    except OSError:
        st.error("spaCy model missing. Run: python -m spacy download en_core_web_sm")
        return None

@st.cache_resource
def load_model():
    try:
        return {
            'model':           joblib.load('best_model.pkl'),
            'tfidf':           joblib.load('tfidf_vectorizer.pkl'),
            'scaler':          joblib.load('feature_scaler.pkl'),
            'feature_columns': joblib.load('feature_columns.pkl'),
        }
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None

@st.cache_resource
def load_word_lists():
    out = {}
    for key, fname in [('chat_words','chat_words.txt'),
                        ('function_words','function_words.txt'),
                        ('discourse_markers','discourse_markers.txt')]:
        try:
            with open(fname, encoding='utf-8') as f:
                out[key] = {l.strip().lower() for l in f if l.strip()}
        except FileNotFoundError:
            out[key] = set()
    return out

# ── Feature extraction (exact pipeline match) ─────────────────────────────────
def clean_text(x):
    if isinstance(x, list): return ' '.join(map(str, x))
    if isinstance(x, dict): return ' '.join(map(str, x.values()))
    if isinstance(x, str):
        try:
            p = ast.literal_eval(x)
            return ' '.join(map(str, p)) if isinstance(p, list) else str(p)
        except (ValueError, SyntaxError):
            return x
    return str(x)

def split_sentences(text, nlp):
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]

def extract_features(text, nlp, wl):
    doc          = nlp(text)
    tokens       = [t.text.lower() for t in doc if not t.is_space]
    alpha_tokens = [t.text.lower() for t in doc if t.is_alpha]
    tt, ta       = len(tokens), len(alpha_tokens)
    if tt == 0 or ta == 0:
        return {k: 0.0 for k in ['chat_word_ratio','punct_ratio','ttr',
                                   'function_word_ratio','discourse_ratio','sentence_length']}
    return {
        'chat_word_ratio':     sum(1 for t in tokens if t in wl['chat_words'])        / tt,
        'punct_ratio':         sum(1 for t in doc if t.is_punct)                      / tt,
        'ttr':                 len(set(alpha_tokens))                                 / ta,
        'function_word_ratio': sum(1 for t in tokens if t in wl['function_words'])    / tt,
        'discourse_ratio':     sum(1 for t in tokens if t in wl['discourse_markers']) / tt,
        'sentence_length':     float(tt),
    }

def aggregate_features(sentences, nlp, wl):
    feats    = [extract_features(s, nlp, wl) for s in sentences]
    if not feats:
        return {k: 0.0 for k in [
            'chat_word_ratio','punct_ratio','ttr','function_word_ratio','discourse_ratio',
            'sentence_length_std','sentence_length_cv','contraction_ratio',
            'transition_ratio','formal_start_ratio']}
    df       = pd.DataFrame(feats)
    lengths  = df['sentence_length'].tolist()
    mean_len = np.mean(lengths) if lengths else 1.0
    std_len  = np.std(lengths)  if len(lengths) > 1 else 0.0
    full     = ' '.join(sentences)
    toks     = full.split()
    n        = max(len(toks), 1)
    contracs = sum(1 for t in toks if "'" in t and
                   any(t.endswith(s) for s in ["'t","'re","'ve","'ll","'d","'m","'s"]))
    fl       = full.lower()
    trans    = sum(fl.count(tw) for tw in TRANSITION_WORDS)
    formal   = sum(1 for s in sentences
                   if s.strip() and s.strip().split()[0].lower() in FORMAL_STARTS)
    return {
        'chat_word_ratio':     df['chat_word_ratio'].mean(),
        'punct_ratio':         df['punct_ratio'].mean(),
        'ttr':                 df['ttr'].mean(),
        'function_word_ratio': df['function_word_ratio'].mean(),
        'discourse_ratio':     df['discourse_ratio'].mean(),
        'sentence_length_std': std_len,
        'sentence_length_cv':  std_len / mean_len if mean_len > 0 else 0.0,
        'contraction_ratio':   contracs / n,
        'transition_ratio':    trans    / n,
        'formal_start_ratio':  formal   / max(len(sentences), 1),
    }

def _proba_for_window(window_text, mc, nlp, wl):
    """Document-level prediction on a text window — matches training exactly."""
    sents  = split_sentences(window_text, nlp)
    feats  = aggregate_features(sents, nlp, wl)
    df     = pd.DataFrame([feats])
    df     = df.replace([np.inf, -np.inf], 0).fillna(0)                          # fix: NaN guard
    df     = df.reindex(columns=mc['feature_columns'], fill_value=0)              # fix: col safety
    scaled = mc['scaler'].transform(df)
    tfidf  = mc['tfidf'].transform([window_text])
    X      = hstack([tfidf, scaled])
    return mc['model'].predict_proba(X)[0]

def predict_sentences_windowed(sentences, mc, nlp, wl, window=WINDOW_SIZE):
    """
    Fix: instead of predicting isolated sentences, use a sliding window of
    `window` sentences centred on each target. This gives the model enough
    context to produce meaningful TF-IDF and aggregate features.
    """
    n, results = len(sentences), []
    for i, sent in enumerate(sentences):
        lo  = max(0, i - window // 2)
        hi  = min(n, lo + window)
        lo  = max(0, hi - window)
        ctx = ' '.join(sentences[lo:hi])
        p   = _proba_for_window(ctx, mc, nlp, wl)
        results.append({'sentence': sent, 'ai_probability': float(p[1])})
    return results

def predict_text(text, mc, nlp, wl):
    cleaned = clean_text(text).lower()
    if len(cleaned.split()) < MIN_WORDS:
        return None, f"Need at least {MIN_WORDS} words."
    sents      = split_sentences(cleaned, nlp)
    prob       = _proba_for_window(cleaned, mc, nlp, wl)
    doc_feats  = aggregate_features(sents, nlp, wl)
    sent_preds = predict_sentences_windowed(sents, mc, nlp, wl)
    return {
        'probability':  prob,
        'sentences':    sent_preds,
        'doc_features': doc_feats,
    }, None

# ── PDF helpers ───────────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file):
    try:
        doc  = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return None

def build_pdf_report(result, threshold):
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=letter,
                               leftMargin=50, rightMargin=50,
                               topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    ai_p   = float(result['probability'][1])
    verdict = "AI-Generated" if ai_p > threshold else "Human-Written"

    title_style = ParagraphStyle('title', fontName='Helvetica-Bold',
                                 fontSize=18, spaceAfter=6, textColor=rl_colors.black)
    sub_style   = ParagraphStyle('sub', fontName='Helvetica',
                                 fontSize=11, spaceAfter=12, textColor=rl_colors.grey)
    body_style  = ParagraphStyle('body', fontName='Helvetica', fontSize=10,
                                 leading=14, spaceAfter=4)

    body = [
        Paragraph("🔍 AI Sniff-O-Meter — Detection Report", title_style),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", sub_style),
        Spacer(1, 8),
    ]

    # Summary table
    summary_data = [
        ["Verdict", verdict],
        ["AI Probability", f"{ai_p:.1%}"],
        ["Human Probability", f"{1-ai_p:.1%}"],
        ["Threshold Used", f"{threshold:.0%}"],
        ["Sentences Analysed", str(len(result['sentences']))],
    ]
    tbl = Table(summary_data, colWidths=[160, 260])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), rl_colors.lightyellow),
        ('BACKGROUND', (1,0), (1,0),
         rl_colors.lightcoral if ai_p > threshold else rl_colors.lightgreen),
        ('FONTNAME',  (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME',  (0,0), (0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',  (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor('#F9F9F9')]),
        ('BOX',       (0,0), (-1,-1), 1, rl_colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, rl_colors.grey),
        ('TOPPADDING',(0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
    ]))
    body.extend([tbl, Spacer(1, 16),
                 Paragraph("Sentence-Level Analysis", ParagraphStyle(
                     'h2', fontName='Helvetica-Bold', fontSize=13, spaceAfter=6))])

    for i, sd in enumerate(result['sentences']):
        p = sd['ai_probability']
        if p > 0.7:   color, label = rl_colors.lightcoral, "AI"
        elif p < 0.3:  color, label = rl_colors.lightgreen, "Human"
        else:          color, label = rl_colors.HexColor('#FFF3CD'), "Uncertain"
        row = Table([[Paragraph(f"[{label} {p:.0%}] {sd['sentence']}", body_style)]],
                    colWidths=[460])
        row.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), color),
            ('BOX',        (0,0), (-1,-1), 0.5, rl_colors.grey),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        body.extend([row, Spacer(1, 3)])

    doc.build(body)
    buf.seek(0)
    return buf

# ── UI renderers ──────────────────────────────────────────────────────────────
def render_verdict(result, threshold):
    ai_p = float(result['probability'][1])
    hu_p = float(result['probability'][0])
    pred = ai_p > threshold

    if pred and ai_p > 0.70:
        cls, icon, label = "verdict-ai",    "🤖", "Looks AI-Generated!"
    elif not pred and hu_p > 0.50:
        cls, icon, label = "verdict-human", "✍️", "Looks Human-Written!"
    else:
        cls, icon, label = "verdict-mixed", "🤔", "Hmm, Hard to Tell…"

    conf = max(ai_p, hu_p)
    st.markdown(f"""
    <div class="verdict-wrap {cls} slide-up">
        <div class="verdict-icon">{icon}</div>
        <div>
            <div class="verdict-tag">VERDICT (threshold {threshold:.0%})</div>
            <div class="verdict-title">{label}</div>
            <div class="verdict-prob">AI: {ai_p:.1%} &nbsp;|&nbsp; Human: {hu_p:.1%}</div>
        </div>
        <div class="verdict-score">
            <div class="verdict-score-num">{ai_p:.0%}</div>
            <div class="verdict-score-label">AI Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_chips(result):
    sents  = result['sentences']
    n      = len(sents)
    n_ai   = sum(1 for s in sents if s['ai_probability'] > 0.70)
    n_hu   = sum(1 for s in sents if s['ai_probability'] < 0.50)
    n_mx   = n - n_ai - n_hu
    ai_p   = float(result['probability'][1])
    st.markdown(f"""
    <div class="chips-row slide-up-1">
        <div class="chip">
            <div class="chip-val" style="color:#1A1A1A;">{n}</div>
            <div class="chip-label">Sentences</div>
        </div>
        <div class="chip">
            <div class="chip-val" style="color:#C53030;">{n_ai}</div>
            <div class="chip-label">🤖 AI Segments</div>
        </div>
        <div class="chip">
            <div class="chip-val" style="color:#276749;">{n_hu}</div>
            <div class="chip-label">✍️ Human Segments</div>
        </div>
        <div class="chip">
            <div class="chip-val" style="color:#B7791F;">{n_mx}</div>
            <div class="chip-label">🤔 Uncertain</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_gauge(ai_p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ai_p * 100,
        domain={'x': [0, 1], 'y': [0.2, 1]},
        number={'suffix':'%','font':{'family':'Nunito','size':38,'color':'#2D2D2D'}},
        title={'text':'AI Score','font':{'family':'Architects Daughter, cursive','size':14,'color':'#666'}},
        gauge={
            'axis': {'range':[0,100],'tickfont':{'color':'#999','size':10},'tickcolor':'#CCC'},
            'bar':  {'color':'#FFD93D','thickness':0.25},
            'bgcolor':'#F5F5F5',
            'borderwidth':2,
            'bordercolor':'#2D2D2D',
            'steps':[
                {'range':[0,30],  'color':'#C6F6D5'},
                {'range':[30,70], 'color':'#FEFCBF'},
                {'range':[70,100],'color':'#FED7D7'},
            ],
            'threshold':{'line':{'color':'#2D2D2D','width':3},'thickness':0.8,'value':ai_p*100},
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=10, r=10),
        height=180,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_bar_chart(result):
    sents  = result['sentences']
    probs  = [s['ai_probability'] for s in sents]
    labels = [f"S{i+1}" for i in range(len(sents))]
    clrs   = ['#FC8181' if p>0.7 else ('#68D391' if p<0.3 else '#F6E05E') for p in probs]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=probs,
        marker_color=clrs,
        marker_line_color='#2D2D2D',
        marker_line_width=1.5,
        hovertemplate='<b>%{x}</b><br>AI probability: %{y:.1%}<extra></extra>',
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#999", line_width=1.5)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=30, l=10, r=10),
        height=220,
        yaxis=dict(range=[0,1], tickformat='.0%',
                   gridcolor='rgba(0,0,0,0.05)',
                   tickfont=dict(color='#666', size=9)),
        xaxis=dict(tickfont=dict(color='#666', size=9), gridcolor='rgba(0,0,0,0)'),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_highlighted_text(result):
    st.markdown("""
    <div class="legend">
        <span><span class="legend-dot dot-ai"></span>AI (&gt;70%)</span>
        <span><span class="legend-dot dot-uncertain"></span>Uncertain (30–70%)</span>
        <span><span class="legend-dot dot-human"></span>Human (&lt;30%)</span>
    </div>
    """, unsafe_allow_html=True)
    parts = ['<div class="sent-annotated slide-up-2">']
    for sd in result['sentences']:
        p = sd['ai_probability']
        cls = "sent-ai" if p>0.7 else ("sent-human" if p<0.3 else "sent-uncertain")
        parts.append(f'<span class="sent-token {cls}" title="{p:.1%} AI">{sd["sentence"]}&nbsp;</span> ')
    parts.append('</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)

def render_sent_cards(result):
    for sd in result['sentences']:
        p = sd['ai_probability']
        if p > 0.7:
            card, badge, label = "sent-card-ai", "badge-ai", f"🤖 AI {p:.0%}"
        elif p < 0.5:
            card, badge, label = "sent-card-human", "badge-human", f"✍️ Human {p:.0%}"
        else:
            card, badge, label = "sent-card-uncertain", "badge-uncertain", f"🤔 Mixed {p:.0%}"
        st.markdown(f"""
        <div class="sent-card {card}">
            <span class="sent-badge {badge}">{label}</span>
            <span class="sent-text">{sd['sentence']}</span>
            <span class="sent-pct">{p:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

def render_feature_bars(doc_feats):
    ranges = {
        'chat_word_ratio':0.15,'punct_ratio':0.25,'ttr':1.0,
        'function_word_ratio':0.7,'discourse_ratio':0.15,
        'sentence_length_std':30.,'sentence_length_cv':1.5,
        'contraction_ratio':0.1,'transition_ratio':0.05,'formal_start_ratio':1.0,
    }
    colors = ['#FF6B6B','#FF9F43','#FFD93D','#6BCB77','#4D96FF',
              '#C77DFF','#F72585','#4CC9F0','#43AA8B','#F9844A']
    labels = {
        'chat_word_ratio':'Chat Word Ratio','punct_ratio':'Punctuation Ratio',
        'ttr':'Type-Token Ratio','function_word_ratio':'Function Word Ratio',
        'discourse_ratio':'Discourse Markers','sentence_length_std':'Sent Length Std',
        'sentence_length_cv':'Sent Length CV','contraction_ratio':'Contraction Ratio',
        'transition_ratio':'Transition Words','formal_start_ratio':'Formal Starts',
    }
    rows = []
    for i, (feat, val) in enumerate(doc_feats.items()):
        pct = min(val / max(ranges.get(feat,1.0), 1e-9), 1.0) * 100
        col = colors[i % len(colors)]
        rows.append(f"""
        <div class="feat-row">
            <div class="feat-top">
                <span style="color:#2D2D2D;">{labels.get(feat,feat)}</span>
                <span style="color:#666;">{val:.4f}</span>
            </div>
            <div class="feat-bg">
                <div class="feat-fill" style="width:{pct:.1f}%;background:{col};"></div>
            </div>
        </div>""")
    st.markdown('<div class="slide-up-3">' + ''.join(rows) + '</div>', unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Load models ───────────────────────────────────────────────────────────
    nlp = load_spacy()
    if nlp is None: st.stop()
    mc  = load_model()
    if mc  is None: st.stop()
    wl  = load_word_lists()

    # ── Masthead ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="masthead">
        <span class="masthead-emoji">🔍</span>
        <div class="masthead-title">AI Sniff-O-Meter</div>
        <div class="masthead-sub">Paste text or upload a PDF — we'll sniff out the robots! 🐾</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar settings ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        threshold = st.slider(
            "🎯 AI Detection Threshold",
            min_value=0.10, max_value=0.90,
            value=0.50, step=0.05,
            help="Scores above this → AI. Lower = more sensitive."
        )
        st.markdown(f"""
        <div style="background:#FFD93D;border:2px solid #2D2D2D;border-radius:10px;
             padding:0.6rem 0.8rem;margin:0.5rem 0;font-weight:700;font-size:0.85rem;">
            🤖 AI if score &gt; <b>{threshold:.0%}</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎨 Legend")
        st.markdown("""
        <div style="font-size:0.85rem;font-weight:600;line-height:2;">
            🔴 <b>AI</b> — probability &gt; 70%<br>
            🟡 <b>Uncertain</b> — 30% – 70%<br>
            🟢 <b>Human</b> — probability &lt; 30%
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem;color:#888;font-weight:600;">
            Uses windowed context prediction —<br>
            each sentence is scored with its<br>
            surrounding neighbours for accuracy.
        </div>
        """, unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">📝 Enter Your Text</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Input method",
        ["✏️ Paste text", "📄 Upload PDF"],
        horizontal=True,
    )

    text_input     = ""
    pdf_bytes_buf  = None   # store original PDF bytes for download

    if mode == "✏️ Paste text":
        text_input = st.text_area(
            "Text to analyse",
            height=220,
            placeholder="Paste your essay, article, report, or anything else here…",
        )
    else:
        uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded:
            if uploaded.size > 2 * 1024 * 1024:
                st.error("⚠️ File too large! Please upload a PDF under 2MB for best performance.")
                st.stop()
            raw_bytes = uploaded.read()
            pdf_bytes_buf = raw_bytes          # keep original for download
            try:
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
                text_input = "".join(page.get_text() for page in doc)
            except Exception as e:
                st.error(f"Could not read PDF: {e}")
                text_input = ""
            if text_input:
                wc = len(text_input.split())
                st.success(f"✅ Extracted **{wc:,} words** from PDF!")
                with st.expander("📄 Preview extracted text"):
                    st.text(text_input[:2000] + ("…" if len(text_input) > 2000 else ""))

    # ── Analyse button ────────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run = st.button("🔍 Sniff It Out!", use_container_width=True)

    # ── Run analysis ──────────────────────────────────────────────────────────
    if run:
        if not text_input or len(text_input.strip().split()) < MIN_WORDS:
            st.error(f"⚠️ Please enter at least {MIN_WORDS} words!")
            return

        prog = st.progress(0)
        prog.progress(15, "🧹 Cleaning text…")
        prog.progress(45, "🔬 Running detection pipeline…")

        result, err = predict_text(text_input, mc, nlp, wl)

        if err:
            st.error(f"❌ {err}")
            prog.empty()
            return

        prog.progress(100, "✅ Done!")
        prog.empty()

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-head">📊 Results</div>', unsafe_allow_html=True)

        render_verdict(result, threshold)
        render_chips(result)

        g_col, b_col = st.columns([1, 2], gap="large")
        with g_col:
            render_gauge(float(result['probability'][1]))
        with b_col:
            st.markdown("**Per-sentence AI probability**")
            render_bar_chart(result)

        st.markdown("---")
        st.markdown('<div class="section-head">🔎 Sentence Analysis</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🎨 Highlighted Text", "📋 Sentence Cards", "📈 Feature Profile"])
        with tab1:
            render_highlighted_text(result)
        with tab2:
            render_sent_cards(result)
        with tab3:
            render_feature_bars(result['doc_features'])

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-head">💾 Download</div>', unsafe_allow_html=True)

        ecol1, ecol2 = st.columns(2, gap="small")

        with ecol1:
            report_buf = build_pdf_report(result, threshold)
            st.download_button(
                label="📄 Download Analysis Report (PDF)",
                data=report_buf,
                file_name=f"ai_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with ecol2:
            # If analysis came from a PDF upload, offer original PDF download too
            if pdf_bytes_buf is not None:
                st.download_button(
                    label="📂 Download Original PDF",
                    data=pdf_bytes_buf,
                    file_name=f"original_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.info("ℹ️ Upload a PDF to also download the original here.", icon="📎")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;margin-top:3rem;padding:1rem 0;
         border-top:2px dashed #CCC;font-size:0.8rem;color:#999;font-weight:600;">
        🤖 AI Sniff-O-Meter · 10-feature linguistic model · Python 3.11 · Streamlit 1.29
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
