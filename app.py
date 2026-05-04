"""
Rezolv Intelligence — PM Experimentation App

A workspace for iterating on prompts, inputs, and final briefings until you
get the output you want. Built for PMs, not engineers — minimal setup, fast
iteration loop, everything visible and editable.

Layout:
  Top bar: account picker · today · model · API key
  Tabs:
    1. 📱 Field App         — mobile preview of final briefing
    2. ⚡ Pipeline          — Run Stage 1 / Stage 2 buttons + JSON outputs
    3. ✏️  Prompts          — edit Stage 1 and Stage 2 prompts inline
    4. 📊 Input Data        — view + tweak the features dict for what-if testing

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime, date
from anthropic import Anthropic

# Optional providers — gracefully degrade if not installed
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from features import build_features
from prompts_analyser import (
    SYSTEM_PROMPT as DEFAULT_ANALYSER_PROMPT,
    build_user_message as build_analyser_user_msg,
)
from prompts_writer import (
    SYSTEM_PROMPT as DEFAULT_WRITER_PROMPT,
    build_user_message as build_writer_user_msg,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Rezolv Intelligence — Experimentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Cleaner styling — borrowing from a calm experimentation tool aesthetic
st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    padding: 10px 16px;
    background: transparent;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    background: rgba(74, 144, 226, 0.1);
    color: #4a90e2;
}

/* Mobile preview frame */
.mobile-frame {
    max-width: 380px;
    margin: 20px auto;
    background: #0f1216;
    border: 8px solid #1a1f25;
    border-radius: 28px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    overflow: hidden;
    min-height: 600px;
}
.mobile-statusbar {
    background: #0a0d11;
    color: #888;
    padding: 6px 18px;
    font-size: 11px;
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #1a1f25;
}
.mobile-content {
    padding: 18px 16px 24px;
    color: #e8e8e8;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.mobile-customer-header {
    border-bottom: 1px solid #2a2f35;
    padding-bottom: 14px;
    margin-bottom: 16px;
}
.mobile-case-ref {
    font-size: 11px;
    color: #6f7780;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.mobile-customer-name {
    font-size: 20px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 2px;
}
.mobile-meta {
    font-size: 12px;
    color: #8a8f96;
    margin-top: 4px;
}
.mobile-briefing-label {
    font-size: 10px;
    color: #4a90e2;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
    margin-bottom: 8px;
}
.mobile-headline {
    font-size: 15px;
    font-weight: 500;
    color: #fff;
    line-height: 1.45;
    margin-bottom: 18px;
}
.mobile-section-label {
    font-size: 10px;
    color: #6f7780;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    font-weight: 600;
}
.mobile-bullet {
    font-size: 13px;
    color: #cccccc;
    line-height: 1.55;
    margin-bottom: 8px;
    padding-left: 14px;
    position: relative;
}
.mobile-bullet:before {
    content: "•";
    color: #4a90e2;
    position: absolute;
    left: 0;
    font-weight: bold;
}
.mobile-angle {
    font-size: 13px;
    color: #d8d8d8;
    line-height: 1.55;
    background: #1a1f25;
    border-left: 3px solid #4a90e2;
    padding: 12px 14px;
    border-radius: 0 6px 6px 0;
    margin-top: 8px;
    font-style: italic;
}
.mobile-empty {
    text-align: center;
    color: #555;
    padding: 60px 20px;
    font-size: 13px;
}
.mobile-action-row {
    display: flex;
    gap: 8px;
    margin-top: 24px;
}
.mobile-action {
    flex: 1;
    padding: 12px;
    background: #1a1f25;
    color: #d8d8d8;
    text-align: center;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
}
.mobile-action.primary {
    background: #4a90e2;
    color: #fff;
}
.indicator-pill {
    display: inline-block;
    background: #1a1f25;
    color: #aaa;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 10px;
    margin-right: 4px;
    margin-bottom: 4px;
    font-family: monospace;
}
.indicator-pill.high { background: #3a1f1f; color: #ff8a8a; }
.indicator-pill.med { background: #2f2a1a; color: #e8c270; }

/* Run buttons */
.stage-card {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.stage-status {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: #57606a;
    margin-bottom: 4px;
    font-weight: 600;
}

/* Inline run badge */
.run-badge {
    display: inline-block;
    background: #1f883d;
    color: white;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 10px;
    margin-left: 6px;
}
.run-badge.stale { background: #bf8700; }
.run-badge.error { background: #cf222e; }

/* Editor area */
.editor-info {
    font-size: 12px;
    color: #57606a;
    background: #f6f8fa;
    border-left: 3px solid #4a90e2;
    padding: 8px 12px;
    border-radius: 4px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Initialise session state
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    if "analyser_prompt" not in st.session_state:
        st.session_state.analyser_prompt = DEFAULT_ANALYSER_PROMPT
    if "writer_prompt" not in st.session_state:
        st.session_state.writer_prompt = DEFAULT_WRITER_PROMPT
    if "features_dict" not in st.session_state:
        st.session_state.features_dict = None
    if "features_dict_edited" not in st.session_state:
        st.session_state.features_dict_edited = None
    if "stage1_output" not in st.session_state:
        st.session_state.stage1_output = None
    if "stage2_output" not in st.session_state:
        st.session_state.stage2_output = None
    if "stage1_meta" not in st.session_state:
        st.session_state.stage1_meta = None  # tokens, latency
    if "stage2_meta" not in st.session_state:
        st.session_state.stage2_meta = None
    if "stage1_stale" not in st.session_state:
        st.session_state.stage1_stale = False
    if "stage2_stale" not in st.session_state:
        st.session_state.stage2_stale = False

init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    try:
        disp = pd.read_csv("sample_data/dispositions.csv")
        pay = pd.read_csv("sample_data/payments.csv")
        bom = pd.read_csv("sample_data/bom_snapshots.csv")
        with open("sample_data/case_summary.json") as f:
            cases = {c["case_ref"]: c for c in json.load(f)}
        return disp, pay, bom, cases
    except FileNotFoundError:
        return None, None, None, None


disp_df, pay_df, bom_df, cases = load_data()

if disp_df is None:
    st.error("Sample data not found. Run `python scripts/generate_realistic.py` first to generate test accounts.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def strip_fences(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    return m.group(1).strip() if m else text


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — declarative list of supported models
# Each entry: (display_label, model_id, provider)
# Add new models here and they appear in both stage dropdowns automatically.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = [
    # Anthropic
    ("Claude Opus 4.7 (best quality)",  "claude-opus-4-7",         "anthropic"),
    ("Claude Sonnet 4.6 (balanced)",    "claude-sonnet-4-6",       "anthropic"),
    ("Claude Haiku 4.5 (fast/cheap)",   "claude-haiku-4-5-20251001","anthropic"),
    # OpenAI
    ("GPT-4.1",                          "gpt-4.1",                 "openai"),
    ("GPT-4o",                           "gpt-4o",                  "openai"),
    ("GPT-4o mini (fast/cheap)",         "gpt-4o-mini",             "openai"),
    # Google
    ("Gemini 3 Pro (best quality)",     "gemini-3.1-pro-preview",       "google"),
    ("Gemini 3 Flash (balanced)",       "gemini-3-flash-preview",       "google"),
    ("Gemini 2.5 Pro",                  "gemini-2.5-pro",               "google"),
    ("Gemini 2.5 Flash (fast/cheap)",   "gemini-2.5-flash",             "google"),
]
]

MODEL_LABELS = [m[0] for m in MODEL_REGISTRY]
MODEL_BY_LABEL = {m[0]: (m[1], m[2]) for m in MODEL_REGISTRY}
PROVIDER_KEY_NAMES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "google":    "GOOGLE_API_KEY",
}


def get_provider_key(provider, manual_keys):
    """Get API key for a provider. Order: manual input → secrets → env var."""
    key_name = PROVIDER_KEY_NAMES[provider]
    # 1. Manual entry in UI
    if manual_keys.get(provider):
        return manual_keys[provider]
    # 2. Streamlit secrets (cloud)
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    # 3. Environment variable (local)
    return os.environ.get(key_name, "")


def call_llm(system_prompt, user_message, model_id, provider, api_key, max_tokens=4096):
    """Unified LLM caller — dispatches to the right provider SDK.
    Returns (parsed_json, meta). On error, parsed_json contains error key."""
    t0 = time.time()

    if not api_key:
        return {"error": "no_api_key", "exception": f"No API key for {provider}"}, {
            "latency_s": 0, "error": "no_key", "model": model_id, "provider": provider,
        }

    try:
        if provider == "anthropic":
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                return {"error": "openai_not_installed",
                        "exception": "Add 'openai' to requirements.txt"}, {
                    "latency_s": 0, "error": "missing_pkg",
                    "model": model_id, "provider": provider,
                }
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = response.choices[0].message.content.strip()
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        elif provider == "google":
            if not GEMINI_AVAILABLE:
                return {"error": "gemini_not_installed",
                        "exception": "Add 'google-generativeai' to requirements.txt"}, {
                    "latency_s": 0, "error": "missing_pkg",
                    "model": model_id, "provider": provider,
                }
            genai.configure(api_key=api_key)
            # Gemini takes system instruction at model init, user msg at generate
            model = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=system_prompt,
            )
            response = model.generate_content(
                user_message,
                generation_config={"max_output_tokens": max_tokens},
            )
            raw = response.text.strip()
            # Gemini reports usage on response
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

        else:
            return {"error": "unknown_provider", "exception": f"Provider '{provider}' not supported"}, {
                "latency_s": 0, "error": "unknown", "model": model_id, "provider": provider,
            }

        elapsed = time.time() - t0
        cleaned = strip_fences(raw)

        try:
            parsed = json.loads(cleaned)
            return parsed, {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_s": round(elapsed, 2),
                "model": model_id,
                "provider": provider,
                "error": None,
            }
        except json.JSONDecodeError as e:
            return {"error": "json_parse_failed", "raw": raw, "exception": str(e)}, {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_s": round(elapsed, 2),
                "model": model_id,
                "provider": provider,
                "error": "parse",
            }

    except Exception as e:
        return {"error": "api_call_failed", "exception": str(e)}, {
            "latency_s": round(time.time() - t0, 2),
            "error": "api",
            "model": model_id,
            "provider": provider,
        }


def build_or_get_features(case_ref, today_dt, emi, lat, lng):
    """Build features dict (or use edited one if it exists for this case)."""
    if (st.session_state.features_dict_edited is not None
            and st.session_state.features_dict_edited.get("case_ref_number") == case_ref):
        return st.session_state.features_dict_edited

    return build_features(
        case_ref=case_ref,
        dispositions_df=disp_df, payments_df=pay_df, bom_df=bom_df,
        registered_lat=lat, registered_lng=lng,
        today=today_dt, emi_amount=emi,
    )


def status_pill(label, status):
    """Small inline status indicator."""
    color = {"ok": "#1f883d", "stale": "#bf8700", "error": "#cf222e", "none": "#57606a"}.get(status, "#57606a")
    return f'<span style="display:inline-block;background:{color};color:white;padding:1px 8px;border-radius:10px;font-size:10px;margin-left:6px;">{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Top bar — account, date, model, API key
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 🔬 Rezolv Intelligence — Experimentation Workspace")
st.caption("Tweak prompts. Tweak data. Mix providers. Iterate till the briefing reads right.")

bar = st.columns([1.8, 1.0, 1.6, 1.6, 1.0])

with bar[0]:
    case_options = sorted(disp_df["case_ref_number"].unique().tolist())
    selected_case = st.selectbox("Account", case_options, key="account_picker",
                                  format_func=lambda x: f"{x.replace('CR_', '').replace('_', ' · ')}")

with bar[1]:
    today_input = st.date_input("Today", value=date(2026, 4, 30), key="today_input")

with bar[2]:
    stage1_model_label = st.selectbox(
        "Stage 1 model (Analyser)",
        MODEL_LABELS,
        index=0,  # default Opus
        key="stage1_model_choice",
        help="Use a stronger model here — it does the structured analysis.",
    )

with bar[3]:
    stage2_model_label = st.selectbox(
        "Stage 2 model (Writer)",
        MODEL_LABELS,
        index=1,  # default Sonnet
        key="stage2_model_choice",
        help="Cheaper models work fine here — the job is rewriting in tone, not reasoning.",
    )

with bar[4]:
    emi_amount = st.number_input("EMI (₹)", value=12300, step=500, key="emi_input")

# Resolve model_id and provider for each stage
stage1_model_id, stage1_provider = MODEL_BY_LABEL[stage1_model_label]
stage2_model_id, stage2_provider = MODEL_BY_LABEL[stage2_model_label]

# API keys section — collapsed by default once keys are set
needed_providers = {stage1_provider, stage2_provider}
with st.expander("🔑 API keys", expanded=False):
    st.caption(
        "Enter keys for the providers you're using. "
        "Keys are also auto-detected from Streamlit Cloud secrets or environment variables. "
        f"**This run needs:** {', '.join(sorted(needed_providers))}"
    )
    key_cols = st.columns(3)
    manual_keys = {}
    with key_cols[0]:
        anthropic_input = st.text_input(
            "Anthropic API key",
            type="password",
            key="anthropic_key_input",
            placeholder="sk-ant-...",
            help="Sets ANTHROPIC_API_KEY for this session",
        )
        if anthropic_input:
            manual_keys["anthropic"] = anthropic_input
    with key_cols[1]:
        openai_input = st.text_input(
            "OpenAI API key",
            type="password",
            key="openai_key_input",
            placeholder="sk-...",
            help="Sets OPENAI_API_KEY for this session",
        )
        if openai_input:
            manual_keys["openai"] = openai_input
    with key_cols[2]:
        google_input = st.text_input(
            "Google API key",
            type="password",
            key="google_key_input",
            placeholder="AIza...",
            help="Sets GOOGLE_API_KEY for this session",
        )
        if google_input:
            manual_keys["google"] = google_input

    # Show which keys are detected
    status_cols = st.columns(3)
    for col, prov in zip(status_cols, ["anthropic", "openai", "google"]):
        with col:
            key = get_provider_key(prov, manual_keys)
            if key:
                st.markdown(f"<span style='color:#1f883d;font-size:11px;'>✓ {prov} key detected</span>",
                            unsafe_allow_html=True)
            else:
                needed = "(needed)" if prov in needed_providers else ""
                st.markdown(f"<span style='color:#888;font-size:11px;'>○ {prov} key not set {needed}</span>",
                            unsafe_allow_html=True)

# Customer/case context — pull from case summary
case_info = cases.get(selected_case, {})
reg_lat = case_info.get("registered_lat", 19.1136)
reg_lng = case_info.get("registered_lng", 72.8697)
customer_name = case_info.get("name", "Unknown Customer")

# Build features for current case (or use edited version)
today_dt = datetime.combine(today_input, datetime.min.time())
current_features = build_or_get_features(selected_case, today_dt, emi_amount, reg_lat, reg_lng)
st.session_state.features_dict = current_features

# If account changed, mark Stage 1+2 stale
if (st.session_state.stage1_output and
        st.session_state.stage1_output.get("case_ref_number") != selected_case):
    st.session_state.stage1_stale = True
    st.session_state.stage2_stale = True


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_field, tab_pipeline, tab_prompts, tab_input, tab_triggers = st.tabs([
    "📱 Field App", "⚡ Pipeline", "✏️ Prompts", "📊 Input Data", "📚 Triggers Reference"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Field App preview
# ═════════════════════════════════════════════════════════════════════════════

with tab_field:
    col_phone, col_meta = st.columns([1, 1.2])

    with col_phone:
        # Mobile preview
        briefing = st.session_state.stage2_output
        analyser = st.session_state.stage1_output

        # Top: case header
        if briefing and "headline" in briefing:
            # Header info derived from features
            current_dpd = current_features.get("bom", {}).get("current_dpd", 0)
            current_bucket = current_features.get("bom", {}).get("current_bucket", "—")
            n_disp = current_features.get("dispositions", {}).get("total_dispositions", 0)

            # Headline + bullets + angle
            bullets_html = "".join(
                f'<div class="mobile-bullet">{b}</div>'
                for b in briefing.get("whats_notable", [])
            )

            # Indicator pills (top 4 from ranked_top, with severity color hint)
            pills_html = ""
            if analyser:
                ranked = analyser.get("ranked_top", [])[:4]
                for ind in ranked:
                    sev = analyser.get("indicators", {}).get(ind, {}).get("severity", 0)
                    pill_class = "high" if sev >= 75 else ("med" if sev >= 50 else "")
                    short_name = ind.replace("_", " ").replace("indicator", "").strip()
                    pills_html += f'<span class="indicator-pill {pill_class}">{short_name}</span>'

            mobile_html = f"""
            <div class="mobile-frame">
                <div class="mobile-statusbar">
                    <span>9:41 AM</span>
                    <span>📶 Rezolv</span>
                </div>
                <div class="mobile-content">
                    <div class="mobile-customer-header">
                        <div class="mobile-case-ref">{selected_case}</div>
                        <div class="mobile-customer-name">{customer_name}</div>
                        <div class="mobile-meta">DPD {current_dpd} · Bucket {current_bucket} · {n_disp} interactions</div>
                    </div>

                    <div class="mobile-briefing-label">Pre-visit briefing</div>
                    <div class="mobile-headline">{briefing.get("headline", "")}</div>

                    <div class="mobile-section-label">What's notable</div>
                    {bullets_html}

                    <div class="mobile-section-label" style="margin-top:18px;">Suggested angle</div>
                    <div class="mobile-angle">{briefing.get("suggested_angle", "")}</div>

                    {f'<div style="margin-top:20px;">{pills_html}</div>' if pills_html else ''}

                    <div class="mobile-action-row">
                        <div class="mobile-action">View full history</div>
                        <div class="mobile-action primary">Mark visit</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(mobile_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="mobile-frame">
                <div class="mobile-statusbar">
                    <span>9:41 AM</span>
                    <span>📶 Rezolv</span>
                </div>
                <div class="mobile-content">
                    <div class="mobile-empty">
                        Run Stage 1 + Stage 2 in the<br>Pipeline tab to see the briefing.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_meta:
        st.markdown("##### What you're looking at")
        st.caption("This is the briefing the field agent reads on their phone before knocking on the customer's door. They have ~15 seconds to read it.")

        if briefing:
            st.markdown("##### Quick checks")
            checks = []
            headline_len = len(briefing.get("headline", ""))
            checks.append((
                "Headline under 30 words",
                "✅" if len(briefing.get("headline", "").split()) <= 30 else "⚠️",
                f"{len(briefing.get('headline', '').split())} words",
            ))
            n_bullets = len(briefing.get("whats_notable", []))
            checks.append((
                "2-4 bullets",
                "✅" if 2 <= n_bullets <= 4 else "⚠️",
                f"{n_bullets} bullets",
            ))
            text_blob = (briefing.get("headline", "") + " " +
                          " ".join(briefing.get("whats_notable", [])) + " " +
                          briefing.get("suggested_angle", "")).lower()
            forbidden = ["demand", "must ", "do not accept", "ensure", "it is critical",
                          "evasive", "lying", "fraudulent", "dodging"]
            found = [w for w in forbidden if w in text_blob]
            checks.append((
                "No forbidden words",
                "✅" if not found else "⚠️",
                "" if not found else f"found: {', '.join(found)}",
            ))

            for label, mark, detail in checks:
                st.markdown(f"{mark} {label}  <span style='color:#888;font-size:12px;'>{detail}</span>",
                            unsafe_allow_html=True)

            st.markdown("##### Cost & latency")
            if st.session_state.stage1_meta and st.session_state.stage2_meta:
                m1 = st.session_state.stage1_meta
                m2 = st.session_state.stage2_meta
                total_in = (m1.get("input_tokens", 0) or 0) + (m2.get("input_tokens", 0) or 0)
                total_out = (m1.get("output_tokens", 0) or 0) + (m2.get("output_tokens", 0) or 0)
                total_latency = (m1.get("latency_s", 0) or 0) + (m2.get("latency_s", 0) or 0)
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Input tokens", f"{total_in:,}")
                cc2.metric("Output tokens", f"{total_out:,}")
                cc3.metric("Total latency", f"{total_latency:.1f}s")

        st.markdown("##### Iteration tips")
        st.caption(
            "• Edit the writer prompt in the **Prompts** tab to change tone.\n\n"
            "• Edit the features dict in the **Input Data** tab to test edge cases.\n\n"
            "• Re-run only Stage 2 if you've only changed the writer prompt — saves tokens.\n\n"
            "• Headline >30 words usually means writer is summarising too much instead of leading with one thing."
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Pipeline (run Stage 1 / Stage 2)
# ═════════════════════════════════════════════════════════════════════════════

with tab_pipeline:
    pcols = st.columns(2)

    # ── Stage 1 ──
    with pcols[0]:
        s1_status = "none"
        if st.session_state.stage1_meta:
            if st.session_state.stage1_meta.get("error"):
                s1_status = "error"
            elif st.session_state.stage1_stale:
                s1_status = "stale"
            else:
                s1_status = "ok"

        st.markdown(f"#### Stage 1 — Analyser {status_pill(s1_status.upper(), s1_status)}",
                    unsafe_allow_html=True)
        st.caption("Reads the features dict, picks states, scores severity. No prose, just structured analysis.")

        run_s1 = st.button("▶ Run Stage 1", type="primary", use_container_width=True, key="run_s1")

        if run_s1:
            stage1_key = get_provider_key(stage1_provider, manual_keys)
            if not stage1_key:
                st.error(f"No API key for {stage1_provider}. Add it in the API keys section.")
            else:
                with st.spinner(f"Running analyser ({stage1_model_label})..."):
                    user_msg = build_analyser_user_msg(current_features)
                    output, meta = call_llm(
                        st.session_state.analyser_prompt,
                        user_msg,
                        stage1_model_id,
                        stage1_provider,
                        stage1_key,
                        max_tokens=4096,
                    )
                    st.session_state.stage1_output = output
                    st.session_state.stage1_meta = meta
                    st.session_state.stage1_stale = False
                    st.session_state.stage2_stale = True  # writer needs re-run
                st.rerun()

        if st.session_state.stage1_meta:
            m = st.session_state.stage1_meta
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Input tok", f"{m.get('input_tokens', 0):,}" if m.get('input_tokens') else "—")
            cm2.metric("Output tok", f"{m.get('output_tokens', 0):,}" if m.get('output_tokens') else "—")
            cm3.metric("Latency", f"{m.get('latency_s', 0)}s")
            if m.get('provider'):
                st.caption(f"via {m['provider']} · {m.get('model', '')}")

        if st.session_state.stage1_output:
            with st.expander("📄 Stage 1 output (analyser JSON)", expanded=False):
                st.json(st.session_state.stage1_output)

            # Quick view: top indicators
            if "ranked_top" in st.session_state.stage1_output:
                st.markdown("**Ranked top indicators:**")
                ranked = st.session_state.stage1_output.get("ranked_top", [])
                indicators = st.session_state.stage1_output.get("indicators", {})
                for i, ind in enumerate(ranked[:8], 1):
                    detail = indicators.get(ind, {})
                    sev = detail.get("severity", 0)
                    state = detail.get("state", "?")
                    amp = detail.get("amplifier", "neutral")
                    amp_marker = " ⬆" if amp == "amplified" else (" ⬇" if amp == "muted" else "")
                    st.markdown(
                        f"{i}. `{ind}` → **{state}**{amp_marker}  "
                        f"<span style='color:#888;font-size:12px;'>severity {sev}</span>",
                        unsafe_allow_html=True,
                    )

    # ── Stage 2 ──
    with pcols[1]:
        s2_status = "none"
        if st.session_state.stage2_meta:
            if st.session_state.stage2_meta.get("error"):
                s2_status = "error"
            elif st.session_state.stage2_stale:
                s2_status = "stale"
            else:
                s2_status = "ok"

        st.markdown(f"#### Stage 2 — Writer {status_pill(s2_status.upper(), s2_status)}",
                    unsafe_allow_html=True)
        st.caption("Reads Stage 1's output, writes the agent-facing briefing. Tone happens here.")

        s2_disabled = st.session_state.stage1_output is None
        run_s2 = st.button(
            "▶ Run Stage 2",
            type="primary",
            use_container_width=True,
            key="run_s2",
            disabled=s2_disabled,
            help="Run Stage 1 first" if s2_disabled else None,
        )

        if run_s2:
            stage2_key = get_provider_key(stage2_provider, manual_keys)
            if not stage2_key:
                st.error(f"No API key for {stage2_provider}. Add it in the API keys section.")
            else:
                with st.spinner(f"Running writer ({stage2_model_label})..."):
                    user_msg = build_writer_user_msg(st.session_state.stage1_output)
                    output, meta = call_llm(
                        st.session_state.writer_prompt,
                        user_msg,
                        stage2_model_id,
                        stage2_provider,
                        stage2_key,
                        max_tokens=4096,
                    )
                    st.session_state.stage2_output = output
                    st.session_state.stage2_meta = meta
                    st.session_state.stage2_stale = False
                st.rerun()

        if st.session_state.stage2_meta:
            m = st.session_state.stage2_meta
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("Input tok", f"{m.get('input_tokens', 0):,}" if m.get('input_tokens') else "—")
            cm2.metric("Output tok", f"{m.get('output_tokens', 0):,}" if m.get('output_tokens') else "—")
            cm3.metric("Latency", f"{m.get('latency_s', 0)}s")
            if m.get('provider'):
                st.caption(f"via {m['provider']} · {m.get('model', '')}")

        if st.session_state.stage2_output:
            with st.expander("📄 Stage 2 output (briefing JSON)", expanded=False):
                st.json(st.session_state.stage2_output)

    # Pipeline tips
    st.divider()
    st.caption(
        "💡 **Iteration tip:** when you only edit the writer prompt, re-run *only Stage 2* — "
        "Stage 1's output is reusable. When you edit the features dict or the analyser prompt, "
        "re-run Stage 1 (which will mark Stage 2 as stale)."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Prompts editor
# ═════════════════════════════════════════════════════════════════════════════

with tab_prompts:
    sub_a, sub_b = st.tabs(["Stage 1 — Analyser", "Stage 2 — Writer"])

    with sub_a:
        st.markdown(
            '<div class="editor-info">Edit the analyser prompt below. '
            'Any change marks Stage 1 as stale — re-run from the Pipeline tab.</div>',
            unsafe_allow_html=True,
        )
        cc = st.columns([4, 1])
        with cc[1]:
            if st.button("↩ Reset to default", key="reset_a", use_container_width=True):
                st.session_state.analyser_prompt = DEFAULT_ANALYSER_PROMPT
                st.session_state.stage1_stale = True
                st.rerun()

        new_analyser = st.text_area(
            "Analyser system prompt",
            value=st.session_state.analyser_prompt,
            height=600,
            key="analyser_editor",
            label_visibility="collapsed",
        )
        if new_analyser != st.session_state.analyser_prompt:
            st.session_state.analyser_prompt = new_analyser
            st.session_state.stage1_stale = True
        st.caption(f"{len(new_analyser):,} chars · ~{len(new_analyser)//4:,} tokens")

    with sub_b:
        st.markdown(
            '<div class="editor-info">Edit the writer prompt below. This is where most of your '
            'iteration will happen — tone, structure, language. Re-run only Stage 2 to see changes.</div>',
            unsafe_allow_html=True,
        )
        cc = st.columns([4, 1])
        with cc[1]:
            if st.button("↩ Reset to default", key="reset_b", use_container_width=True):
                st.session_state.writer_prompt = DEFAULT_WRITER_PROMPT
                st.session_state.stage2_stale = True
                st.rerun()

        new_writer = st.text_area(
            "Writer system prompt",
            value=st.session_state.writer_prompt,
            height=600,
            key="writer_editor",
            label_visibility="collapsed",
        )
        if new_writer != st.session_state.writer_prompt:
            st.session_state.writer_prompt = new_writer
            st.session_state.stage2_stale = True
        st.caption(f"{len(new_writer):,} chars · ~{len(new_writer)//4:,} tokens")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Input data editor
# ═════════════════════════════════════════════════════════════════════════════

with tab_input:
    st.markdown(
        '<div class="editor-info">This is the features dict for the selected account — '
        'about 40 metrics that Stage 1 reads. You can edit values inline to test edge cases '
        '(what if PTP honour rate were 49% instead of 50%? what if there were no settlement requests?). '
        'Edits mark both stages as stale.</div>',
        unsafe_allow_html=True,
    )

    bcols = st.columns([1, 1, 4])
    with bcols[0]:
        if st.button("↩ Reset to original", use_container_width=True, key="reset_features"):
            st.session_state.features_dict_edited = None
            st.session_state.stage1_stale = True
            st.session_state.stage2_stale = True
            st.rerun()
    with bcols[1]:
        view_mode = st.radio("View", ["Tree", "Raw JSON"], horizontal=True, label_visibility="collapsed")

    if view_mode == "Tree":
        # Editable tree view — major sections expandable
        edited = deepcopy(current_features)
        changed = False

        for section_key in ["dispositions", "payments", "bom", "ptp_honour", "address_intelligence_resolved"]:
            if section_key not in edited:
                continue
            with st.expander(f"📂 {section_key}", expanded=(section_key in ["dispositions", "ptp_honour"])):
                section = edited.get(section_key)
                if section is None:
                    st.write("None")
                    continue
                if isinstance(section, dict):
                    # Show top-level scalars editable, nested as JSON
                    for k, v in list(section.items()):
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            new_v = st.number_input(
                                f"{section_key}.{k}",
                                value=v,
                                key=f"edit_{section_key}_{k}",
                                step=1 if isinstance(v, int) else 0.01,
                            )
                            if new_v != v:
                                edited[section_key][k] = new_v
                                changed = True
                        elif isinstance(v, str):
                            new_v = st.text_input(
                                f"{section_key}.{k}",
                                value=v,
                                key=f"edit_{section_key}_{k}",
                            )
                            if new_v != v:
                                edited[section_key][k] = new_v
                                changed = True
                        elif isinstance(v, bool):
                            new_v = st.checkbox(
                                f"{section_key}.{k}",
                                value=v,
                                key=f"edit_{section_key}_{k}",
                            )
                            if new_v != v:
                                edited[section_key][k] = new_v
                                changed = True
                        elif v is None:
                            st.text_input(f"{section_key}.{k}", value="None (read-only)", disabled=True,
                                          key=f"edit_{section_key}_{k}")
                        else:
                            # List or dict — show as JSON, allow editing
                            json_str = json.dumps(v, default=str, indent=2)
                            new_str = st.text_area(
                                f"{section_key}.{k}",
                                value=json_str,
                                key=f"edit_{section_key}_{k}",
                                height=120,
                            )
                            if new_str != json_str:
                                try:
                                    edited[section_key][k] = json.loads(new_str)
                                    changed = True
                                except json.JSONDecodeError:
                                    st.warning(f"Invalid JSON in {section_key}.{k}; keeping original.")

        if changed:
            st.session_state.features_dict_edited = edited
            st.session_state.stage1_stale = True
            st.session_state.stage2_stale = True
            st.success("Features dict updated. Re-run Stage 1 to apply.")

    else:
        # Raw JSON editor
        json_str = json.dumps(current_features, indent=2, default=str)
        new_str = st.text_area(
            "Features dict",
            value=json_str,
            height=600,
            label_visibility="collapsed",
        )
        if new_str != json_str:
            try:
                parsed = json.loads(new_str)
                st.session_state.features_dict_edited = parsed
                st.session_state.stage1_stale = True
                st.session_state.stage2_stale = True
                st.success("Features dict updated.")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    st.caption(
        f"This dict is what Stage 1 reads. ~{len(json.dumps(current_features, default=str)):,} bytes. "
        "In production, this would be precomputed by a pandas job from raw CSVs/DB."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Triggers Reference
# ═════════════════════════════════════════════════════════════════════════════

# Static catalogue of all 17 indicators with states + severity table + amplifier rules
INDICATOR_CATALOGUE = [
    # (id, category, description, states_with_severity, sample_size, drives_action)
    {
        "id": "defaulter_type_classifier",
        "category": "Account trajectory",
        "description": "Classifies the customer's default pattern across the BOM history. Drives the overall conversation tone since archetype shapes which approach works.",
        "states": [
            ("serial_early_emi_defaulter", 85, "DPD > 0 in any of first 3 months — likely fraud or willful default from origination."),
            ("deteriorating_defaulter", 80, "≥3 default cycles, each peak bucket worse than the last."),
            ("bucket_bouncer", 60, "≥3 forward-and-back bucket boundary crossings — chaotic pattern."),
            ("chronic_late_payer", 45, "Last 12 months all show DPD 1-30, never 0, never above 30."),
            ("first_time_defaulter", 55, "≥18 consecutive clean months before current default."),
            ("no_clear_pattern", 0, "None of the above match."),
            ("not_evaluable", 0, "<6 BOM snapshots — insufficient history."),
        ],
        "sample_size": "≥6 BOM snapshots OR loan_age_months ≥ 6",
    },
    {
        "id": "predictable_payment_cycle",
        "category": "Account trajectory",
        "description": "Detects whether the customer's DPD oscillates in a predictable cycle (e.g. 30→60→0 every 3 months).",
        "states": [
            ("predictable_cycle_with_day", 50, "Cycle confirmed AND payment days cluster tightly (e.g. always around the 28th)."),
            ("predictable_cycle_no_day", 35, "Cycle confirmed but no clear day pattern."),
            ("irregular", 0, "≥6 BOMs but no cycle pattern detected."),
            ("not_evaluable", 0, "<6 BOMs."),
        ],
        "sample_size": "≥6 BOM snapshots, ≥3 reset events with stddev ≤ 1.0",
    },
    {
        "id": "address_intelligence",
        "category": "Geospatial",
        "description": "Compares GPS of last successful field event vs the registered address. Flags address mismatches that may explain contact failures.",
        "states": [
            ("address_offset_detected", 70, "≥500m gap between registered and confirmed location."),
            ("address_likely_correct", 5, "Gap <500m, address appears reliable."),
            ("not_evaluable", 0, "No untrusted-address → positive event pair found."),
        ],
        "sample_size": "≥1 untrusted disposition (DL/NOT_PRESENT/etc.) followed by positive event with valid GPS at same address",
    },
    {
        "id": "legal_stage_indicator",
        "category": "Legal",
        "description": "Tracks legal escalation. v2 — pending schema additions for LEGAL_NOTICE_SENT / CASE_FILED / CASE_CLOSED dispositions.",
        "states": [
            ("legal_case_filed", 90, "Lawsuit on file — visit must be coordinated with legal team."),
            ("legal_notice_sent", 75, "Notice sent — significant escalation."),
            ("legal_case_closed", 20, "Closed case — historical context only."),
            ("no_legal_activity", 0, "No legal events on file."),
            ("not_evaluable", 0, "Schema pending — currently always not_evaluable."),
        ],
        "sample_size": "Awaiting master data definition",
    },
    {
        "id": "dif_death_recurrence",
        "category": "Evasion tactics",
        "description": "Counts Death-in-Family disposition repetitions — suspicious clustering can indicate misuse.",
        "states": [
            ("recurrence_flagged", 65, "≥2 DIF events in last 12 months."),
            ("single_event", 25, "1 DIF event — note in record."),
            ("none", 0, "No DIF events."),
        ],
        "sample_size": "Any history",
    },
    {
        "id": "ptp_fptp_honour_rate",
        "category": "Payment behaviour",
        "description": "Cross-resolved: of all PTP/FPTP records with followup_datetime, what fraction had a SUCCESS payment within 7 days. Drives whether to take future-dated promises at face value.",
        "states": [
            ("reliable", 5, "≥80% honoured — promises convert."),
            ("partial_trust", 30, "50-79% honoured — moderate reliability."),
            ("unreliable", 55, "20-49% honoured — promises don't always land."),
            ("theatre", 70, "<20% honoured — taking another PTP is unlikely to convert."),
            ("not_evaluable", 0, "<3 PTP records with followup."),
        ],
        "sample_size": "≥3 PTP/FPTP records with non-null followup_datetime",
    },
    {
        "id": "account_freeze",
        "category": "Risk / escalation",
        "description": "Merged DNC + DEATH + DIF Permanent. When in any blocking state, this overrides everything — visit not appropriate.",
        "states": [
            ("freeze_death_permanent", 100, "Customer deceased — permanent freeze."),
            ("freeze_dnc_permanent", 100, "Permanent do-not-contact request."),
            ("freeze_dnc_temporary_active", 100, "Temporary DNC, window not expired yet."),
            ("freeze_dif_permanent_recent", 100, "Permanent DIF freeze raised <30 days ago."),
            ("freeze_dif_permanent_aged", 95, "Permanent DIF freeze ≥30 days old — re-verification possible."),
            ("freeze_dnc_expired", 30, "DNC window has expired — soft re-engagement OK."),
            ("no_freeze", 0, "No freeze in history."),
        ],
        "sample_size": "Any history",
    },
    {
        "id": "self_pay_independence",
        "category": "Payment behaviour",
        "description": "Counts SUCCESS payments via CUSTOMER_PORTAL/LINK that had no preceding RPC in the 30 days before. Indicates how independent the customer is.",
        "states": [
            ("low_touch_payer", 25, "≥2 self-pays in 180 days — minimal prompting needed."),
            ("occasional_self_payer", 15, "1 self-pay in 180 days."),
            ("prompt_dependent", 10, "0 self-pays — needs agent prompt."),
            ("not_evaluable", 0, "<6 months of payment history."),
        ],
        "sample_size": "≥6 months of payment history",
    },
    {
        "id": "partial_payment_dependency",
        "category": "Payment behaviour",
        "description": "Of PAID + PARTIAL_PAID dispositions, what fraction were partial. Indicates whether customer can pay in full or struggles with cashflow.",
        "states": [
            ("chronic_partial", 45, "≥60% partial — full EMIs are rare."),
            ("frequent_partial", 30, "30-59% partial — mixed pattern."),
            ("occasional_partial", 5, "<30% partial — mostly pays in full."),
            ("not_evaluable", 0, "<3 paid+partial dispositions."),
        ],
        "sample_size": "≥3 PAID or PARTIAL_PAID dispositions",
    },
    {
        "id": "payment_day_pattern",
        "category": "Payment behaviour",
        "description": "Day-of-month bucket where the customer typically pays. Useful for timing follow-ups and visits.",
        "states": [
            ("pays_early_month", 25, "≥70% of payments in days 1-7 — likely salaried, paid on 1st."),
            ("pays_mid_month", 25, "≥70% in days 8-24 — variable salary or business income."),
            ("pays_end_month", 25, "≥70% in days 25-31 — last-minute payer."),
            ("no_clear_pattern", 0, "<70% in any single bucket."),
            ("not_evaluable", 0, "<4 successful payments."),
        ],
        "sample_size": "≥4 SUCCESS payments",
    },
    {
        "id": "utp_recurrence_pattern",
        "category": "Intent signals",
        "description": "Distribution of Unable-to-Pay reasons. Same reason repeated suggests genuine hardship; rotating reasons suggest opportunism.",
        "states": [
            ("consistent_hardship", 55, "Top reason ≥60% — likely genuine ongoing issue."),
            ("rotating_reasons", 50, "No reason >40% — different excuse each time."),
            ("mixed", 30, "Top reason 40-59% — somewhere in between."),
            ("not_evaluable", 0, "<3 UTP dispositions."),
        ],
        "sample_size": "≥3 UTP dispositions",
    },
    {
        "id": "settlement_intent_score",
        "category": "Intent signals",
        "description": "Counts SETTLEMENT_REQUEST + FORECLOSURE_REQUEST + LOAN_CANCELLATION dispositions. Customer is signalling exit — changes what success means for the visit.",
        "states": [
            ("high_intent", 75, "≥2 settlement events on file."),
            ("warm_intent", 50, "1 settlement event."),
            ("no_intent", 0, "No settlement events."),
        ],
        "sample_size": "Any history",
    },
    {
        "id": "doubtful_sentiment_trail",
        "category": "Intent signals",
        "description": "Counts DOUBTFUL dispositions in last 60 days. Recent agent observations of evasive customer behaviour.",
        "states": [
            ("repeated_doubt", 60, "≥2 doubtful flags in past 60 days."),
            ("single_doubt", 30, "1 doubtful flag in past 60 days."),
            ("none", 0, "No recent doubtful flags."),
        ],
        "sample_size": "Any history (60-day window)",
    },
    {
        "id": "best_contact_hour",
        "category": "Contactability",
        "description": "Slides a 2-hour window across RPC dispositions to find when the customer is most reachable. Useful for timing the next call before the visit.",
        "states": [
            ("window_identified", 25, "Best 2-hour window holds ≥3 RPCs and >40% of total."),
            ("no_clear_window", 0, "RPCs spread evenly across hours."),
            ("not_evaluable", 0, "<5 RPC dispositions."),
        ],
        "sample_size": "≥5 RPC dispositions",
    },
    {
        "id": "tpc_ratio",
        "category": "Contactability",
        "description": "Of all contact-type dispositions, what fraction are Third-Party Contact. High TPC means the customer is reachable via family but not directly.",
        "states": [
            ("high_tpc", 50, "≥50% of contacts via third party — customer dodging direct contact."),
            ("moderate_tpc", 25, "25-49% via TPC."),
            ("low_tpc", 5, "<25% via TPC."),
            ("not_evaluable", 0, "<5 contact attempts."),
        ],
        "sample_size": "≥5 contact attempts (RPC+NC+TPC+WN)",
    },
    {
        "id": "no_contact_rate",
        "category": "Contactability",
        "description": "Of all contact-type dispositions, what fraction were No-Contact. Distinct from TPC ratio — measures unreachability.",
        "states": [
            ("ghost_account", 65, "≥70% NC — customer essentially unreachable."),
            ("mostly_unreachable", 40, "40-69% NC — significant contact failure."),
            ("reachable", 5, "<40% NC — fine."),
            ("not_evaluable", 0, "<5 contact attempts."),
        ],
        "sample_size": "≥5 contact attempts",
    },
    {
        "id": "refusal_escalation_flag",
        "category": "Risk / escalation",
        "description": "Visit risk indicator. Counts RTP (refuse-to-pay) and FRAUD dispositions. Critical state means the visit needs supervisor awareness.",
        "states": [
            ("critical", 95, "Fraud on file OR ≥2 RTPs — meaningful refusal pattern."),
            ("caution", 50, "1 RTP, 0 FRAUD — note but not severe."),
            ("clear", 0, "No refusal or fraud history."),
        ],
        "sample_size": "Any history",
    },
]

AMPLIFIER_RULES = {
    "Co-occurrence (between indicators)": [
        ("ptp_fptp_honour_rate ⬆ amplified",
         "If state ∈ [unreliable, theatre] AND refusal_escalation_flag = critical",
         "Combined with refusal flags, the gap isn't just unreliability."),
        ("settlement_intent_score ⬆ amplified",
         "If state = high_intent AND defaulter_type ∈ [deteriorating, bucket_bouncer]",
         "Settlement intent reads stronger given the deteriorating trajectory."),
        ("address_intelligence ⬆ amplified",
         "If state = address_offset_detected AND no_contact_rate ∈ [ghost, mostly_unreachable]",
         "Address mismatch may explain the contact failures."),
        ("tpc_ratio ⬆ amplified",
         "If state = high_tpc AND no_contact_rate ∈ [ghost, mostly_unreachable]",
         "Customer is reachable through family but not directly — likely active dodging."),
        ("doubtful_sentiment_trail ⬆ amplified",
         "If state = repeated_doubt AND ptp_fptp_honour_rate ∈ [unreliable, theatre]",
         "Recent evasiveness compounds the broken-promise pattern."),
        ("utp_recurrence_pattern ⬇ muted",
         "If state = rotating_reasons AND refusal=clear AND ptp_fptp_honour_rate ∈ [reliable, partial_trust]",
         "Rotating reasons less alarming when other signals are clean."),
        ("language_barrier_persistence ⬆ amplified",
         "If state = persistent_barrier AND no_contact_rate ∈ [ghost, mostly_unreachable]",
         "Language gap may be why contact has failed."),
        ("predictable_payment_cycle ⬇ muted",
         "If state ∈ [cycle_with_day, cycle_no_day] AND refusal_escalation_flag = critical",
         "Cycle insight less useful when refusal escalation overshadows."),
    ],
    "Recency / pattern (within indicator)": [
        ("settlement_intent_score ⬆ amplified",
         "If high_intent AND multiple events span >60 days",
         "Settlement intent has been sustained across multiple months, not a one-off ask."),
        ("settlement_intent_score ⬇ muted",
         "If warm_intent AND latest event within 14 days",
         "Single recent settlement mention — may be one conversation."),
        ("doubtful_sentiment_trail ⬆ amplified",
         "If repeated_doubt AND latest_date within 7 days",
         "Most recent doubtful flag is fresh — pattern is active right now."),
        ("dif_death_recurrence ⬆ amplified",
         "If recurrence_flagged AND any 2 DIF dates within 30 days",
         "Bereavement claims cluster suspiciously close together."),
        ("ptp_fptp_honour_rate ⬆ amplified",
         "If state ∈ [unreliable, theatre] AND total ≥ 10",
         "Pattern holds across many promises, not a small sample."),
        ("ptp_fptp_honour_rate ⬇ muted",
         "If state ∈ [unreliable, theatre] AND total IN [3, 4]",
         "Sample size is small — pattern may not hold."),
        ("refusal_escalation_flag ⬆ amplified",
         "If critical AND latest escalation within 30 days",
         "Most recent refusal/fraud is fresh — risk is current, not historical."),
        ("refusal_escalation_flag ⬇ muted",
         "If critical AND latest escalation older than 180 days",
         "Most recent escalation is old — situation may have shifted."),
        ("address_intelligence ⬆ amplified",
         "If address_offset_detected AND distance_m ≥ 1000",
         "Customer's actual location is over a kilometre from registered address."),
    ],
}


with tab_triggers:
    st.markdown("##### 17 Indicator Reference Catalogue")
    st.caption(
        "All triggers, states, severity values, and rules — for quick reference while iterating on prompts. "
        "The 'Currently firing' panel at the bottom shows which indicators are active for the selected account right now."
    )

    # Search/filter
    cf1, cf2 = st.columns([3, 2])
    with cf1:
        search_q = st.text_input("🔎 Search indicators",
                                  placeholder="e.g. 'ptp', 'freeze', 'address', 'theatre'",
                                  label_visibility="collapsed")
    with cf2:
        category_filter = st.selectbox(
            "Filter by category",
            ["All"] + sorted(set(ind["category"] for ind in INDICATOR_CATALOGUE)),
            label_visibility="collapsed",
        )

    # Filtered indicator list
    filtered = INDICATOR_CATALOGUE
    if search_q:
        q = search_q.lower()
        filtered = [
            ind for ind in filtered
            if q in ind["id"].lower()
            or q in ind["description"].lower()
            or any(q in s[0].lower() for s in ind["states"])
        ]
    if category_filter != "All":
        filtered = [ind for ind in filtered if ind["category"] == category_filter]

    st.caption(f"Showing {len(filtered)} of {len(INDICATOR_CATALOGUE)} indicators")

    # Render each indicator
    for ind in filtered:
        with st.expander(f"**{ind['id']}**  ·  *{ind['category']}*", expanded=False):
            st.markdown(f"_{ind['description']}_")
            st.caption(f"**Sample size required:** {ind['sample_size']}")

            # States table
            st.markdown("**States and severity (base values):**")
            states_df = pd.DataFrame(
                [(s[0], s[1], s[2]) for s in ind["states"]],
                columns=["State", "Base severity", "When it fires"],
            )
            st.dataframe(states_df, use_container_width=True, hide_index=True)

    st.divider()

    # Amplifier rules
    st.markdown("##### Amplifier rules (modify base severity by ±15)")
    for rule_type, rules in AMPLIFIER_RULES.items():
        with st.expander(f"**{rule_type}** ({len(rules)} rules)", expanded=False):
            rules_df = pd.DataFrame(
                rules,
                columns=["Indicator + direction", "Trigger condition", "Reason text"],
            )
            st.dataframe(rules_df, use_container_width=True, hide_index=True)

    st.markdown("**Freeze override (special rule):** if account_freeze fires in any blocking state "
                "(DNC permanent, DNC temp active, DEATH permanent, DIF permanent), severity "
                "stays at 95-100, ALL other fired indicators get muted, and ranked_top contains "
                "only `account_freeze`. The writer should output a single 'do not visit' action.")

    st.divider()

    # Currently firing — based on session state
    st.markdown("##### Currently firing on this account")
    st.caption(f"Account: **{selected_case}**")

    if st.session_state.stage1_output and "indicators" in st.session_state.stage1_output:
        analyser = st.session_state.stage1_output
        ranked = analyser.get("ranked_top", [])
        indicators_out = analyser.get("indicators", {})

        if ranked:
            firing_rows = []
            for ind_id in ranked:
                detail = indicators_out.get(ind_id, {})
                firing_rows.append({
                    "Rank": len(firing_rows) + 1,
                    "Indicator": ind_id,
                    "State": detail.get("state", "?"),
                    "Severity": detail.get("severity", 0),
                    "Amplifier": detail.get("amplifier", "neutral"),
                    "Reason": detail.get("amplifier_reason", "") or "—",
                })
            st.dataframe(pd.DataFrame(firing_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Stage 1 ran but no indicators are currently fired (all in default states).")
    else:
        st.info("Run Stage 1 in the Pipeline tab to see which indicators fire for this account.")
