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


def call_claude(system_prompt, user_message, model, api_key, max_tokens=4096):
    """Call Claude and return (parsed_json, meta). On error, parsed_json contains error key."""
    client = Anthropic(api_key=api_key)
    t0 = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        elapsed = time.time() - t0
        raw = response.content[0].text.strip()
        cleaned = strip_fences(raw)
        try:
            parsed = json.loads(cleaned)
            return parsed, {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_s": round(elapsed, 2),
                "model": model,
                "error": None,
            }
        except json.JSONDecodeError as e:
            return {"error": "json_parse_failed", "raw": raw, "exception": str(e)}, {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_s": round(elapsed, 2),
                "model": model,
                "error": "parse",
            }
    except Exception as e:
        return {"error": "api_call_failed", "exception": str(e)}, {
            "latency_s": round(time.time() - t0, 2),
            "error": "api",
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
st.caption("Tweak prompts. Tweak data. See what comes out. Iterate till the briefing reads right.")

bar = st.columns([2.2, 1.4, 1.4, 1.4, 1.6])

with bar[0]:
    case_options = sorted(disp_df["case_ref_number"].unique().tolist())
    selected_case = st.selectbox("Account", case_options, key="account_picker",
                                  format_func=lambda x: f"{x.replace('CR_', '').replace('_', ' · ')}")

with bar[1]:
    today_input = st.date_input("Today", value=date(2026, 4, 30), key="today_input")

with bar[2]:
    model_choice = st.selectbox(
        "Model",
        ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        index=1,
        key="model_choice",
    )

with bar[3]:
    emi_amount = st.number_input("EMI (₹)", value=12300, step=500, key="emi_input")

with bar[4]:
    api_key = st.text_input(
        "API key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        key="api_key_input",
        help="Or set ANTHROPIC_API_KEY env var",
    )

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

tab_field, tab_pipeline, tab_prompts, tab_input = st.tabs([
    "📱 Field App", "⚡ Pipeline", "✏️ Prompts", "📊 Input Data"
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
            if not api_key:
                st.error("API key needed.")
            else:
                with st.spinner(f"Running analyser ({model_choice})..."):
                    user_msg = build_analyser_user_msg(current_features)
                    output, meta = call_claude(
                        st.session_state.analyser_prompt,
                        user_msg,
                        model_choice,
                        api_key,
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
            if not api_key:
                st.error("API key needed.")
            else:
                with st.spinner(f"Running writer ({model_choice})..."):
                    user_msg = build_writer_user_msg(st.session_state.stage1_output)
                    output, meta = call_claude(
                        st.session_state.writer_prompt,
                        user_msg,
                        model_choice,
                        api_key,
                        max_tokens=2048,
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
