"""
Pure Python rule-based evaluator. Mirrors what Claude is asked to do, but
deterministically. Useful for:
  1. Validating Claude's state selection (compare side-by-side)
  2. A zero-cost fallback if Claude API is unavailable
  3. Regression-testing the spec itself

This file does NOT generate the agent_note text — that's Claude's job. Here we
only return state + evidence.

Usage:
    from rules_evaluator import evaluate_rules
    result = evaluate_rules(features_dict)
"""

import numpy as np


# Bucket severity ordering — matches features.py
BUCKET_ORDER = ["X1", "X2", "X3", "B1", "B2", "B3", "B4", "B5", "NPA", "WO"]


def evaluate_rules(features):
    """Apply all 17 indicator rules deterministically. Returns dict matching
    Claude's output shape but with empty agent_note (caller fills if needed)."""
    indicators = {
        "defaulter_type_classifier": _defaulter_type(features),
        "predictable_payment_cycle": _predictable_cycle(features),
        "address_intelligence": _address_intelligence(features),
        "legal_stage_indicator": _legal_stage(features),
        "dif_death_recurrence": _dif_recurrence(features),
        "ptp_fptp_honour_rate": _ptp_honour(features),
        "account_freeze": _account_freeze(features),
        "self_pay_independence": _self_pay(features),
        "partial_payment_dependency": _partial_payment(features),
        "payment_day_pattern": _payment_day_pattern(features),
        "utp_recurrence_pattern": _utp_recurrence(features),
        "settlement_intent_score": _settlement_intent(features),
        "doubtful_sentiment_trail": _doubtful_trail(features),
        "best_contact_hour": _best_contact_hour(features),
        "tpc_ratio": _tpc_ratio(features),
        "no_contact_rate": _no_contact_rate(features),
        "refusal_escalation_flag": _refusal_flag(features),
    }

    return {
        "case_ref_number": features.get("case_ref_number"),
        "evaluated_at": features.get("today"),
        "indicators": indicators,
        "priority_alerts": _priority_alerts(indicators),
    }


def _state(state, evidence=None):
    return {"state": state, "evidence": evidence or {}, "agent_note": ""}


# ─── 1. Defaulter type ────────────────────────────────────────────────────────
def _defaulter_type(f):
    bom = f.get("bom", {})
    if bom.get("bom_count", 0) < 6 and bom.get("loan_age_months", 0) < 6:
        return _state("not_evaluable", {"reason": "insufficient_data"})

    # a) serial early EMI defaulter
    if bom.get("first_3_months_any_dpd_gt_0"):
        return _state("serial_early_emi_defaulter",
                     {"first_3_months_max_dpd": bom.get("first_3_months_max_dpd")})

    # b) deteriorating defaulter
    cycles = bom.get("default_cycles", [])
    if len(cycles) >= 3:
        peaks = [c.get("peak_bucket") for c in cycles if c.get("peak_bucket") in BUCKET_ORDER]
        if len(peaks) >= 3:
            indices = [BUCKET_ORDER.index(p) for p in peaks]
            if all(indices[i] > indices[i - 1] for i in range(1, len(indices))):
                return _state("deteriorating_defaulter",
                             {"cycle_peaks": peaks})

    # c) bucket bouncer
    if bom.get("bucket_boundary_crossings", 0) >= 3:
        return _state("bucket_bouncer",
                     {"crossings": bom.get("bucket_boundary_crossings")})

    # d) chronic late payer
    if bom.get("last_12_chronic_late_pattern"):
        return _state("chronic_late_payer", {})

    # e) first-time defaulter
    clean = bom.get("consecutive_clean_months_before_current_default")
    if clean is not None and clean >= 18:
        return _state("first_time_defaulter", {"clean_months_before": clean})

    return _state("no_clear_pattern", {})


# ─── 2. Predictable payment cycle ─────────────────────────────────────────────
def _predictable_cycle(f):
    bom = f.get("bom", {})
    if bom.get("bom_count", 0) < 6:
        return _state("not_evaluable", {"reason": "insufficient_data"})

    if bom.get("reset_event_count", 0) < 3:
        return _state("irregular", {"reset_event_count": bom.get("reset_event_count")})

    stddev = bom.get("reset_interval_stddev")
    if stddev is None or stddev > 1.0:
        return _state("irregular",
                     {"reset_event_count": bom.get("reset_event_count"),
                      "interval_stddev": stddev})

    # Check for day-of-month dominance in payment_day_distribution
    pay = f.get("payments", {})
    day_dist = pay.get("payment_day_distribution", {})
    if day_dist:
        days = []
        for d, c in day_dist.items():
            days.extend([int(d)] * int(c))
        if len(days) >= 3 and np.std(days) <= 4:
            return _state("predictable_cycle_with_day", {
                "interval_months": int(round(bom.get("reset_interval_mean", 0))),
                "dominant_day_of_month": int(round(np.mean(days))),
            })

    return _state("predictable_cycle_no_day", {
        "interval_months": int(round(bom.get("reset_interval_mean", 0))),
    })


# ─── 3. Address intelligence ──────────────────────────────────────────────────
def _address_intelligence(f):
    resolved = f.get("address_intelligence_resolved")
    if not resolved:
        return _state("not_evaluable", {"reason": "no_pair_found"})

    distance = resolved.get("distance_m")
    if distance is None:
        return _state("not_evaluable", {"reason": "gps_missing"})

    if distance < 500:
        return _state("address_likely_correct", {"distance_m": distance})
    return _state("address_offset_detected", {
        "distance_m": distance,
        "corrected_lat": resolved["latest_pair"]["positive_lat"],
        "corrected_lng": resolved["latest_pair"]["positive_lng"],
    })


# ─── 4. Legal stage ───────────────────────────────────────────────────────────
def _legal_stage(f):
    return _state("not_evaluable", {"reason": "schema_pending"})


# ─── 5. DIF / Death recurrence ────────────────────────────────────────────────
def _dif_recurrence(f):
    n = f.get("dispositions", {}).get("dif_count_12mo", 0)
    if n >= 2:
        return _state("recurrence_flagged", {"dif_count_12mo": n})
    if n == 1:
        return _state("single_event", {"dif_count_12mo": n})
    return _state("none", {})


# ─── 6. PTP honour rate ───────────────────────────────────────────────────────
def _ptp_honour(f):
    h = f.get("ptp_honour", {})
    total = h.get("total", 0)
    if total < 3:
        return _state("not_evaluable", {"total": total})

    pct = h.get("honour_rate_pct", 0)
    evidence = {"total": total, "honoured_count": h.get("honoured_count"), "honour_rate_pct": pct}

    if pct >= 80:
        return _state("reliable", evidence)
    if pct >= 50:
        return _state("partial_trust", evidence)
    if pct >= 20:
        return _state("unreliable", evidence)
    return _state("theatre", evidence)


# ─── 7. Account freeze ────────────────────────────────────────────────────────
def _account_freeze(f):
    fz = f.get("dispositions", {}).get("freeze_latest")
    if not fz:
        return _state("no_freeze", {})

    disp_code = fz.get("disposition")
    ftype = fz.get("type")
    age = fz.get("age_days")

    if disp_code == "DEATH" and ftype == "Permanent":
        return _state("freeze_death_permanent", fz)
    if disp_code == "DIF" and ftype == "Permanent":
        if age is not None and age < 30:
            return _state("freeze_dif_permanent_recent", fz)
        return _state("freeze_dif_permanent_aged", fz)
    if disp_code == "DNC":
        if ftype == "Permanent":
            return _state("freeze_dnc_permanent", fz)
        # Temporary
        from datetime import datetime
        followup = fz.get("followup_datetime")
        today = f.get("today")
        if followup and today:
            if followup > today:
                return _state("freeze_dnc_temporary_active", fz)
            return _state("freeze_dnc_expired", fz)
        return _state("freeze_dnc_temporary_active", fz)  # safer default

    return _state("no_freeze", {})


# ─── 8. Self-pay independence ────────────────────────────────────────────────
def _self_pay(f):
    pay = f.get("payments", {})
    if pay.get("payment_history_months", 0) < 6:
        return _state("not_evaluable", {"reason": "insufficient_history"})

    n = pay.get("self_pay_count_180d", 0)
    if n >= 2:
        return _state("low_touch_payer", {"self_pay_count_180d": n})
    if n == 1:
        return _state("occasional_self_payer", {"self_pay_count_180d": n})
    return _state("prompt_dependent", {"self_pay_count_180d": n})


# ─── 9. Partial payment dependency ───────────────────────────────────────────
def _partial_payment(f):
    d = f.get("dispositions", {})
    paid = d.get("paid_count", 0)
    partial = d.get("partial_paid_count", 0)
    total = paid + partial

    if total < 3:
        return _state("not_evaluable", {"total_payment_dispositions": total})

    pct = int(round(100 * partial / total))
    evidence = {"partial_count": partial, "paid_count": paid, "partial_pct": pct}

    if pct >= 60:
        return _state("chronic_partial", evidence)
    if pct >= 30:
        return _state("frequent_partial", evidence)
    return _state("occasional_partial", evidence)


# ─── 10. Payment day pattern ─────────────────────────────────────────────────
def _payment_day_pattern(f):
    pay = f.get("payments", {})
    success = pay.get("success_count", 0)
    if success < 4:
        return _state("not_evaluable", {"success_count": success})

    buckets = pay.get("payment_day_buckets", {})
    if not buckets or sum(buckets.values()) == 0:
        return _state("no_clear_pattern", {})

    top_bucket = max(buckets, key=buckets.get)
    top_pct = buckets[top_bucket] / success
    evidence = {"buckets": buckets, "dominant_bucket": top_bucket,
                "dominant_pct": int(round(100 * top_pct))}

    if top_pct < 0.70:
        return _state("no_clear_pattern", evidence)

    if top_bucket == "early_month":
        return _state("pays_early_month", evidence)
    if top_bucket in ("mid_month_early", "mid_month_late"):
        return _state("pays_mid_month", evidence)
    if top_bucket == "end_month":
        return _state("pays_end_month", evidence)
    return _state("no_clear_pattern", evidence)


# ─── 11. UTP recurrence ──────────────────────────────────────────────────────
def _utp_recurrence(f):
    d = f.get("dispositions", {})
    total = d.get("utp_total", 0)
    if total < 3:
        return _state("not_evaluable", {"utp_total": total})

    by_reason = d.get("utp_by_reason", {})
    if not by_reason:
        return _state("not_evaluable", {"reason": "no_reason_data"})

    top_reason = max(by_reason, key=by_reason.get)
    top_pct = by_reason[top_reason] / total
    evidence = {"total_utp": total, "by_reason": by_reason,
                "dominant_reason": top_reason, "dominant_pct": int(round(100 * top_pct))}

    if top_pct >= 0.60:
        return _state("consistent_hardship", evidence)
    if top_pct >= 0.40:
        return _state("mixed", evidence)
    return _state("rotating_reasons", evidence)


# ─── 12. Settlement intent ───────────────────────────────────────────────────
def _settlement_intent(f):
    d = f.get("dispositions", {})
    n = d.get("settlement_total", 0)
    evidence = {
        "settlement_total": n,
        "latest_event": d.get("settlement_latest_disposition"),
        "latest_date": d.get("settlement_latest_date"),
    }
    if n >= 2:
        return _state("high_intent", evidence)
    if n == 1:
        return _state("warm_intent", evidence)
    return _state("no_intent", evidence)


# ─── 13. Doubtful sentiment trail ────────────────────────────────────────────
def _doubtful_trail(f):
    d = f.get("dispositions", {})
    n = d.get("doubtful_count_60d", 0)
    evidence = {
        "count_60d": n,
        "latest_comment": d.get("doubtful_latest_comment"),
        "latest_date": d.get("doubtful_latest_date"),
    }
    if n >= 2:
        return _state("repeated_doubt", evidence)
    if n == 1:
        return _state("single_doubt", evidence)
    return _state("none", evidence)


# ─── 14. Best contact hour ───────────────────────────────────────────────────
def _best_contact_hour(f):
    d = f.get("dispositions", {})
    rpc_count = d.get("rpc_count", 0)
    if rpc_count < 5:
        return _state("not_evaluable", {"rpc_count": rpc_count})

    dist = d.get("rpc_hour_distribution", {})
    # Convert keys to int
    dist = {int(k): int(v) for k, v in dist.items()}

    # Slide 2-hour window
    best_window = None
    best_count = 0
    for h in range(0, 23):
        window_count = dist.get(h, 0) + dist.get(h + 1, 0)
        if window_count > best_count:
            best_count = window_count
            best_window = (h, h + 2)

    if best_window and best_count >= 3 and best_count / rpc_count > 0.40:
        return _state("window_identified", {
            "start_hour": best_window[0],
            "end_hour": best_window[1],
            "rpc_count_in_window": best_count,
            "total_rpc": rpc_count,
        })
    return _state("no_clear_window", {"total_rpc": rpc_count})


# ─── 15. TPC ratio ───────────────────────────────────────────────────────────
def _tpc_ratio(f):
    d = f.get("dispositions", {})
    total = d.get("contact_type_total", 0)
    if total < 5:
        return _state("not_evaluable", {"total": total})

    tpc = d.get("tpc_count", 0)
    pct = int(round(100 * tpc / total))
    evidence = {"tpc_count": tpc, "total_attempts": total, "tpc_pct": pct}

    if pct >= 50:
        return _state("high_tpc", evidence)
    if pct >= 25:
        return _state("moderate_tpc", evidence)
    return _state("low_tpc", evidence)


# ─── 16. No-contact rate ─────────────────────────────────────────────────────
def _no_contact_rate(f):
    d = f.get("dispositions", {})
    total = d.get("contact_type_total", 0)
    if total < 5:
        return _state("not_evaluable", {"total": total})

    nc = d.get("nc_count", 0)
    pct = int(round(100 * nc / total))
    evidence = {"nc_count": nc, "total_attempts": total, "nc_pct": pct}

    if pct >= 70:
        return _state("ghost_account", evidence)
    if pct >= 40:
        return _state("mostly_unreachable", evidence)
    return _state("reachable", evidence)


# ─── 17. Refusal escalation ──────────────────────────────────────────────────
def _refusal_flag(f):
    d = f.get("dispositions", {})
    rtp = d.get("rtp_count", 0)
    fraud = d.get("fraud_count", 0)
    evidence = {"rtp_count": rtp, "fraud_count": fraud}

    if fraud >= 1 or rtp >= 2:
        return _state("critical", evidence)
    if rtp == 1:
        return _state("caution", evidence)
    return _state("clear", evidence)


# ─── Priority alerts ─────────────────────────────────────────────────────────
def _priority_alerts(indicators):
    alerts = []
    blocking_freezes = {"freeze_dnc_permanent", "freeze_dnc_temporary_active",
                        "freeze_death_permanent", "freeze_dif_permanent_recent",
                        "freeze_dif_permanent_aged"}

    if indicators["account_freeze"]["state"] in blocking_freezes:
        alerts.append("account_freeze")
    if indicators["refusal_escalation_flag"]["state"] == "critical":
        alerts.append("refusal_escalation_flag")
    if indicators["legal_stage_indicator"]["state"] in ("legal_case_filed", "legal_notice_sent"):
        alerts.append("legal_stage_indicator")
    if indicators["settlement_intent_score"]["state"] == "high_intent":
        alerts.append("settlement_intent_score")
    if indicators["defaulter_type_classifier"]["state"] in (
        "deteriorating_defaulter", "serial_early_emi_defaulter"
    ):
        alerts.append("defaulter_type_classifier")
    if indicators["address_intelligence"]["state"] == "address_offset_detected":
        alerts.append("address_intelligence")
    if indicators["ptp_fptp_honour_rate"]["state"] == "theatre":
        alerts.append("ptp_fptp_honour_rate")

    return alerts[:3]
