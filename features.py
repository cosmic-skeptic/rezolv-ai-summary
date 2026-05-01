"""
Feature extractor for Rezolv Intelligence prototype.

Takes raw CSVs (dispositions, payments, bom_snapshots) and produces a compact
features dict per case_ref_number. Everything the 17 indicators need is in
this dict — Stage 2 (LLM evaluation) operates only on this output, never on
raw data.

Usage:
    from features import build_features
    features = build_features(case_ref="CR_SAMPLE_001",
                              dispositions_df=...,
                              payments_df=...,
                              bom_df=...,
                              registered_lat=..., registered_lng=...,
                              today="2026-04-30",
                              emi_amount=12300)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from math import radians, sin, cos, sqrt, atan2


# ─────────────────────────────────────────────────────────────────────────────
# Constants — disposition code groupings
# ─────────────────────────────────────────────────────────────────────────────

UNTRUSTED_ADDRESS_CODES = {"DL", "NOT_PRESENT", "ANF", "WA", "SA", "SHIFTED"}
POSITIVE_EVENT_CODES = {"PTP", "FPTP", "PAID", "PARTIAL_PAID", "NACH_REP", "RP"}
SETTLEMENT_CODES = {"SETTLEMENT_REQUEST", "FORECLOSURE_REQUEST", "LOAN_CANCELLATION"}
SYSTEM_CODES = {"PPS", "PPA", "PPR", "CP", "DCR"}
FREEZE_CODES = {"DNC", "DEATH", "DIF"}
PRODUCTIVE_FIELD_CODES = {"PTP", "FPTP", "PAID", "PARTIAL_PAID", "NACH_REP"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def haversine_m(lat1, lng1, lat2, lng2):
    """Distance between two GPS points in metres."""
    if any(pd.isna(x) for x in [lat1, lng1, lat2, lng2]):
        return None
    R = 6371000  # metres
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return int(R * c)


def parse_dt(val):
    """Parse a datetime field, return None if unparseable."""
    if pd.isna(val) or val == "" or val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return pd.to_datetime(val).to_pydatetime()
    except Exception:
        return None


def safe_str(val, default=""):
    if pd.isna(val) or val is None:
        return default
    return str(val)


def to_serializable(obj):
    """Convert numpy/pandas types to plain Python for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Disposition features
# ─────────────────────────────────────────────────────────────────────────────

def disposition_features(disp, today):
    """Compute all disposition-derived features."""
    f = {}

    if disp.empty:
        return _empty_disposition_features()

    # Sort chronologically — matters for "latest" lookups
    disp = disp.sort_values("created_at").reset_index(drop=True)

    # ── Overall counts ──
    f["total_dispositions"] = len(disp)
    f["total_attempts_excluding_system"] = int(
        (~disp["disposition"].isin(SYSTEM_CODES)).sum()
    )

    # ── Contact type counts (for TPC ratio + No-contact rate cards) ──
    ctype = disp.get("contact_type", pd.Series(dtype=str))
    f["rpc_count"] = int((ctype == "RPC").sum())
    f["nc_count"] = int((ctype == "NC").sum())
    f["tpc_count"] = int((ctype == "TPC").sum())
    f["wn_count"] = int((ctype == "WN").sum())
    f["contact_type_total"] = int(ctype.isin(["RPC", "NC", "TPC", "WN"]).sum())

    # ── Disposition-specific counts ──
    code_counts = disp["disposition"].value_counts().to_dict()
    f["ptp_count"] = code_counts.get("PTP", 0)
    f["fptp_count"] = code_counts.get("FPTP", 0)
    f["bptp_count"] = code_counts.get("BPTP", 0)
    f["paid_count"] = code_counts.get("PAID", 0)
    f["partial_paid_count"] = code_counts.get("PARTIAL_PAID", 0)
    f["rtp_count"] = code_counts.get("RTP", 0)
    f["fraud_count"] = code_counts.get("FRAUD", 0)
    f["lgt_count"] = code_counts.get("LGT", 0)
    f["lb_count"] = code_counts.get("LB", 0)
    f["lm_count"] = code_counts.get("LM", 0)
    f["doubtful_count"] = code_counts.get("DOUBTFUL", 0)
    f["dif_count"] = code_counts.get("DIF", 0)

    # ── Settlement intent (sum of three codes) ──
    f["settlement_request_count"] = code_counts.get("SETTLEMENT_REQUEST", 0)
    f["foreclosure_request_count"] = code_counts.get("FORECLOSURE_REQUEST", 0)
    f["loan_cancellation_count"] = code_counts.get("LOAN_CANCELLATION", 0)
    settlement_disp = disp[disp["disposition"].isin(SETTLEMENT_CODES)]
    f["settlement_total"] = len(settlement_disp)
    if len(settlement_disp) > 0:
        latest = settlement_disp.iloc[-1]
        f["settlement_latest_date"] = parse_dt(latest["created_at"]).isoformat() if parse_dt(latest["created_at"]) else None
        f["settlement_latest_disposition"] = latest["disposition"]
    else:
        f["settlement_latest_date"] = None
        f["settlement_latest_disposition"] = None

    # ── DOUBTFUL trail (last 60 days) ──
    cutoff_60d = today - timedelta(days=60)
    doubtful = disp[
        (disp["disposition"] == "DOUBTFUL")
        & (pd.to_datetime(disp["created_at"]) >= cutoff_60d)
    ]
    f["doubtful_count_60d"] = len(doubtful)
    if len(doubtful) > 0:
        latest = doubtful.iloc[-1]
        f["doubtful_latest_comment"] = safe_str(latest.get("comment", ""))[:200]
        f["doubtful_latest_date"] = parse_dt(latest["created_at"]).isoformat()
    else:
        f["doubtful_latest_comment"] = None
        f["doubtful_latest_date"] = None

    # ── DIF recurrence (last 365 days) ──
    cutoff_365d = today - timedelta(days=365)
    dif_recent = disp[
        (disp["disposition"] == "DIF")
        & (pd.to_datetime(disp["created_at"]) >= cutoff_365d)
    ]
    f["dif_count_12mo"] = len(dif_recent)
    f["dif_dates_12mo"] = [parse_dt(d).isoformat() for d in dif_recent["created_at"] if parse_dt(d)]

    # ── UTP recurrence ──
    utp = disp[disp["disposition"] == "UTP"]
    f["utp_total"] = len(utp)
    if len(utp) > 0:
        f["utp_by_reason"] = utp["sub_disposition"].value_counts().to_dict()
    else:
        f["utp_by_reason"] = {}

    # ── Account freeze (latest among DNC/DEATH/DIF) ──
    freeze = disp[disp["disposition"].isin(FREEZE_CODES)].sort_values("created_at")
    if len(freeze) > 0:
        latest = freeze.iloc[-1]
        raised = parse_dt(latest["created_at"])
        # Type lives in sub_disposition; some clients put it in comment
        sub = safe_str(latest.get("sub_disposition", "")).lower()
        comment = safe_str(latest.get("comment", "")).lower()
        if "permanent" in sub or "permanent" in comment:
            freeze_type = "Permanent"
        elif "temporary" in sub or "temporary" in comment:
            freeze_type = "Temporary"
        else:
            freeze_type = None  # ambiguous
        followup = parse_dt(latest.get("followup_datetime"))
        f["freeze_latest"] = {
            "disposition": latest["disposition"],
            "raised_on": raised.isoformat() if raised else None,
            "type": freeze_type,
            "followup_datetime": followup.isoformat() if followup else None,
            "age_days": (today - raised).days if raised else None,
        }
    else:
        f["freeze_latest"] = None

    # ── RPC hour distribution (for best contact hour) ──
    rpc = disp[disp.get("contact_type", "") == "RPC"].copy()
    if len(rpc) > 0:
        rpc["hour"] = pd.to_datetime(rpc["created_at"]).dt.hour
        f["rpc_hour_distribution"] = rpc["hour"].value_counts().sort_index().to_dict()
    else:
        f["rpc_hour_distribution"] = {}

    # ── Untrusted address → positive event pairs (for address intelligence) ──
    f["address_intelligence_pairs"] = _address_intelligence_pairs(disp)

    # ── PTP/FPTP records with followup_datetime (used for honour rate) ──
    ptp_records = disp[
        disp["disposition"].isin(["PTP", "FPTP"]) & disp["followup_datetime"].notna()
    ]
    f["ptp_fptp_records"] = [
        {
            "disposition_ref": r.get("disposition_ref_number"),
            "followup_datetime": parse_dt(r["followup_datetime"]).isoformat()
            if parse_dt(r["followup_datetime"]) else None,
        }
        for _, r in ptp_records.iterrows()
    ]
    f["ptp_fptp_count_with_followup"] = len(ptp_records)

    # ── Field visit efficiency (computed but not displayed to agent) ──
    field = disp[disp["source"] == "FIELD"]
    f["field_total"] = len(field)
    f["field_productive"] = int(field["disposition"].isin(PRODUCTIVE_FIELD_CODES).sum())

    return f


def _address_intelligence_pairs(disp):
    """For each untrusted-address disposition, find subsequent positive events
    at same address_ref_number with valid GPS."""
    if "address_ref_number" not in disp.columns:
        return []

    untrusted = disp[disp["disposition"].isin(UNTRUSTED_ADDRESS_CODES)]
    pairs = []

    for _, u in untrusted.iterrows():
        u_addr = u.get("address_ref_number")
        u_time = parse_dt(u["created_at"])
        if not u_addr or not u_time:
            continue

        # Find subsequent positive events at same address with GPS
        subsequent = disp[
            (disp["address_ref_number"] == u_addr)
            & (pd.to_datetime(disp["created_at"]) > u_time)
            & (disp["disposition"].isin(POSITIVE_EVENT_CODES))
            & (disp.get("is_true_visit", 0) == 1)
        ]

        for _, p in subsequent.iterrows():
            lat = p.get("location_lat") if "location_lat" in p else None
            lng = p.get("location_lng") if "location_lng" in p else None
            if pd.isna(lat) or pd.isna(lng):
                continue
            pairs.append({
                "untrusted_disposition": u["disposition"],
                "untrusted_date": u_time.isoformat(),
                "positive_disposition": p["disposition"],
                "positive_disposition_ref": p.get("disposition_ref_number"),
                "positive_date": parse_dt(p["created_at"]).isoformat() if parse_dt(p["created_at"]) else None,
                "positive_lat": float(lat),
                "positive_lng": float(lng),
            })

    return pairs


def _empty_disposition_features():
    return {
        "total_dispositions": 0,
        "total_attempts_excluding_system": 0,
        "rpc_count": 0, "nc_count": 0, "tpc_count": 0, "wn_count": 0,
        "contact_type_total": 0,
        "ptp_count": 0, "fptp_count": 0, "bptp_count": 0,
        "paid_count": 0, "partial_paid_count": 0,
        "rtp_count": 0, "fraud_count": 0, "lgt_count": 0,
        "lb_count": 0, "lm_count": 0, "doubtful_count": 0, "dif_count": 0,
        "settlement_request_count": 0, "foreclosure_request_count": 0,
        "loan_cancellation_count": 0, "settlement_total": 0,
        "settlement_latest_date": None, "settlement_latest_disposition": None,
        "doubtful_count_60d": 0, "doubtful_latest_comment": None, "doubtful_latest_date": None,
        "dif_count_12mo": 0, "dif_dates_12mo": [],
        "utp_total": 0, "utp_by_reason": {},
        "freeze_latest": None,
        "rpc_hour_distribution": {},
        "address_intelligence_pairs": [],
        "ptp_fptp_records": [], "ptp_fptp_count_with_followup": 0,
        "field_total": 0, "field_productive": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Payment features
# ─────────────────────────────────────────────────────────────────────────────

def payment_features(payments, dispositions, today):
    """Compute payment-derived features."""
    f = {}

    if payments.empty:
        return _empty_payment_features()

    payments = payments.copy()
    payments["payment_datetime"] = pd.to_datetime(payments["payment_datetime"])
    payments = payments.sort_values("payment_datetime").reset_index(drop=True)

    success = payments[payments["status"] == "SUCCESS"]
    f["total_payments"] = len(payments)
    f["success_count"] = len(success)
    f["failed_count"] = int((payments["status"] == "FAILED").sum())

    # ── Payment day-of-month distribution (for payment_day_pattern card) ──
    if len(success) > 0:
        success_copy = success.copy()
        success_copy["dom"] = success_copy["payment_datetime"].dt.day
        f["payment_day_distribution"] = success_copy["dom"].value_counts().sort_index().to_dict()

        # Pre-compute bucket counts for the indicator
        buckets = {"early_month": 0, "mid_month_early": 0, "mid_month_late": 0, "end_month": 0}
        for d in success_copy["dom"]:
            if 1 <= d <= 7:
                buckets["early_month"] += 1
            elif 8 <= d <= 15:
                buckets["mid_month_early"] += 1
            elif 16 <= d <= 24:
                buckets["mid_month_late"] += 1
            elif 25 <= d <= 31:
                buckets["end_month"] += 1
        f["payment_day_buckets"] = buckets
    else:
        f["payment_day_distribution"] = {}
        f["payment_day_buckets"] = {"early_month": 0, "mid_month_early": 0,
                                     "mid_month_late": 0, "end_month": 0}

    # ── Self-pay independence (180d window) ──
    cutoff_180d = today - timedelta(days=180)
    success_180d = success[success["payment_datetime"] >= cutoff_180d]
    f["success_count_180d"] = len(success_180d)
    f["payment_history_months"] = _payment_history_months(payments, today)

    # Self-pay = payment_source in [CUSTOMER_PORTAL, LINK] AND no RPC in 30d before payment
    self_pay_count = 0
    last_self_pay_date = None
    if len(success_180d) > 0 and not dispositions.empty:
        rpc_disp = dispositions[dispositions.get("contact_type", "") == "RPC"].copy()
        if len(rpc_disp) > 0:
            rpc_disp["created_at"] = pd.to_datetime(rpc_disp["created_at"])

        for _, p in success_180d.iterrows():
            if p.get("payment_source") not in ("CUSTOMER_PORTAL", "LINK"):
                continue
            window_start = p["payment_datetime"] - timedelta(days=30)
            if len(rpc_disp) > 0:
                has_rpc = rpc_disp[
                    (rpc_disp["created_at"] >= window_start)
                    & (rpc_disp["created_at"] <= p["payment_datetime"])
                ]
                if len(has_rpc) == 0:
                    self_pay_count += 1
                    last_self_pay_date = p["payment_datetime"].isoformat()
            else:
                self_pay_count += 1
                last_self_pay_date = p["payment_datetime"].isoformat()

    f["self_pay_count_180d"] = self_pay_count
    f["last_self_pay_date"] = last_self_pay_date

    # ── PTP honour resolution (matches PTP/FPTP followup_datetime to SUCCESS payment within 7d) ──
    # We pre-resolve here so the LLM doesn't need to do date-range joins
    # The disposition module already produced ptp_fptp_records; we need that here.
    # We'll compute the resolution in build_features() where both are available.

    return f


def _payment_history_months(payments, today):
    """How many months of payment history exist (oldest payment → today)."""
    if payments.empty:
        return 0
    oldest = pd.to_datetime(payments["payment_datetime"]).min()
    delta_days = (today - oldest.to_pydatetime()).days
    return int(delta_days / 30)


def _empty_payment_features():
    return {
        "total_payments": 0, "success_count": 0, "failed_count": 0,
        "payment_day_distribution": {},
        "payment_day_buckets": {"early_month": 0, "mid_month_early": 0,
                                "mid_month_late": 0, "end_month": 0},
        "success_count_180d": 0, "payment_history_months": 0,
        "self_pay_count_180d": 0, "last_self_pay_date": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BOM (beginning-of-month) features
# ─────────────────────────────────────────────────────────────────────────────

def bom_features(bom):
    """Compute BOM-derived features for trajectory cards."""
    f = {}

    if bom.empty:
        return _empty_bom_features()

    bom = bom.sort_values("month").reset_index(drop=True)
    f["bom_count"] = len(bom)
    f["loan_age_months"] = len(bom)

    if len(bom) == 0:
        return f

    # Current state (latest snapshot)
    latest = bom.iloc[-1]
    f["current_dpd"] = int(latest.get("dpd", 0))
    f["current_bucket"] = safe_str(latest.get("bucket", ""))
    f["current_resolution_status"] = safe_str(latest.get("resolution_status", ""))

    # ── Reset events (DPD goes from >0 to ≤5) ──
    reset_events = []
    dpds = bom["dpd"].tolist()
    months = bom["month"].tolist()
    for i in range(1, len(dpds)):
        if dpds[i - 1] > 0 and dpds[i] <= 5:
            reset_events.append({
                "month": str(months[i]),
                "prev_dpd": int(dpds[i - 1]),
            })
    f["reset_events"] = reset_events
    f["reset_event_count"] = len(reset_events)

    # ── Reset intervals (in months between consecutive resets) ──
    if len(reset_events) >= 2:
        # Convert YYYY-MM to month indices
        month_idx = {str(m): i for i, m in enumerate(months)}
        intervals = []
        for j in range(1, len(reset_events)):
            i1 = month_idx.get(reset_events[j - 1]["month"])
            i2 = month_idx.get(reset_events[j]["month"])
            if i1 is not None and i2 is not None:
                intervals.append(i2 - i1)
        f["reset_intervals_months"] = intervals
        if intervals:
            f["reset_interval_mean"] = float(np.mean(intervals))
            f["reset_interval_stddev"] = float(np.std(intervals))
        else:
            f["reset_interval_mean"] = None
            f["reset_interval_stddev"] = None
    else:
        f["reset_intervals_months"] = []
        f["reset_interval_mean"] = None
        f["reset_interval_stddev"] = None

    # ── Default cycles (DPD>0 episodes with peak bucket) ──
    cycles = _identify_default_cycles(bom)
    f["default_cycles"] = cycles
    f["default_cycle_count"] = len(cycles)

    # ── First 3 months — for serial early EMI defaulter check ──
    first_3 = bom.head(3)
    f["first_3_months_max_dpd"] = int(first_3["dpd"].max()) if len(first_3) > 0 else 0
    f["first_3_months_any_dpd_gt_0"] = bool((first_3["dpd"] > 0).any())

    # ── Last 12 months DPD pattern (for chronic late payer) ──
    last_12 = bom.tail(12)
    if len(last_12) >= 12:
        all_in_1_30 = ((last_12["dpd"] >= 1) & (last_12["dpd"] <= 30)).all()
        none_zero = (last_12["dpd"] != 0).all()
        none_gt_30 = (last_12["dpd"] <= 30).all()
        f["last_12_chronic_late_pattern"] = bool(all_in_1_30 and none_zero and none_gt_30)
    else:
        f["last_12_chronic_late_pattern"] = False

    # ── Consecutive clean months immediately before current default ──
    f["consecutive_clean_months_before_current_default"] = _clean_streak_before_default(bom)

    # ── Bucket boundary crossing count (for bucket bouncer) ──
    f["bucket_boundary_crossings"] = _bucket_boundary_crossings(bom)

    return f


def _identify_default_cycles(bom):
    """A default cycle = consecutive months where DPD>0, bracketed by DPD=0/≤5.
    Returns list with start_month, end_month, peak_bucket."""
    cycles = []
    in_cycle = False
    cycle_start = None
    cycle_buckets = []

    for _, row in bom.iterrows():
        dpd = row["dpd"]
        if dpd > 5 and not in_cycle:
            in_cycle = True
            cycle_start = str(row["month"])
            cycle_buckets = [safe_str(row.get("bucket", ""))]
        elif dpd > 5 and in_cycle:
            cycle_buckets.append(safe_str(row.get("bucket", "")))
        elif dpd <= 5 and in_cycle:
            in_cycle = False
            cycles.append({
                "start_month": cycle_start,
                "end_month": str(row["month"]),
                "peak_bucket": _max_bucket(cycle_buckets),
                "duration_months": len(cycle_buckets),
            })
            cycle_buckets = []

    # If still in cycle at end of data
    if in_cycle:
        cycles.append({
            "start_month": cycle_start,
            "end_month": None,
            "peak_bucket": _max_bucket(cycle_buckets),
            "duration_months": len(cycle_buckets),
        })

    return cycles


def _max_bucket(buckets):
    """Return the most severe bucket from a list — naive ordering for now."""
    # Order: X1 < X2 < X3 < B1 < B2 < B3 < NPA. Adjust to your bucket system.
    order = ["X1", "X2", "X3", "B1", "B2", "B3", "B4", "B5", "NPA", "WO"]
    valid = [b for b in buckets if b in order]
    if not valid:
        # If buckets aren't in our known list, return the last one seen
        return buckets[-1] if buckets else None
    return max(valid, key=lambda b: order.index(b))


def _clean_streak_before_default(bom):
    """Count consecutive DPD=0 months immediately before the current default."""
    if bom.empty:
        return 0
    dpds = bom["dpd"].tolist()
    # Walk backwards from end. If currently DPD=0, return None (no current default).
    if dpds[-1] == 0:
        return None
    # Find last index where DPD > 0 starts (i.e. transition from 0 to >0)
    streak = 0
    for i in range(len(dpds) - 1, -1, -1):
        if dpds[i] > 0:
            continue
        else:
            # Walk further back counting clean months
            for j in range(i, -1, -1):
                if dpds[j] == 0:
                    streak += 1
                else:
                    break
            return streak
    return streak


def _bucket_boundary_crossings(bom):
    """Count how many times the customer crossed any bucket boundary
    forward-and-back. Used for bucket_bouncer detection."""
    if len(bom) < 3:
        return 0
    buckets = [safe_str(b) for b in bom["bucket"].tolist()]
    crossings = 0
    for i in range(1, len(buckets)):
        if buckets[i] != buckets[i - 1] and buckets[i] and buckets[i - 1]:
            crossings += 1
    return crossings // 2  # forward+back = 2 transitions


def _empty_bom_features():
    return {
        "bom_count": 0, "loan_age_months": 0,
        "current_dpd": 0, "current_bucket": "", "current_resolution_status": "",
        "reset_events": [], "reset_event_count": 0,
        "reset_intervals_months": [],
        "reset_interval_mean": None, "reset_interval_stddev": None,
        "default_cycles": [], "default_cycle_count": 0,
        "first_3_months_max_dpd": 0, "first_3_months_any_dpd_gt_0": False,
        "last_12_chronic_late_pattern": False,
        "consecutive_clean_months_before_current_default": None,
        "bucket_boundary_crossings": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-table features (resolved between dispositions and payments)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_ptp_honour(ptp_records, payments):
    """For each PTP/FPTP record, check if a SUCCESS payment exists within
    7 days after the followup_datetime. This is the cross-table join the LLM
    shouldn't have to do."""
    if not ptp_records or payments.empty:
        return {"honoured_count": 0, "total": len(ptp_records), "honour_rate_pct": 0}

    success = payments[payments["status"] == "SUCCESS"].copy()
    if success.empty:
        return {"honoured_count": 0, "total": len(ptp_records), "honour_rate_pct": 0}
    success["payment_datetime"] = pd.to_datetime(success["payment_datetime"])

    honoured = 0
    for rec in ptp_records:
        followup = parse_dt(rec.get("followup_datetime"))
        if not followup:
            continue
        window_end = followup + timedelta(days=7)
        match = success[
            (success["payment_datetime"] >= followup)
            & (success["payment_datetime"] <= window_end)
        ]
        if len(match) > 0:
            honoured += 1

    total = len(ptp_records)
    return {
        "honoured_count": honoured,
        "total": total,
        "honour_rate_pct": int(round(100 * honoured / total)) if total > 0 else 0,
    }


def resolve_address_intelligence(pairs, registered_lat, registered_lng):
    """Compute distance from registered address for the latest positive event
    in each address_intelligence pair. Returns the latest pair's distance."""
    if not pairs:
        return None

    # Take the latest pair across all (sorted by positive_date)
    pairs_sorted = sorted(pairs, key=lambda x: x.get("positive_date") or "")
    latest = pairs_sorted[-1]

    distance_m = haversine_m(
        latest["positive_lat"], latest["positive_lng"],
        registered_lat, registered_lng
    )

    return {
        "latest_pair": latest,
        "distance_m": distance_m,
        "untrusted_disposition_count": len(set(p["untrusted_disposition"] for p in pairs)),
        "positive_event_count": len(pairs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_features(case_ref, dispositions_df, payments_df, bom_df,
                   registered_lat, registered_lng, today, emi_amount):
    """Build the complete features dict for one case.

    Args:
        case_ref: case_ref_number to compute features for
        dispositions_df: full dispositions table (will be filtered to this case)
        payments_df: full payments table
        bom_df: full BOM snapshots table
        registered_lat, registered_lng: registered address GPS
        today: datetime — reference for all recency calcs
        emi_amount: standard EMI in INR

    Returns:
        dict ready to JSON-serialize, keyed by feature name
    """
    # Filter to this case
    disp = dispositions_df[dispositions_df["case_ref_number"] == case_ref].copy()
    pay = payments_df[payments_df["case_ref_number"] == case_ref].copy()
    bom = bom_df[bom_df["case_ref_number"] == case_ref].copy() if "case_ref_number" in bom_df.columns else pd.DataFrame()

    # Normalise today
    if isinstance(today, str):
        today = datetime.fromisoformat(today)

    features = {
        "case_ref_number": case_ref,
        "computed_at": today.isoformat() if today else None,
        "today": today.isoformat() if today else None,
        "emi_amount": emi_amount,
        "registered_address": {"lat": registered_lat, "lng": registered_lng},
    }

    # Stage A: independent features
    features["dispositions"] = disposition_features(disp, today)
    features["payments"] = payment_features(pay, disp, today)
    features["bom"] = bom_features(bom)

    # Stage B: cross-table resolutions
    features["ptp_honour"] = resolve_ptp_honour(
        features["dispositions"]["ptp_fptp_records"], pay
    )
    features["address_intelligence_resolved"] = resolve_address_intelligence(
        features["dispositions"]["address_intelligence_pairs"],
        registered_lat, registered_lng
    )

    return to_serializable(features)
