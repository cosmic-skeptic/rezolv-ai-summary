"""
Generates 10 ambiguous test accounts for the Rezolv Intelligence prototype.

Unlike the earlier generate_sample.py which engineered specific personas, this
generates accounts with random-ish patterns — varied disposition volumes, mixed
event types, realistic noise, and edge cases near indicator thresholds. The
goal is to test whether the prompt + features pipeline correctly *reads*
patterns out of messy data, rather than verifying it against pre-decided
outcomes.

Run: python scripts/generate_realistic.py
Output: sample_data/{dispositions,payments,bom_snapshots}.csv
"""

import pandas as pd
from datetime import datetime, timedelta
import random
import os

random.seed(7)  # reproducible but not engineered to specific outcomes

OUT_DIR = "sample_data"
os.makedirs(OUT_DIR, exist_ok=True)

TODAY = datetime(2026, 4, 30)

# Realistic Mumbai/Pune/Bangalore-ish coordinates
BASE_COORDS = [
    (19.1136, 72.8697),  # Andheri East
    (19.0760, 72.8777),  # Bandra
    (18.5204, 73.8567),  # Pune central
    (12.9716, 77.5946),  # Bangalore central
    (13.0827, 80.2707),  # Chennai
    (28.6139, 77.2090),  # Delhi
    (17.3850, 78.4867),  # Hyderabad
    (22.5726, 88.3639),  # Kolkata
    (23.0225, 72.5714),  # Ahmedabad
    (26.9124, 75.7873),  # Jaipur
]

# Realistic Indian names
NAMES = [
    "Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sunita Desai", "Vikram Singh",
    "Meera Nair", "Arjun Reddy", "Kavita Joshi", "Sanjay Verma", "Pooja Iyer",
]

DISPOSITION_POOL_NC = [
    ("NC", "NC Reason", ["Switch off", "Number busy", "Out of Service",
                          "Not Reachable", "Auto Disconnect"]),
    ("RNR", "RNR Reason", ["No Answer", "Customer Disconnected"]),
    ("HUNG_UP", None, [None]),
    ("WN", None, [None]),
]

DISPOSITION_POOL_FIELD_NC = [
    ("DL", None, [None]),
    ("NOT_PRESENT", None, [None]),
    ("OS", None, [None]),
    ("ANF", None, [None]),
]

UTP_REASONS = ["Medical Issue", "Loss of income", "Unexpected Family Expense"]

# Random comment fragments — mostly innocuous, occasionally diagnostic
COMMENT_FRAGMENTS = [
    "Customer not responsive", "Will call back",
    "Asked for time", "No commitment given",
    "Mother said son not home", "Spoke briefly, busy",
    "Promised to pay next week", "Says cash flow issue this month",
    "Door was locked, neighbours unaware", "Will pay after salary",
    "Says NACH should have worked", "Customer travelling",
]

DOUBTFUL_COMMENTS = [
    "Customer evasive, gave vague answers about payment",
    "Behaviour seems different from last visit, not committing",
    "Did not give straight answer on payment date",
    "Avoided discussing the loan amount, kept changing topic",
]


def gen_disposition(case_ref, drn_idx, source, code, contact_type, days_ago,
                    sub_label=None, sub=None, comment="", followup=None,
                    addr_ref=None, hour=None, lat=None, lng=None,
                    is_true_visit=None):
    if hour is None:
        hour = random.randint(9, 19)
    base = TODAY - timedelta(days=days_ago)
    base = base.replace(hour=hour, minute=random.randint(0, 59), second=0)
    return {
        "disposition_ref_number": f"DRN_{case_ref[-4:]}_{drn_idx:04d}",
        "source": source,
        "case_ref_number": case_ref,
        "customer_ref_number": f"CT_{case_ref[-5:]}",
        "disposition": code,
        "contact_type": contact_type,
        "sub_disposition_label": sub_label,
        "sub_disposition": sub,
        "comment": comment,
        "created_at": base.isoformat(),
        "promised_amount": 0,
        "followup_datetime": followup.isoformat() if followup else None,
        "address_ref_number": addr_ref or "ADR_001",
        "is_true_visit": is_true_visit if is_true_visit is not None else (1 if source == "FIELD" else None),
        "location_lat": lat,
        "location_lng": lng,
    }


def gen_payment(case_ref, prn_idx, amount, mode, status, source, days_ago, hour=None):
    if hour is None:
        hour = random.randint(8, 22)
    base = TODAY - timedelta(days=days_ago)
    base = base.replace(hour=hour, minute=random.randint(0, 59), second=0)
    return {
        "payment_ref_number": f"PRN_{case_ref[-4:]}_{prn_idx:04d}",
        "object": "CUSTOMER",
        "object_ref_number": f"CT_{case_ref[-5:]}",
        "payment_for": "CASE",
        "payment_for_ref_number": case_ref,
        "payment_reason": "LOAN_PAYMENT",
        "payment_mode": mode,
        "amount": amount,
        "status": status,
        "payment_datetime": base.isoformat(),
        "case_ref_number": case_ref,
        "payment_source": source,
    }


def gen_bom(case_ref, month_offset, dpd, principal, bucket, status, total_due=None):
    month = (datetime(2026, 4, 1) - timedelta(days=30 * month_offset)).strftime("%Y-%m")
    return {
        "case_ref_number": case_ref,
        "month": month,
        "dpd": dpd,
        "principal_outstanding": principal,
        "bucket": bucket,
        "total_due": total_due or 12300,
        "resolution_status": status,
    }


def jitter_coords(base_lat, base_lng, jitter_m=50):
    """Add small random offset to GPS in metres (rough conversion)."""
    # 1 degree lat ≈ 111km; 1 degree lng at this latitude ≈ 105km
    lat_offset = random.uniform(-jitter_m, jitter_m) / 111000
    lng_offset = random.uniform(-jitter_m, jitter_m) / 105000
    return base_lat + lat_offset, base_lng + lng_offset


def random_walk_dpd(months, start_dpd=0, behaviour=None):
    """Generate a DPD time series with some character but not engineered.

    behaviour can be 'volatile', 'steady', 'declining', 'rising', 'cyclic',
    'mostly_clean', or None (random). The function adds noise so behaviours
    aren't perfectly clean — closer to real data.
    """
    if behaviour is None:
        behaviour = random.choice(["volatile", "steady", "declining", "rising",
                                    "cyclic", "mostly_clean"])
    series = []
    dpd = start_dpd
    for m in range(months):
        if behaviour == "volatile":
            change = random.choice([-30, -15, 0, 15, 30, 45])
        elif behaviour == "steady":
            change = random.choice([-5, 0, 0, 0, 5])
        elif behaviour == "declining":
            change = random.choice([-15, -10, -5, 0, 5])
        elif behaviour == "rising":
            change = random.choice([-5, 0, 10, 15, 20])
        elif behaviour == "cyclic":
            # Rough 3-month cycle but with noise
            phase = m % 3
            if phase == 0:
                change = random.choice([20, 30, 35])
            elif phase == 1:
                change = random.choice([20, 30])
            else:
                change = -dpd if dpd > 0 else 0  # reset
        elif behaviour == "mostly_clean":
            if random.random() < 0.85:
                dpd = 0
                change = 0
            else:
                change = random.choice([15, 30])
        else:
            change = 0

        dpd = max(0, dpd + change)
        series.append(dpd)
    return series, behaviour


def dpd_to_bucket(dpd):
    """Convert DPD to a rough bucket name."""
    if dpd == 0:
        return "Norm"
    if dpd <= 30:
        return "X1"
    if dpd <= 60:
        return "X2"
    if dpd <= 90:
        return "X3"
    if dpd <= 120:
        return "B1"
    if dpd <= 180:
        return "B2"
    return "B3"


def dpd_to_status(prev_dpd, dpd):
    """Resolution status given month-over-month DPD change."""
    if dpd == 0 and prev_dpd > 0:
        return "Rollback"
    if dpd == 0:
        return "Norm"
    if dpd > prev_dpd + 15:
        return "Rollforward"
    if dpd < prev_dpd - 5:
        return "Rollback"
    if dpd > 0 and abs(dpd - prev_dpd) <= 5:
        return "Stab"
    return "Open"


# ─────────────────────────────────────────────────────────────────────────────
# Generators per case
# ─────────────────────────────────────────────────────────────────────────────

def generate_account(case_ref, base_lat, base_lng, name, idx):
    """Generate a single account's full data with random-ish characteristics."""
    disp_records = []
    pay_records = []
    bom_records = []

    # ── BOM generation ──
    months = random.choice([4, 8, 12, 14, 18, 24, 30])  # varied loan ages
    dpd_series, behaviour = random_walk_dpd(months)

    principal = random.choice([85000, 120000, 175000, 230000, 310000])
    emi = principal // random.choice([18, 24, 36])

    prev_dpd = 0
    for m_offset, dpd in enumerate(reversed(dpd_series)):
        bom_records.append(gen_bom(
            case_ref, months - 1 - m_offset, dpd,
            principal - (months - m_offset) * (emi // 4),
            dpd_to_bucket(dpd),
            dpd_to_status(prev_dpd, dpd),
            total_due=emi,
        ))
        prev_dpd = dpd

    # ── Disposition generation ──
    # Volume varies wildly: some accounts get 5 attempts, some get 200
    n_attempts = random.choice([3, 8, 15, 30, 60, 120, 200])
    drn_idx = 0

    # Address registry — most accounts have 1 address, some have 2
    n_addresses = 1 if random.random() < 0.75 else 2
    addr_refs = [f"ADR_{idx}_{i:02d}" for i in range(n_addresses)]

    # Generate dispositions across the loan lifetime
    days_span = months * 30
    for _ in range(n_attempts):
        drn_idx += 1
        days_ago = random.randint(1, days_span)

        # Pick channel
        channel_roll = random.random()
        if channel_roll < 0.45:
            source = "AUTODIALER"
        elif channel_roll < 0.70:
            source = "TELECALLING"
        elif channel_roll < 0.90:
            source = "FIELD"
        else:
            source = "VOICEBOT"

        # Outcome distribution depends on the account's underlying behaviour
        outcome_roll = random.random()
        addr_ref = random.choice(addr_refs)

        # Inject reachability variance — some accounts are easier to reach
        rpc_propensity = random.uniform(0.05, 0.45)

        if source == "FIELD":
            if outcome_roll < rpc_propensity:
                # Field RPC — pick a positive or neutral outcome
                code_roll = random.random()
                # Slight GPS jitter from registered address
                lat, lng = jitter_coords(base_lat, base_lng, jitter_m=random.choice([30, 80, 150, 600]))
                if code_roll < 0.30:
                    followup = TODAY - timedelta(days=days_ago - random.randint(2, 5))
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "PTP", "RPC",
                        days_ago=days_ago,
                        sub_label="Payment Mode",
                        sub=random.choice(["Online", "Pick-up", "Casa Debit"]),
                        comment=random.choice(COMMENT_FRAGMENTS),
                        followup=followup, addr_ref=addr_ref, lat=lat, lng=lng,
                    ))
                elif code_roll < 0.45:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "PARTIAL_PAID", "RPC",
                        days_ago=days_ago,
                        comment="Customer paid partial amount",
                        addr_ref=addr_ref, lat=lat, lng=lng,
                    ))
                elif code_roll < 0.55:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "PAID", "RPC",
                        days_ago=days_ago,
                        addr_ref=addr_ref, lat=lat, lng=lng,
                    ))
                elif code_roll < 0.65:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "UTP", "RPC",
                        days_ago=days_ago,
                        sub_label="UTP Reason",
                        sub=random.choice(UTP_REASONS),
                        comment=random.choice(COMMENT_FRAGMENTS),
                        addr_ref=addr_ref, lat=lat, lng=lng,
                    ))
                elif code_roll < 0.72:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "DOUBTFUL", "RPC",
                        days_ago=days_ago,
                        comment=random.choice(DOUBTFUL_COMMENTS),
                        addr_ref=addr_ref, lat=lat, lng=lng,
                    ))
                elif code_roll < 0.80:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "BPTP", "RPC",
                        days_ago=days_ago,
                        comment="Previous PTP not honoured",
                        addr_ref=addr_ref, lat=lat, lng=lng,
                    ))
                elif code_roll < 0.88:
                    # Settlement requests show up in some accounts
                    if random.random() < 0.5:
                        disp_records.append(gen_disposition(
                            case_ref, drn_idx, source, "SETTLEMENT_REQUEST", "RPC",
                            days_ago=days_ago,
                            comment="Customer asking for settlement",
                            addr_ref=addr_ref, lat=lat, lng=lng,
                        ))
                    else:
                        disp_records.append(gen_disposition(
                            case_ref, drn_idx, source, "FORECLOSURE_REQUEST", "RPC",
                            days_ago=days_ago,
                            addr_ref=addr_ref, lat=lat, lng=lng,
                        ))
                elif code_roll < 0.94:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "RTP", "RPC",
                        days_ago=days_ago,
                        comment="Customer refused to pay",
                        addr_ref=addr_ref, lat=lat, lng=lng,
                    ))
                else:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "LB", "RPC",
                        days_ago=days_ago,
                        comment="Could not communicate",
                        addr_ref=addr_ref, lat=lat, lng=lng,
                    ))
            else:
                # Field NC outcome
                code, sub_label, subs = random.choice(DISPOSITION_POOL_FIELD_NC)
                disp_records.append(gen_disposition(
                    case_ref, drn_idx, source, code, "NC",
                    days_ago=days_ago, sub_label=sub_label,
                    sub=random.choice(subs), addr_ref=addr_ref,
                    lat=base_lat + random.uniform(-0.001, 0.001),
                    lng=base_lng + random.uniform(-0.001, 0.001),
                ))
        else:
            # Tele/auto/voicebot outcomes
            if outcome_roll < rpc_propensity:
                code_roll = random.random()
                if code_roll < 0.35:
                    followup = TODAY - timedelta(days=days_ago - random.randint(2, 7))
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "PTP", "RPC",
                        days_ago=days_ago,
                        sub_label="Payment Mode",
                        sub=random.choice(["Online", "Casa Debit"]),
                        comment=random.choice(COMMENT_FRAGMENTS),
                        followup=followup, addr_ref=addr_ref,
                    ))
                elif code_roll < 0.50:
                    followup = TODAY - timedelta(days=days_ago - random.randint(7, 30))
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "FPTP", "RPC",
                        days_ago=days_ago,
                        sub_label="Payment Mode",
                        sub="Online",
                        followup=followup, addr_ref=addr_ref,
                    ))
                elif code_roll < 0.62:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "UTP", "RPC",
                        days_ago=days_ago,
                        sub_label="UTP Reason",
                        sub=random.choice(UTP_REASONS),
                        addr_ref=addr_ref,
                    ))
                elif code_roll < 0.72:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "CB", "RPC",
                        days_ago=days_ago,
                        comment="Will call back",
                        addr_ref=addr_ref,
                    ))
                elif code_roll < 0.80:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "DOUBTFUL", "RPC",
                        days_ago=days_ago,
                        comment=random.choice(DOUBTFUL_COMMENTS),
                        addr_ref=addr_ref,
                    ))
                elif code_roll < 0.88:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "BPTP", "RPC",
                        days_ago=days_ago,
                        addr_ref=addr_ref,
                    ))
                elif code_roll < 0.94:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "RTP", "RPC",
                        days_ago=days_ago,
                        comment="Refused to pay",
                        addr_ref=addr_ref,
                    ))
                else:
                    disp_records.append(gen_disposition(
                        case_ref, drn_idx, source, "LB", "RPC",
                        days_ago=days_ago,
                        addr_ref=addr_ref,
                    ))
            elif outcome_roll < rpc_propensity + 0.15:
                # TPC — third party (mother/spouse/etc.) picked up
                disp_records.append(gen_disposition(
                    case_ref, drn_idx, source, "TPC", "TPC",
                    days_ago=days_ago,
                    sub_label="Party Contacted",
                    sub=random.choice(["Mother", "Spouse", "Brother", "Father"]),
                    comment="Spoke to family member",
                    addr_ref=addr_ref,
                ))
            else:
                code, sub_label, subs = random.choice(DISPOSITION_POOL_NC)
                disp_records.append(gen_disposition(
                    case_ref, drn_idx, source, code, "NC",
                    days_ago=days_ago,
                    sub_label=sub_label,
                    sub=random.choice(subs) if sub_label else None,
                    addr_ref=addr_ref,
                ))

    # Maybe inject a freeze event on some accounts
    freeze_roll = random.random()
    if freeze_roll < 0.10:
        # DNC raised
        days_back = random.randint(15, 90)
        followup = TODAY + timedelta(days=random.randint(15, 60)) if random.random() < 0.5 else None
        drn_idx += 1
        disp_records.append(gen_disposition(
            case_ref, drn_idx, "TELECALLING", "DNC", "RPC",
            days_ago=days_back,
            sub_label="Type",
            sub=random.choice(["Permanent", "Temporary"]),
            comment="Customer requested no contact",
            followup=followup,
        ))
    elif freeze_roll < 0.13:
        # DIF
        drn_idx += 1
        disp_records.append(gen_disposition(
            case_ref, drn_idx, "FIELD", "DIF", "RPC",
            days_ago=random.randint(20, 200),
            sub_label="Type",
            sub="Permanent",
            comment="Family member passed away",
        ))

    # ── Payments ──
    n_payments = random.choice([2, 5, 10, 18, 25])
    for p_idx in range(n_payments):
        days_ago = random.randint(5, days_span)
        amount = random.choice([emi, emi // 2, int(emi * 0.7), int(emi * 0.4),
                                 int(emi * 1.0), int(emi * 0.85)])
        mode = random.choice(["UPI", "NACH", "CASH", "ONLINE", "CASA_DEBIT"])
        status = "SUCCESS" if random.random() < 0.75 else "FAILED"
        source = random.choice(["LINK", "CUSTOMER_PORTAL", "FIELD_APP", "TELECALLING"])
        pay_records.append(gen_payment(
            case_ref, p_idx, amount, mode, status, source, days_ago=days_ago,
        ))

    return disp_records, pay_records, bom_records


# ─────────────────────────────────────────────────────────────────────────────
# Generate 10 accounts
# ─────────────────────────────────────────────────────────────────────────────

all_disp = []
all_pay = []
all_bom = []

case_summaries = []

for i in range(10):
    case_ref = f"CR_{i+1:03d}_{NAMES[i].split()[0].upper()}"
    base_lat, base_lng = BASE_COORDS[i]
    name = NAMES[i]

    disp, pay, bom = generate_account(case_ref, base_lat, base_lng, name, i + 1)

    all_disp.extend(disp)
    all_pay.extend(pay)
    all_bom.extend(bom)

    case_summaries.append({
        "case_ref": case_ref,
        "name": name,
        "registered_lat": base_lat,
        "registered_lng": base_lng,
        "n_dispositions": len(disp),
        "n_payments": len(pay),
        "n_bom": len(bom),
    })


# Sort each by created_at / payment_datetime / month so files look natural
disp_df = pd.DataFrame(all_disp).sort_values("created_at").reset_index(drop=True)
pay_df = pd.DataFrame(all_pay).sort_values("payment_datetime").reset_index(drop=True)
bom_df = pd.DataFrame(all_bom).sort_values(["case_ref_number", "month"]).reset_index(drop=True)

disp_df.to_csv(f"{OUT_DIR}/dispositions.csv", index=False)
pay_df.to_csv(f"{OUT_DIR}/payments.csv", index=False)
bom_df.to_csv(f"{OUT_DIR}/bom_snapshots.csv", index=False)

# Also save a summary so user knows what cases exist
import json
with open(f"{OUT_DIR}/case_summary.json", "w") as f:
    json.dump(case_summaries, f, indent=2, default=str)

print(f"Generated {len(case_summaries)} accounts:")
for s in case_summaries:
    print(f"  {s['case_ref']:30s} | {s['name']:18s} | "
          f"disp={s['n_dispositions']:3d}  pay={s['n_payments']:3d}  bom={s['n_bom']:2d}")
print(f"\nWrote: {OUT_DIR}/dispositions.csv ({len(disp_df)} rows)")
print(f"Wrote: {OUT_DIR}/payments.csv ({len(pay_df)} rows)")
print(f"Wrote: {OUT_DIR}/bom_snapshots.csv ({len(bom_df)} rows)")
print(f"Wrote: {OUT_DIR}/case_summary.json")
