"""
Generates synthetic but realistic CSV data for 3 accounts:
- CR_SAMPLE_ARUNA — chronic late payer, recently doubtful
- CR_SAMPLE_RAMESH — predictable cycle payer (DPD oscillation pattern)
- CR_SAMPLE_PRIYA — first-time defaulter with settlement intent

Output: sample_data/{dispositions,payments,bom_snapshots}.csv

Run: python scripts/generate_sample.py
"""

import pandas as pd
from datetime import datetime, timedelta
import random
import os

random.seed(42)

OUT_DIR = "sample_data"
os.makedirs(OUT_DIR, exist_ok=True)

# Mumbai-ish coords
REG_LAT = 19.1136
REG_LNG = 72.8697


def disp(case, drn_seq, source, code, contact_type, comment="", days_ago=0,
         hour=12, addr_ref="ADR_001", followup=None, sub_label=None,
         sub=None, lat=None, lng=None, true_visit=1):
    base = datetime(2026, 4, 30) - timedelta(days=days_ago, hours=24-hour)
    return {
        "disposition_ref_number": f"DRN_{case[-3:]}_{drn_seq:03d}",
        "source": source,
        "case_ref_number": case,
        "customer_ref_number": f"CT_{case[-5:]}",
        "disposition": code,
        "contact_type": contact_type,
        "sub_disposition_label": sub_label,
        "sub_disposition": sub,
        "comment": comment,
        "created_at": base.isoformat(),
        "promised_amount": 0,
        "followup_datetime": followup.isoformat() if followup else None,
        "address_ref_number": addr_ref,
        "is_true_visit": true_visit,
        "location_lat": lat,
        "location_lng": lng,
    }


def payment(case, prn_seq, amount, mode, status, source, days_ago=0):
    base = datetime(2026, 4, 30) - timedelta(days=days_ago)
    return {
        "payment_ref_number": f"PRN_{case[-3:]}_{prn_seq:03d}",
        "object": "CUSTOMER",
        "object_ref_number": f"CT_{case[-5:]}",
        "payment_for": "CASE",
        "payment_for_ref_number": case,
        "payment_reason": "LOAN_PAYMENT",
        "payment_mode": mode,
        "amount": amount,
        "status": status,
        "payment_datetime": base.isoformat(),
        "case_ref_number": case,
        "payment_source": source,
    }


def bom_row(case, month_offset, dpd, principal, bucket, status, total_due=12300):
    month = (datetime(2026, 4, 1) - timedelta(days=30 * month_offset)).strftime("%Y-%m")
    return {
        "case_ref_number": case,
        "month": month,
        "dpd": dpd,
        "principal_outstanding": principal,
        "bucket": bucket,
        "total_due": total_due,
        "resolution_status": status,
    }


# ─── Account 1: Aruna — chronic late payer, doubtful trail ──────────────────
aruna = "CR_SAMPLE_ARUNA"
aruna_disp = []
aruna_pay = []
aruna_bom = []

# 18 BOM snapshots — DPD always 15-25, never 0, never >30 (chronic late pattern)
for i in range(18):
    dpd = random.randint(15, 28)
    aruna_bom.append(bom_row(aruna, 17 - i, dpd, 200000 - i * 500, "X1", "Stab"))

# Dispositions — many attempts, recent doubtful, mostly NC
seq = 0
for d in range(180, 1, -3):
    seq += 1
    if d % 3 == 0:
        aruna_disp.append(disp(aruna, seq, "AUTODIALER", "NC", "NC",
                              sub_label="NC Reason", sub="Switch off", days_ago=d))
    elif d % 4 == 0:
        aruna_disp.append(disp(aruna, seq, "TELECALLING", "RNR", "NC",
                              sub_label="RNR Reason", sub="No Answer", days_ago=d))
    elif d % 7 == 0:
        seq += 1
        followup = datetime(2026, 4, 30) - timedelta(days=d - 5)
        aruna_disp.append(disp(aruna, seq, "TELECALLING", "PTP", "RPC",
                              comment="Will pay by next week",
                              days_ago=d, hour=15, followup=followup,
                              sub_label="Payment Mode", sub="Online"))

# Recent: 2 doubtful in last 30 days, low PTP honour
aruna_disp.append(disp(aruna, 90, "FIELD", "DOUBTFUL", "RPC",
                      comment="Customer evasive, gave vague answers about payment", days_ago=20,
                      lat=REG_LAT, lng=REG_LNG))
aruna_disp.append(disp(aruna, 91, "FIELD", "DOUBTFUL", "RPC",
                      comment="Behaviour changed since last visit, not committing to date", days_ago=10,
                      lat=REG_LAT, lng=REG_LNG))

# Payments: regular but late, mostly partial, some self-pay
for m in range(8):
    days_ago = 30 * m + random.randint(15, 25)
    if random.random() < 0.7:
        # partial
        aruna_pay.append(payment(aruna, m * 2, 8000, "UPI", "SUCCESS", "LINK", days_ago=days_ago))
    else:
        aruna_pay.append(payment(aruna, m * 2, 12300, "UPI", "SUCCESS", "LINK", days_ago=days_ago))


# ─── Account 2: Ramesh — predictable cycle payer ────────────────────────────
ramesh = "CR_SAMPLE_RAMESH"
ramesh_disp = []
ramesh_pay = []
ramesh_bom = []

# 12 BOM snapshots showing the cycle: 30, 60, 0, 30, 60, 0, ...
cycle_pattern = [30, 60, 0, 30, 60, 0, 30, 60, 0, 30, 60, 0]
for i, dpd in enumerate(cycle_pattern):
    bucket = "X3" if dpd == 60 else ("X1" if dpd == 30 else "Norm")
    status = "Rollback" if dpd == 0 else "Rollforward"
    ramesh_bom.append(bom_row(ramesh, 11 - i, dpd, 180000, bucket, status))

# Dispositions and payments aligned to the 3-month cycle
seq = 0
# Payments cluster around the 4th of every 3rd month (when DPD resets)
for cycle in range(4):
    days_ago = 30 * (3 * cycle) + 4  # day 4 of reset month
    seq += 1
    ramesh_pay.append(payment(ramesh, seq, 24600, "UPI", "SUCCESS", "LINK", days_ago=days_ago))

# Some PTP/FPTP and an RPC pattern around the reset windows
for m in range(5):
    seq += 1
    days_ago = 30 * m + 15
    ramesh_disp.append(disp(ramesh, seq, "TELECALLING", "RNR", "NC",
                           sub_label="RNR Reason", sub="No Answer", days_ago=days_ago, hour=14))


# ─── Account 3: Priya — first-time defaulter with settlement intent ────────
priya = "CR_SAMPLE_PRIYA"
priya_disp = []
priya_pay = []
priya_bom = []

# 24 BOM snapshots: 22 clean + 2 recent defaults
for i in range(22):
    priya_bom.append(bom_row(priya, 23 - i, 0, 150000 - i * 1000, "Norm", "Norm"))
for i, dpd in enumerate([35, 65]):
    priya_bom.append(bom_row(priya, 1 - i, dpd, 130000, "X2", "Rollforward"))

# Recent settlement request + foreclosure request
priya_disp.append(disp(priya, 1, "FIELD", "SETTLEMENT_REQUEST", "RPC",
                      comment="Customer asking for one-time settlement", days_ago=15,
                      lat=REG_LAT, lng=REG_LNG))
priya_disp.append(disp(priya, 2, "TELECALLING", "FORECLOSURE_REQUEST", "RPC",
                      comment="Wants to close the loan", days_ago=8))
priya_disp.append(disp(priya, 3, "FIELD", "PTP", "RPC",
                      comment="Will pay 50k as settlement", days_ago=5,
                      followup=datetime(2026, 5, 5),
                      sub_label="Payment Mode", sub="Online", lat=REG_LAT, lng=REG_LNG))

# Mostly clean payment history, recent failure
for m in range(15):
    priya_pay.append(payment(priya, m, 12300, "NACH", "SUCCESS", "FIELD_APP", days_ago=60 + 30 * m))
priya_pay.append(payment(priya, 99, 12300, "NACH", "FAILED", "FIELD_APP", days_ago=20))


# ─── Combine and write ──────────────────────────────────────────────────────
all_disp = aruna_disp + ramesh_disp + priya_disp
all_pay = aruna_pay + ramesh_pay + priya_pay
all_bom = aruna_bom + ramesh_bom + priya_bom

pd.DataFrame(all_disp).to_csv(f"{OUT_DIR}/dispositions.csv", index=False)
pd.DataFrame(all_pay).to_csv(f"{OUT_DIR}/payments.csv", index=False)
pd.DataFrame(all_bom).to_csv(f"{OUT_DIR}/bom_snapshots.csv", index=False)

print(f"Wrote {len(all_disp)} dispositions, {len(all_pay)} payments, {len(all_bom)} BOM rows to {OUT_DIR}/")
print(f"Cases: {sorted(set(d['case_ref_number'] for d in all_disp))}")
