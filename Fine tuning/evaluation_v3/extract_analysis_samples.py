#!/usr/bin/env python3
# =============================================================================
# extract_analysis_samples.py
# Smart extraction of representative samples from all_predictions.csv
# for research-level error analysis
# =============================================================================

import csv
import json
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"c:\Users\huynh\OneDrive\Desktop\OCR - TEST\Fine tuning\User Validation\evaluation_v3"
CSV_PATH = os.path.join(BASE, "all_predictions.csv")
METRICS_PATH = os.path.join(BASE, "metrics_summary.json")
WORST_PATH = os.path.join(BASE, "error_analysis", "worst_cases_report.csv")
OUTPUT = os.path.join(BASE, "analysis_extract.txt")

# ── Load data ──
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for r in rows:
    for k in ["cer_raw", "cer_sanitized", "cer_corrected", "cer_phobert",
              "wer_corrected", "wer_phobert"]:
        r[k] = float(r[k])

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

print(f"Loaded {len(rows)} rows")

out_lines = []
def emit(line=""):
    out_lines.append(line)
    print(line)

def emit_sample(r, label="", extra_note=""):
    emit(f"  [{label}] idx={r['idx']} domain={r['domain']} cat={r['error_category']}")
    emit(f"    GT:       {r['ref'][:120]}")
    emit(f"    RAW:      {r['raw_pred'][:120]}")
    emit(f"    SANITIZ:  {r['sanitized_pred'][:120]}")
    emit(f"    FINAL:    {r['corrected_pred'][:120]}")
    emit(f"    +PhoBERT: {r['phobert_corrected_pred'][:120]}")
    emit(f"    CER: raw={r['cer_raw']:.4f} san={r['cer_sanitized']:.4f} "
         f"final={r['cer_corrected']:.4f} phobert={r['cer_phobert']:.4f}")
    emit(f"    WER: final={r['wer_corrected']:.4f} phobert={r['wer_phobert']:.4f}")
    if extra_note:
        emit(f"    NOTE: {extra_note}")
    emit()

# =============================================================================
# SECTION 0: AGGREGATE METRICS SUMMARY
# =============================================================================
emit("=" * 80)
emit("SECTION 0: AGGREGATE METRICS (from metrics_summary.json)")
emit("=" * 80)
emit()

ov = metrics["overall"]
emit(f"  Overall:     CER={ov['cer']:.6f}  WER={ov['wer']:.6f}  Samples={ov['total_samples']}")
for dom, m in metrics["per_domain"].items():
    cats_str = ", ".join(f"{k}={v}" for k, v in m["categories"].items())
    emit(f"  {dom.upper():12s}: CER={m['cer']:.6f}  WER={m['wer']:.6f}  "
         f"Samples={m['samples']}  FPS={m['fps']:.1f}")
    emit(f"                 Categories: {cats_str}")

pc = metrics["phobert_comparison"]
emit()
emit(f"  PhoBERT Comparison:")
emit(f"    Without PhoBERT: CER={pc['overall_cer_without_phobert']:.6f}  WER={pc['overall_wer_without_phobert']:.6f}")
emit(f"    With PhoBERT:    CER={pc['overall_cer_with_phobert']:.6f}  WER={pc['overall_wer_with_phobert']:.6f}")
emit(f"    Samples: helped={pc['helped']}  hurt={pc['hurt']}  neutral={pc['neutral']}")
for dom, dm in pc.get("per_domain", {}).items():
    emit(f"    {dom.upper()}: no_phobert_CER={dm['cer_without_phobert']:.6f} "
         f"with_phobert_CER={dm['cer_with_phobert']:.6f}")

# Error category distribution
emit()
emit("  Error Category Distribution (overall):")
all_cats = defaultdict(lambda: {"printed": 0, "handwritten": 0})
for r in rows:
    all_cats[r["error_category"]][r["domain"]] += 1
for cat in ["PERFECT", "MINOR_ERROR", "SUBSTITUTION", "HALLUCINATION_LOOP", "TRUNCATION"]:
    if cat in all_cats:
        d = all_cats[cat]
        emit(f"    {cat:25s}  printed={d['printed']:4d}  handwritten={d['handwritten']:3d}  "
             f"total={d['printed']+d['handwritten']:4d}")

# CER distribution
emit()
cer_vals = [r["cer_corrected"] for r in rows]
perfect = sum(1 for c in cer_vals if c == 0)
minor = sum(1 for c in cer_vals if 0 < c <= 0.05)
medium = sum(1 for c in cer_vals if 0.05 < c <= 0.15)
high = sum(1 for c in cer_vals if 0.15 < c <= 0.3)
severe = sum(1 for c in cer_vals if c > 0.3)
emit(f"  CER Distribution:")
emit(f"    CER=0.00 (perfect):   {perfect:4d} ({100*perfect/len(rows):.1f}%)")
emit(f"    CER 0-0.05 (minor):   {minor:4d} ({100*minor/len(rows):.1f}%)")
emit(f"    CER 0.05-0.15 (med):  {medium:4d} ({100*medium/len(rows):.1f}%)")
emit(f"    CER 0.15-0.30 (high): {high:4d} ({100*high/len(rows):.1f}%)")
emit(f"    CER > 0.30 (severe):  {severe:4d} ({100*severe/len(rows):.1f}%)")

emit()
emit()

# =============================================================================
# SECTION 1: PhoBERT IMPACT — WHERE IT HELPS vs HURTS
# =============================================================================
emit("=" * 80)
emit("SECTION 1: PhoBERT IMPACT ANALYSIS")
emit("=" * 80)

# 1a: PhoBERT HELPS (cer_phobert < cer_corrected — PhoBERT path is better)
phobert_helps = [r for r in rows if r["cer_phobert"] < r["cer_corrected"]]
phobert_helps.sort(key=lambda r: r["cer_corrected"] - r["cer_phobert"], reverse=True)

emit()
emit(f"  1a. PhoBERT HELPS ({len(phobert_helps)} samples total)")
emit(f"      Showing top 7 by CER improvement:")
emit()
for r in phobert_helps[:7]:
    delta = r["cer_corrected"] - r["cer_phobert"]
    emit_sample(r, "PhoBERT_HELPS", f"PhoBERT reduces CER by {delta:.4f}")

# 1b: PhoBERT HURTS (cer_phobert > cer_corrected — PhoBERT path is worse)
phobert_hurts = [r for r in rows if r["cer_phobert"] > r["cer_corrected"]]
phobert_hurts.sort(key=lambda r: r["cer_phobert"] - r["cer_corrected"], reverse=True)

emit(f"  1b. PhoBERT HURTS ({len(phobert_hurts)} samples total)")
emit(f"      Showing top 7 by CER degradation:")
emit()
for r in phobert_hurts[:7]:
    delta = r["cer_phobert"] - r["cer_corrected"]
    emit_sample(r, "PhoBERT_HURTS", f"PhoBERT increases CER by {delta:.4f}")

# =============================================================================
# SECTION 2: SANITIZER IMPACT
# =============================================================================
emit("=" * 80)
emit("SECTION 2: SANITIZER (RegexSanitizer) IMPACT")
emit("=" * 80)

# Where sanitizer helped most (cer_raw > cer_sanitized)
san_helps = [r for r in rows if r["cer_sanitized"] < r["cer_raw"]]
san_helps.sort(key=lambda r: r["cer_raw"] - r["cer_sanitized"], reverse=True)

emit()
emit(f"  Sanitizer improved {len(san_helps)} samples")
emit(f"  Top 7 biggest improvements:")
emit()
for r in san_helps[:7]:
    delta = r["cer_raw"] - r["cer_sanitized"]
    emit_sample(r, "SANITIZER_HELPS", f"Sanitizer reduces CER by {delta:.4f} (hallucination collapse)")

# =============================================================================
# SECTION 3: ADDRESS CORRECTOR IMPACT
# =============================================================================
emit("=" * 80)
emit("SECTION 3: ADDRESS CORRECTOR IMPACT")
emit("=" * 80)

# Where address corrector changed the prediction (corrected_pred != sanitized_pred)
addr_changed = [r for r in rows if r["corrected_pred"] != r["sanitized_pred"]]
addr_helped = [r for r in addr_changed if r["cer_corrected"] < r["cer_sanitized"]]
addr_hurt = [r for r in addr_changed if r["cer_corrected"] > r["cer_sanitized"]]
addr_neutral = [r for r in addr_changed if r["cer_corrected"] == r["cer_sanitized"]]

emit()
emit(f"  Address corrector modified {len(addr_changed)} samples:")
emit(f"    Helped: {len(addr_helped)}  Hurt: {len(addr_hurt)}  Neutral: {len(addr_neutral)}")
emit()

if addr_helped:
    emit(f"  3a. Address Corrector HELPS (top 7):")
    emit()
    addr_helped.sort(key=lambda r: r["cer_sanitized"] - r["cer_corrected"], reverse=True)
    for r in addr_helped[:7]:
        delta = r["cer_sanitized"] - r["cer_corrected"]
        emit_sample(r, "ADDR_HELPS", f"Address corrector reduces CER by {delta:.4f}")

if addr_hurt:
    emit(f"  3b. Address Corrector HURTS (top 5):")
    emit()
    addr_hurt.sort(key=lambda r: r["cer_corrected"] - r["cer_sanitized"], reverse=True)
    for r in addr_hurt[:5]:
        delta = r["cer_corrected"] - r["cer_sanitized"]
        emit_sample(r, "ADDR_HURTS", f"Address corrector increases CER by {delta:.4f}")

# =============================================================================
# SECTION 4: ERROR CATEGORIES — REPRESENTATIVE SAMPLES
# =============================================================================
emit("=" * 80)
emit("SECTION 4: ERROR CATEGORIES — REPRESENTATIVE SAMPLES")
emit("=" * 80)

for cat in ["PERFECT", "MINOR_ERROR", "SUBSTITUTION", "HALLUCINATION_LOOP", "TRUNCATION"]:
    cat_rows = [r for r in rows if r["error_category"] == cat]
    if not cat_rows:
        continue

    emit()
    emit(f"  Category: {cat} ({len(cat_rows)} samples)")
    emit()

    for domain in ["printed", "handwritten"]:
        dom_rows = [r for r in cat_rows if r["domain"] == domain]
        if not dom_rows:
            continue

        if cat == "PERFECT":
            # Show 2 examples per domain — show variety of text types
            samples = dom_rows[:1] + dom_rows[len(dom_rows)//2:len(dom_rows)//2+1]
        elif cat == "MINOR_ERROR":
            # Show samples with different CER ranges
            dom_rows.sort(key=lambda r: r["cer_corrected"])
            n = len(dom_rows)
            samples = [dom_rows[n//4], dom_rows[n//2], dom_rows[3*n//4]]
        elif cat in ("SUBSTITUTION", "HALLUCINATION_LOOP", "TRUNCATION"):
            # Show up to 3 worst examples
            dom_rows.sort(key=lambda r: r["cer_corrected"], reverse=True)
            samples = dom_rows[:3]
        else:
            samples = dom_rows[:2]

        for r in samples:
            emit_sample(r, f"{cat}/{domain}")

# =============================================================================
# SECTION 5: INTERESTING METRIC DELTAS
# =============================================================================
emit("=" * 80)
emit("SECTION 5: INTERESTING METRIC RELATIONSHIPS")
emit("=" * 80)

# 5a: Cases where raw prediction was perfect but post-processing degraded it
degraded_from_perfect = [r for r in rows if r["cer_raw"] == 0 and r["cer_corrected"] > 0]
emit()
emit(f"  5a. Perfect RAW degraded by post-processing: {len(degraded_from_perfect)} samples")
if degraded_from_perfect:
    degraded_from_perfect.sort(key=lambda r: r["cer_corrected"], reverse=True)
    emit(f"      Top 5:")
    emit()
    for r in degraded_from_perfect[:5]:
        emit_sample(r, "DEGRADED", f"RAW was perfect, final CER={r['cer_corrected']:.4f}")

# 5b: Huge sanitizer improvement (hallucination collapse)
huge_san = [r for r in rows if r["cer_raw"] - r["cer_sanitized"] > 0.3]
emit(f"  5b. Huge sanitizer improvement (CER delta > 0.3): {len(huge_san)} samples")
if huge_san:
    huge_san.sort(key=lambda r: r["cer_raw"] - r["cer_sanitized"], reverse=True)
    emit(f"      Top 5:")
    emit()
    for r in huge_san[:5]:
        emit_sample(r, "HUGE_SAN", f"Sanitizer CER delta: {r['cer_raw']-r['cer_sanitized']:.4f}")

# 5c: Cases where PhoBERT and no-PhoBERT give very different results
big_phobert_diff = [r for r in rows if abs(r["cer_phobert"] - r["cer_corrected"]) > 0.05]
emit(f"  5c. Large PhoBERT divergence (|delta| > 0.05): {len(big_phobert_diff)} samples")
if big_phobert_diff:
    big_phobert_diff.sort(key=lambda r: abs(r["cer_phobert"] - r["cer_corrected"]), reverse=True)
    emit(f"      Top 5:")
    emit()
    for r in big_phobert_diff[:5]:
        delta = r["cer_phobert"] - r["cer_corrected"]
        direction = "PhoBERT WORSE" if delta > 0 else "PhoBERT BETTER"
        emit_sample(r, direction, f"|delta|={abs(delta):.4f}")

# 5d: Handwritten samples that achieved perfect
hw_perfect = [r for r in rows if r["domain"] == "handwritten" and r["cer_corrected"] == 0]
emit(f"  5d. Handwritten PERFECT predictions: {len(hw_perfect)} samples")
emit(f"      (showing 3 examples)")
emit()
for r in hw_perfect[:3]:
    emit_sample(r, "HW_PERFECT")

# =============================================================================
# SECTION 6: WORST CASES (from worst_cases_report.csv)
# =============================================================================
emit("=" * 80)
emit("SECTION 6: WORST CASES (Top 10)")
emit("=" * 80)
emit()

with open(WORST_PATH, "r", encoding="utf-8") as f:
    wr = list(csv.DictReader(f))

for w in wr[:10]:
    emit(f"  [WORST #{w['Rank']}] domain={w['Domain']} cat={w['Error_Category']}")
    emit(f"    GT:    {w['Ground_Truth'][:120]}")
    emit(f"    RAW:   {w['Raw_Pred'][:120]}")
    emit(f"    FINAL: {w['Corrected_Pred'][:120]}")
    emit(f"    CER: raw={w['CER_Raw']} san={w['CER_Sanitized']} final={w['CER_Corrected']}")
    emit()

# =============================================================================
# SECTION 7: PIPELINE STAGE STATISTICS
# =============================================================================
emit("=" * 80)
emit("SECTION 7: PIPELINE STAGE-BY-STAGE STATISTICS")
emit("=" * 80)
emit()

for domain in ["printed", "handwritten"]:
    dom_rows = [r for r in rows if r["domain"] == domain]
    n = len(dom_rows)
    
    avg_raw = sum(r["cer_raw"] for r in dom_rows) / n
    avg_san = sum(r["cer_sanitized"] for r in dom_rows) / n
    avg_cor = sum(r["cer_corrected"] for r in dom_rows) / n
    avg_pho = sum(r["cer_phobert"] for r in dom_rows) / n
    
    perfect_raw = sum(1 for r in dom_rows if r["cer_raw"] == 0)
    perfect_san = sum(1 for r in dom_rows if r["cer_sanitized"] == 0)
    perfect_cor = sum(1 for r in dom_rows if r["cer_corrected"] == 0)
    perfect_pho = sum(1 for r in dom_rows if r["cer_phobert"] == 0)
    
    emit(f"  {domain.upper()} ({n} samples):")
    emit(f"    Stage            Avg CER    Perfect")
    emit(f"    ─────────────    ───────    ───────")
    emit(f"    Raw TrOCR        {avg_raw:.6f}   {perfect_raw:4d} ({100*perfect_raw/n:.1f}%)")
    emit(f"    + Sanitizer      {avg_san:.6f}   {perfect_san:4d} ({100*perfect_san/n:.1f}%)")
    emit(f"    + Addr Correct   {avg_cor:.6f}   {perfect_cor:4d} ({100*perfect_cor/n:.1f}%)")
    emit(f"    + PhoBERT path   {avg_pho:.6f}   {perfect_pho:4d} ({100*perfect_pho/n:.1f}%)")
    emit()

# ── Save output ──
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print(f"\n{'='*80}")
print(f"Saved analysis extract to: {OUTPUT}")
print(f"Total lines: {len(out_lines)}")
