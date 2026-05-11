"""
Visualization script - local version (converted from Google Colab)
Usage:
    python visualization_local.py --log your_training.log
    python visualization_local.py --log your_training.log --save
"""

import re
import sys
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ============================================================
# ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser(description="Visualize training log metrics.")
parser.add_argument("--log", required=True, help="Path to your training log file")
parser.add_argument("--save", action="store_true", help="Save charts as PNG files instead of showing them")
args = parser.parse_args()

LOG_PATH = args.log
SAVE_CHARTS = args.save

print(f"Reading log: {LOG_PATH}")


# ============================================================
# REGEX
# ============================================================

# PHASE (only applies to Stage 1)
re_phase = re.compile(
    r"phase[_\-\s]*(1[a-c])",
    re.IGNORECASE,
)

# STAGE 1:  [Stage1] Epoch 0 done | avg_loss=1.2328 | time=7636s
re_stage1_epoch = re.compile(
    r"\[Stage1\]\s*Epoch\s*(\d+)\s*done\s*\|\s*avg_loss=([0-9.]+)\s*\|\s*time=(\d+)s",
    re.IGNORECASE,
)

# STAGE 2A:  [Stage2a] Epoch 0 | avg_loss=0.4772
# FIX Bug 1: match avg_loss OR avg_CE (the log uses avg_loss for Stage 2a)
re_stage2a_epoch = re.compile(
    r"\[Stage2a\]\s*Epoch\s*(\d+)\s*\|\s*avg_(?:loss|CE)=([0-9.]+)",
    re.IGNORECASE,
)

# STAGE 2B:  [Stage2b] Epoch 0 | avg_CE=0.5945 | avg_EWC=0.0000
re_stage2b_epoch = re.compile(
    r"\[Stage2b\]\s*Epoch\s*(\d+)\s*\|\s*avg_CE=([0-9.]+)",
    re.IGNORECASE,
)

# TIMESTAMP: extract from log line prefix "2026-04-26 07:24:51,499"
re_timestamp = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

# PRINTED EVAL
re_printed_eval = re.compile(
    r"printed.*?CER=([0-9.]+).*?WER=([0-9.]+)",
    re.IGNORECASE,
)

# HW EVAL
re_hw_eval = re.compile(
    r"(handwritten|hw).*?CER=([0-9.]+).*?WER=([0-9.]+)",
    re.IGNORECASE,
)


# ============================================================
# PARSE LOG
# ============================================================
records = {}
current_phase = None
last_key = None
epoch_timestamps = {}  # key -> datetime, for computing runtime from timestamps

with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for line in lines:
    # DETECT PHASE (only affects Stage 1)
    phase_match = re_phase.search(line)
    if phase_match:
        current_phase = phase_match.group(1).lower()

    # STAGE 1
    m = re_stage1_epoch.search(line)
    if m:
        epoch = int(m.group(1))
        avg_loss = float(m.group(2))
        runtime = int(m.group(3))
        key = ("stage1", current_phase, epoch)
        records[key] = {
            "stage": "stage1",
            "phase": current_phase,
            "epoch": epoch,
            "loss": avg_loss,
            "runtime": runtime,
            "printed_cer": None,
            "printed_wer": None,
            "hw_cer": None,
            "hw_wer": None,
        }
        last_key = key

    # STAGE 2A
    m = re_stage2a_epoch.search(line)
    if m:
        epoch = int(m.group(1))
        avg_loss = float(m.group(2))
        key = ("stage2a", "2a", epoch)
        records[key] = {
            "stage": "stage2a",
            "phase": "2a",
            "epoch": epoch,
            "loss": avg_loss,
            "runtime": None,
            "printed_cer": None,
            "printed_wer": None,
            "hw_cer": None,
            "hw_wer": None,
        }
        last_key = key
        # Capture timestamp for runtime computation
        ts_match = re_timestamp.match(line)
        if ts_match:
            epoch_timestamps[key] = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")

    # STAGE 2B
    m = re_stage2b_epoch.search(line)
    if m:
        epoch = int(m.group(1))
        avg_loss = float(m.group(2))
        key = ("stage2b", "2b", epoch)
        records[key] = {
            "stage": "stage2b",
            "phase": "2b",
            "epoch": epoch,
            "loss": avg_loss,
            "runtime": None,
            "printed_cer": None,
            "printed_wer": None,
            "hw_cer": None,
            "hw_wer": None,
        }
        last_key = key
        # Capture timestamp for runtime computation
        ts_match = re_timestamp.match(line)
        if ts_match:
            epoch_timestamps[key] = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")

    # PRINTED EVAL — attach to last parsed epoch
    pm = re_printed_eval.search(line)
    if pm and last_key is not None and last_key in records:
        records[last_key]["printed_cer"] = float(pm.group(1))
        records[last_key]["printed_wer"] = float(pm.group(2))

    # HW EVAL — attach to last parsed epoch
    hm = re_hw_eval.search(line)
    if hm and last_key is not None and last_key in records:
        records[last_key]["hw_cer"] = float(hm.group(2))
        records[last_key]["hw_wer"] = float(hm.group(3))


# ============================================================
# DATAFRAME CLEANING
# ============================================================
rows = list(records.values())
df = pd.DataFrame(rows)

if df.empty:
    print("ERROR: No matching records found in the log file.")
    print("Check that your log lines match the expected format, e.g.:")
    print("  [Stage1] Epoch 1 done | avg_loss=0.1234 | time=120s")
    print("  [Stage2a] Epoch 0 | avg_loss=0.4772")
    print("  [Stage2b] Epoch 0 | avg_CE=0.5945 | avg_EWC=0.0000")
    sys.exit(1)

df = df[
    (df["loss"].notna()) |
    (df["printed_cer"].notna()) |
    (df["hw_cer"].notna())
]

phase_order = {"1a": 0, "1b": 1, "1c": 2, "2a": 3, "2b": 4}
df["phase_order"] = df["phase"].map(phase_order)
df = df.sort_values(["phase_order", "epoch"]).reset_index(drop=True)
df["global_epoch"] = range(1, len(df) + 1)
df["loss_smooth"] = df["loss"].rolling(window=2, min_periods=1).mean()

# ---- Compute runtime from timestamps for Stage 2a/2b ----
# Sort timestamps by stage then epoch to compute deltas
for stage_name in ["stage2a", "stage2b"]:
    phase_name = "2a" if stage_name == "stage2a" else "2b"
    stage_keys = sorted(
        [k for k in epoch_timestamps if k[0] == stage_name],
        key=lambda k: k[2],  # sort by epoch number
    )
    for i in range(len(stage_keys)):
        key = stage_keys[i]
        if i == 0:
            # First epoch: no previous timestamp, skip or estimate
            continue
        prev_key = stage_keys[i - 1]
        # Skip if same epoch (duplicate from Colab restart)
        if key[2] == prev_key[2]:
            continue
        delta = (epoch_timestamps[key] - epoch_timestamps[prev_key]).total_seconds()
        # Only use reasonable deltas (< 6 hours, to filter out overnight gaps)
        if 0 < delta < 6 * 3600:
            if key in records:
                records[key]["runtime"] = int(delta)

# Rebuild df with computed runtimes
rows = list(records.values())
df = pd.DataFrame(rows)
df = df[
    (df["loss"].notna()) |
    (df["printed_cer"].notna()) |
    (df["hw_cer"].notna())
]
df["phase_order"] = df["phase"].map(phase_order)
df = df.sort_values(["phase_order", "epoch"]).reset_index(drop=True)
df["global_epoch"] = range(1, len(df) + 1)
df["loss_smooth"] = df["loss"].rolling(window=2, min_periods=1).mean()

print("\n==================== SUMMARY ====================")
print(df[["stage", "phase", "epoch", "loss", "runtime", "printed_cer", "hw_cer"]].to_string())
print(f"\nTotal valid epochs: {len(df)}")
print(f"Stages found: {df['stage'].unique().tolist()}")
print(f"Phases found: {df['phase'].unique().tolist()}")

csv_path = "parsed_training_metrics.csv"
df.to_csv(csv_path, index=False)
print(f"CSV saved to: {csv_path}")


# ============================================================
# PLOT HELPERS
# ============================================================
phase_colors = {
    "1a": "#dbeafe",
    "1b": "#dcfce7",
    "1c": "#fde68a",
    "2a": "#fbcfe8",
    "2b": "#ddd6fe",
}

phase_labels = {
    "1a": "Stage 1A\n(Words)",
    "1b": "Stage 1B\n(50% Pseudo-lines)",
    "1c": "Stage 1C\n(80% Pseudo-lines)",
    "2a": "Stage 2A\n(Printed Lines)",
    "2b": "Stage 2B\n(HW+Printed, EWC)",
}

# FIX Bug 3: Include ALL phases (including 1a) in the plot data
plot_df = df.copy()


def save_or_show(filename):
    if SAVE_CHARTS:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {filename}")
        plt.close()
    else:
        plt.show()


def add_phase_shading(ax, data, y_top_frac=0.97):
    """Add colored phase backgrounds and labels to any chart."""
    for phase in data["phase"].unique():
        subset = data[data["phase"] == phase]
        start = subset["global_epoch"].min()
        end = subset["global_epoch"].max()
        color = phase_colors.get(phase, "#e5e7eb")
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.18, zorder=0)
        label = phase_labels.get(phase, f"Phase {phase}")
        y_lim = ax.get_ylim()
        ax.text(
            (start + end) / 2, y_lim[0] + (y_lim[1] - y_lim[0]) * y_top_frac,
            label, ha="center", fontsize=9, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.5),
        )


def add_stage_boundaries(ax, data):
    """Add vertical lines at stage transitions."""
    phases_in_order = data.sort_values("global_epoch")["phase"].values
    for i in range(1, len(phases_in_order)):
        if phases_in_order[i] != phases_in_order[i - 1]:
            boundary_x = data.iloc[i]["global_epoch"] - 0.5
            # Thicker line for stage transitions (1→2a, 2a→2b)
            prev_phase = phases_in_order[i - 1]
            curr_phase = phases_in_order[i]
            is_stage_change = prev_phase[0] != curr_phase[0]
            lw = 2.5 if is_stage_change else 1.0
            ls = "-" if is_stage_change else ":"
            ax.axvline(x=boundary_x, color="#6b7280", linestyle=ls, linewidth=lw, alpha=0.6)


def add_best_marker(ax, data, col, label_prefix, offset_y=25):
    """Add a star marker at the best (minimum) value of a column."""
    valid = data[data[col].notna()]
    if len(valid) == 0:
        return
    best_idx = valid[col].idxmin()
    best_row = data.loc[best_idx]
    ax.scatter(best_row["global_epoch"], best_row[col], s=350, marker="*",
               color="#ef4444", zorder=10, edgecolors="black", linewidths=0.5)
    ax.annotate(
        f"BEST {label_prefix}\n{best_row[col]:.4f}\n({best_row['phase']} E{best_row['epoch']})",
        (best_row["global_epoch"], best_row[col]),
        textcoords="offset points", xytext=(15, offset_y), fontsize=9, weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ef4444", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#ef4444"),
    )


# ============================================================
# CER CHART
# ============================================================
fig, ax = plt.subplots(figsize=(20, 9))
x = plot_df["global_epoch"]

ax.plot(x, plot_df["printed_cer"], marker="o", markersize=5, linewidth=2.5,
        label="Printed CER", color="#2563eb", zorder=5)
ax.plot(x, plot_df["hw_cer"], marker="s", markersize=5, linewidth=2.5,
        linestyle="--", label="Handwritten CER", color="#dc2626", zorder=5)

add_phase_shading(ax, plot_df, y_top_frac=0.93)
add_stage_boundaries(ax, plot_df)
add_best_marker(ax, plot_df, "printed_cer", "PRINTED", offset_y=-40)
add_best_marker(ax, plot_df, "hw_cer", "HW", offset_y=25)

ax.set_title("Character Error Rate (CER) Across All Training Stages", fontsize=18, weight="bold", pad=15)
ax.set_xlabel("Global Training Epoch", fontsize=13)
ax.set_ylabel("CER", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc="upper right")
ax.set_xlim(0.5, len(plot_df) + 0.5)
fig.tight_layout()
save_or_show("chart_cer.png")


# ============================================================
# WER CHART (FIX Bug 4: Add full annotations like CER chart)
# ============================================================
fig, ax = plt.subplots(figsize=(20, 9))

ax.plot(x, plot_df["printed_wer"], marker="o", markersize=5, linewidth=2.5,
        label="Printed WER", color="#2563eb", zorder=5)
ax.plot(x, plot_df["hw_wer"], marker="s", markersize=5, linewidth=2.5,
        linestyle="--", label="Handwritten WER", color="#dc2626", zorder=5)

add_phase_shading(ax, plot_df, y_top_frac=0.93)
add_stage_boundaries(ax, plot_df)
add_best_marker(ax, plot_df, "printed_wer", "PRINTED", offset_y=-40)
add_best_marker(ax, plot_df, "hw_wer", "HW", offset_y=25)

ax.set_title("Word Error Rate (WER) Across All Training Stages", fontsize=18, weight="bold", pad=15)
ax.set_xlabel("Global Training Epoch", fontsize=13)
ax.set_ylabel("WER", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc="upper right")
ax.set_xlim(0.5, len(plot_df) + 0.5)
fig.tight_layout()
save_or_show("chart_wer.png")


# ============================================================
# LOSS CHART (now includes Stage 2a data)
# ============================================================
fig, ax = plt.subplots(figsize=(20, 8))

ax.plot(df["global_epoch"], df["loss"], alpha=0.35, marker="o", markersize=4,
        label="Raw Loss", color="#6b7280")
ax.plot(df["global_epoch"], df["loss_smooth"], linewidth=3,
        label="Smoothed Loss (window=2)", color="#7c3aed")

add_phase_shading(ax, df, y_top_frac=0.93)
add_stage_boundaries(ax, df)

ax.set_title("Training Loss Curve Across All Stages", fontsize=18, weight="bold", pad=15)
ax.set_xlabel("Global Training Epoch", fontsize=13)
ax.set_ylabel("Loss", fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_xlim(0.5, len(df) + 0.5)
fig.tight_layout()
save_or_show("chart_loss.png")


# ============================================================
# RUNTIME CHART — ALL STAGES
# ============================================================
runtime_df = df[df["runtime"].notna()].copy()

if len(runtime_df) > 0:
    fig, ax = plt.subplots(figsize=(20, 7))

    # Color-code by stage
    stage_colors_rt = {"stage1": "#059669", "stage2a": "#d946ef", "stage2b": "#6366f1"}
    for stage_name in runtime_df["stage"].unique():
        subset = runtime_df[runtime_df["stage"] == stage_name]
        label = {"stage1": "Stage 1 (Curriculum)", "stage2a": "Stage 2A (Printed Lines)", "stage2b": "Stage 2B (HW+Printed, EWC)"}.get(stage_name, stage_name)
        ax.plot(subset["global_epoch"], subset["runtime"], marker="o", markersize=6,
                linewidth=2.5, label=label, color=stage_colors_rt.get(stage_name, "gray"))

    add_phase_shading(ax, runtime_df, y_top_frac=0.90)
    add_stage_boundaries(ax, runtime_df)

    # Annotate average runtime per stage
    for stage_name in runtime_df["stage"].unique():
        subset = runtime_df[runtime_df["stage"] == stage_name]
        avg_rt = subset["runtime"].mean()
        mid_x = (subset["global_epoch"].min() + subset["global_epoch"].max()) / 2
        ax.annotate(
            f"avg: {avg_rt:.0f}s\n({avg_rt/60:.0f} min)",
            (mid_x, avg_rt), textcoords="offset points", xytext=(0, -35),
            fontsize=9, ha="center", weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_title("Training Runtime Per Epoch — All Stages", fontsize=18, weight="bold", pad=15)
    ax.set_xlabel("Global Training Epoch", fontsize=13)
    ax.set_ylabel("Seconds", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    save_or_show("chart_runtime.png")
else:
    print("No runtime data found — skipping runtime chart.")


print("\n==================== DONE ====================")
print(f"Charts {'saved' if SAVE_CHARTS else 'displayed'}.")
