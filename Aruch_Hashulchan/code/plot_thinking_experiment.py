"""
Plot thinking level experiment results.
Computes costs using Gemini 3 Flash pricing:
  Input tokens:    $0.50 / 1M
  Output tokens:   $3.00 / 1M
  Thinking tokens: $3.00 / 1M (billed at output rate)

Saves: thinking_experiment_results/thinking_level_analysis.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── Pricing ───────────────────────────────────────────────────────────────────
INPUT_PRICE_PER_M  = 0.50   # $ per 1M input tokens
OUTPUT_PRICE_PER_M = 3.00   # $ per 1M output/thinking tokens

TOTAL_SEIFIM = 23_800  # approx full corpus

LEVELS = ["MINIMAL", "LOW", "MEDIUM", "HIGH"]
COLORS = {
    "MINIMAL": "#4C9BE8",
    "LOW":     "#5DBB63",
    "MEDIUM":  "#F5A623",
    "HIGH":    "#E8534C",
}

# ── Load data ─────────────────────────────────────────────────────────────────

with open("thinking_experiment_results/results.json") as f:
    data = json.load(f)

def cost_per_call(d):
    tokens_in       = d.get("tokens_in") or 0
    tokens_out      = d.get("tokens_out") or 0
    tokens_thinking = d.get("tokens_thinking") or 0
    input_cost   = tokens_in       / 1_000_000 * INPUT_PRICE_PER_M
    output_cost  = tokens_out      / 1_000_000 * OUTPUT_PRICE_PER_M
    think_cost   = tokens_thinking / 1_000_000 * OUTPUT_PRICE_PER_M
    return input_cost + output_cost + think_cost

# Aggregate per level
stats = defaultdict(lambda: {
    "thinking": [], "latency": [], "cost": [],
    "tokens_in": [], "tokens_out": [],
})

for section, samples in data.items():
    for s in samples:
        for level in LEVELS:
            d = s["levels"][level]
            stats[level]["thinking"].append(d.get("tokens_thinking") or 0)
            stats[level]["latency"].append(d.get("latency_s") or 0)
            stats[level]["cost"].append(cost_per_call(d))
            stats[level]["tokens_in"].append(d.get("tokens_in") or 0)
            stats[level]["tokens_out"].append(d.get("tokens_out") or 0)

avg = {}
for level in LEVELS:
    s = stats[level]
    avg[level] = {
        "thinking":    np.mean(s["thinking"]),
        "latency":     np.mean(s["latency"]),
        "cost":        np.mean(s["cost"]),
        "tokens_in":   np.mean(s["tokens_in"]),
        "tokens_out":  np.mean(s["tokens_out"]),
        "proj_cost":   np.mean(s["cost"]) * TOTAL_SEIFIM,
        "proj_hours":  np.mean(s["latency"]) * TOTAL_SEIFIM / 3600,
    }

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "Gemini 3 Flash — Thinking Level Experiment\n"
    f"20 sampled seifim (5 per section: OC, YD, EH, CM) | "
    f"Pricing: input $0.50/M, output+thinking $3.00/M",
    fontsize=13, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

x = np.arange(len(LEVELS))
bar_width = 0.55
colors = [COLORS[l] for l in LEVELS]


def bar_chart(ax, values, title, ylabel, fmt="{:.0f}", add_labels=True):
    bars = ax.bar(x, values, width=bar_width, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if add_labels:
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                fmt.format(val),
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
    return bars


# 1. Average thinking tokens
ax1 = fig.add_subplot(gs[0, 0])
bar_chart(
    ax1,
    [avg[l]["thinking"] for l in LEVELS],
    "Avg Thinking Tokens per Seif",
    "Tokens",
    fmt="{:,.0f}"
)

# 2. Average latency
ax2 = fig.add_subplot(gs[0, 1])
bar_chart(
    ax2,
    [avg[l]["latency"] for l in LEVELS],
    "Avg Latency per Seif",
    "Seconds",
    fmt="{:.1f}s"
)

# 3. Token breakdown (stacked: input / output / thinking)
ax3 = fig.add_subplot(gs[0, 2])
in_vals  = [avg[l]["tokens_in"]   for l in LEVELS]
out_vals = [avg[l]["tokens_out"]  for l in LEVELS]
thi_vals = [avg[l]["thinking"]    for l in LEVELS]

b1 = ax3.bar(x, in_vals,  width=bar_width, label="Input",    color="#6BAED6", edgecolor="white")
b2 = ax3.bar(x, out_vals, width=bar_width, label="Output",   color="#74C476", edgecolor="white", bottom=in_vals)
b3 = ax3.bar(x, thi_vals, width=bar_width, label="Thinking", color="#FD8D3C", edgecolor="white",
             bottom=[i + o for i, o in zip(in_vals, out_vals)])
ax3.set_xticks(x)
ax3.set_xticklabels(LEVELS, fontsize=10)
ax3.set_title("Avg Token Breakdown per Seif", fontsize=11, fontweight="bold", pad=8)
ax3.set_ylabel("Tokens", fontsize=9)
ax3.legend(fontsize=8, loc="upper left")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# 4. Cost per seif
ax4 = fig.add_subplot(gs[1, 0])
bar_chart(
    ax4,
    [avg[l]["cost"] * 1000 for l in LEVELS],  # in millicents for readability
    "Avg Cost per Seif",
    "Cost (¢ × 10⁻³)",
    fmt="${:.4f}"
)
# override label to show actual cents
for bar, level in zip(ax4.patches, LEVELS):
    pass  # already labeled by bar_chart

# redo labels with $ sign
ax4.cla()
vals = [avg[l]["cost"] for l in LEVELS]
bars = ax4.bar(x, vals, width=bar_width, color=colors, edgecolor="white", linewidth=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(LEVELS, fontsize=10)
ax4.set_title("Avg Cost per Seif", fontsize=11, fontweight="bold", pad=8)
ax4.set_ylabel("Cost (USD)", fontsize=9)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
for bar, val in zip(bars, vals):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.02,
        f"${val:.5f}",
        ha="center", va="bottom", fontsize=8, fontweight="bold"
    )

# 5. Projected total cost (full corpus)
ax5 = fig.add_subplot(gs[1, 1])
proj_costs = [avg[l]["proj_cost"] for l in LEVELS]
bars5 = ax5.bar(x, proj_costs, width=bar_width, color=colors, edgecolor="white", linewidth=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(LEVELS, fontsize=10)
ax5.set_title(f"Projected Total Cost\n({TOTAL_SEIFIM:,} seifim)", fontsize=11, fontweight="bold", pad=8)
ax5.set_ylabel("Cost (USD)", fontsize=9)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
for bar, val in zip(bars5, proj_costs):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.02,
        f"${val:.0f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold"
    )

# 6. Projected total wall-clock time
ax6 = fig.add_subplot(gs[1, 2])
proj_hours = [avg[l]["proj_hours"] for l in LEVELS]
bars6 = ax6.bar(x, proj_hours, width=bar_width, color=colors, edgecolor="white", linewidth=0.8)
ax6.set_xticks(x)
ax6.set_xticklabels(LEVELS, fontsize=10)
ax6.set_title(f"Projected Total Run Time\n({TOTAL_SEIFIM:,} seifim, sequential)", fontsize=11, fontweight="bold", pad=8)
ax6.set_ylabel("Hours", fontsize=9)
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)
for bar, val in zip(bars6, proj_hours):
    label = f"{val:.1f}h" if val < 24 else f"{val/24:.1f}d"
    ax6.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.02,
        label,
        ha="center", va="bottom", fontsize=9, fontweight="bold"
    )

out_path = "thinking_experiment_results/thinking_level_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")

# Print summary table
print("\nSUMMARY TABLE")
print(f"{'Level':<10} {'Avg thinking':>14} {'Avg latency':>13} {'Cost/seif':>12} {'Proj cost':>11} {'Proj time':>11}")
print("-" * 75)
for level in LEVELS:
    a = avg[level]
    proj_time = f"{a['proj_hours']:.1f}h" if a['proj_hours'] < 24 else f"{a['proj_hours']/24:.1f}d"
    print(
        f"{level:<10} {a['thinking']:>14,.0f} {a['latency']:>12.1f}s "
        f"${a['cost']:>10.5f} ${a['proj_cost']:>10.0f} {proj_time:>11}"
    )
