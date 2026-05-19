"""
PLOT_HERO.py — Generate publication-quality figures from experiment results.
Reads from results/phase1_verifier_10rounds/ and results/phase2_replay_buffer_10rounds/
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12, 'axes.titlesize': 13, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BASE = Path(__file__).parent
P1 = BASE / "results" / "phase1_verifier_10rounds"
P2 = BASE / "results" / "phase2_replay_buffer_10rounds"
OUT = BASE / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_metrics(path):
    rows = []
    for line in (path / "round_metrics.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    # deduplicate by round (keep last entry for each round)
    seen = {}
    for r in rows:
        seen[r["round"]] = r
    return [seen[k] for k in sorted(seen.keys())]


p1 = load_metrics(P1)
p2 = load_metrics(P2)

rounds_p1 = [r["round"] for r in p1]
rounds_p2 = [r["round"] for r in p2]

# ══════════════════════════════════════════════════════════════
# FIGURE 1: Aggregate Safety Drift (%)
# ══════════════════════════════════════════════════════════════
p1_drift = [r["drift"]["aggregate"]["abs_drift"] * 100 for r in p1]
p2_drift = [r["drift"]["aggregate"]["abs_drift"] * 100 for r in p2]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(rounds_p1, p1_drift, color='#d62728', linestyle='--', marker='X',
        markersize=8, linewidth=2.2, label='Phase 1 (Verifier Only)', zorder=3)
ax.plot(rounds_p2, p2_drift, color='#1f77b4', linestyle='-', marker='o',
        markersize=7, linewidth=2.2, label='Phase 2 (Verifier + Replay Buffer)', zorder=3)
ax.axhline(0, color='grey', linewidth=0.8, linestyle='-', alpha=0.6)
ax.fill_between(rounds_p1, p1_drift, alpha=0.08, color='#d62728')
ax.fill_between(rounds_p2, p2_drift, alpha=0.08, color='#1f77b4')
ax.set_title('Aggregate Safety Drift Across 10 Self-Improvement Rounds', fontweight='bold', pad=12)
ax.set_xlabel('Self-Improvement Round')
ax.set_ylabel('Aggregate Safety Drift (%)\n(Higher = Less Safe)')
ax.set_xticks(range(10))
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='upper left', frameon=True, shadow=True, fancybox=True)
fig.tight_layout()
fig.savefig(OUT / "fig1_aggregate_safety_drift.pdf")
fig.savefig(OUT / "fig1_aggregate_safety_drift.png")
print(f"[SAVED] fig1_aggregate_safety_drift")

# ══════════════════════════════════════════════════════════════
# FIGURE 2: Math Accuracy Over Rounds
# ══════════════════════════════════════════════════════════════
p1_math = [r["math_pass1"]["accuracy"] * 100 for r in p1]
p2_math = [r["math_pass1"]["accuracy"] * 100 for r in p2]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(rounds_p1, p1_math, color='#d62728', linestyle='--', marker='X',
        markersize=8, linewidth=2.2, label='Phase 1 (Verifier Only)')
ax.plot(rounds_p2, p2_math, color='#1f77b4', linestyle='-', marker='o',
        markersize=7, linewidth=2.2, label='Phase 2 (Verifier + Replay Buffer)')
ax.set_title('GSM8K Math Accuracy (pass@1) Over Training Rounds', fontweight='bold', pad=12)
ax.set_xlabel('Self-Improvement Round')
ax.set_ylabel('Math Accuracy (%)')
ax.set_xticks(range(10))
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='lower right', frameon=True, shadow=True, fancybox=True)
fig.tight_layout()
fig.savefig(OUT / "fig2_math_accuracy.pdf")
fig.savefig(OUT / "fig2_math_accuracy.png")
print(f"[SAVED] fig2_math_accuracy")

# ══════════════════════════════════════════════════════════════
# FIGURE 3: Per-Benchmark Safety ASR
# ══════════════════════════════════════════════════════════════
benchmarks = ["advbench_subset", "xstest_subset", "donotanswer_subset"]
bench_labels = ["AdvBench", "XSTEST", "DoNotAnswer"]
colors_p1 = ['#e45756', '#f58518', '#b07aa1']
colors_p2 = ['#4c78a8', '#54a24b', '#72b7b2']

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for i, (bench, label) in enumerate(zip(benchmarks, bench_labels)):
    ax = axes[i]
    p1_asr = [r["safety_asr"][bench]["asr_proxy"] * 100 for r in p1]
    p2_asr = [r["safety_asr"][bench]["asr_proxy"] * 100 for r in p2]
    ax.plot(rounds_p1, p1_asr, color='#d62728', linestyle='--', marker='X',
            markersize=7, linewidth=2, label='P1 (Verifier Only)')
    ax.plot(rounds_p2, p2_asr, color='#1f77b4', linestyle='-', marker='o',
            markersize=6, linewidth=2, label='P2 (V + Replay)')
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('Round')
    ax.set_xticks(range(10))
    ax.grid(True, linestyle=':', alpha=0.5)
    if i == 0:
        ax.set_ylabel('ASR Proxy (%)')
    if i == 2:
        ax.legend(loc='upper left', fontsize=8, frameon=True)
fig.suptitle('Per-Benchmark Safety Attack Success Rate', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig3_per_benchmark_asr.pdf")
fig.savefig(OUT / "fig3_per_benchmark_asr.png")
print(f"[SAVED] fig3_per_benchmark_asr")

# ══════════════════════════════════════════════════════════════
# FIGURE 4: Accepted Training Samples Per Round
# ══════════════════════════════════════════════════════════════
p1_accepted = [r["accepted"] for r in p1]
p2_accepted = [r["accepted"] for r in p2]

fig, ax = plt.subplots(figsize=(8, 4.5))
all_rounds = sorted(set(rounds_p1) | set(rounds_p2))
x = np.arange(len(all_rounds))
width = 0.35
p1_map = {r["round"]: r["accepted"] for r in p1}
p2_map = {r["round"]: r["accepted"] for r in p2}
p1_vals = [p1_map.get(r, 0) for r in all_rounds]
p2_vals = [p2_map.get(r, 0) for r in all_rounds]
ax.bar(x - width/2, p1_vals, width, color='#d62728', alpha=0.75, label='Phase 1', edgecolor='white')
ax.bar(x + width/2, p2_vals, width, color='#1f77b4', alpha=0.75, label='Phase 2', edgecolor='white')
ax.set_title('Accepted Training Samples Per Round', fontweight='bold', pad=12)
ax.set_xlabel('Self-Improvement Round')
ax.set_ylabel('Samples Accepted by Verifier')
ax.set_xticks(x)
ax.set_xticklabels(all_rounds)
ax.grid(True, linestyle=':', alpha=0.5, axis='y')
ax.legend(frameon=True, shadow=True)
fig.tight_layout()
fig.savefig(OUT / "fig4_accepted_samples.pdf")
fig.savefig(OUT / "fig4_accepted_samples.png")
print(f"[SAVED] fig4_accepted_samples")

# ══════════════════════════════════════════════════════════════
# FIGURE 5: Combined — Math vs Safety Tradeoff (Dual Axis)
# ══════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

l1 = ax1.plot(rounds_p1, p1_math, color='#d62728', linestyle='--', marker='X',
              markersize=8, linewidth=2, label='P1 Math Acc.')
l2 = ax1.plot(rounds_p2, p2_math, color='#1f77b4', linestyle='-', marker='o',
              markersize=7, linewidth=2, label='P2 Math Acc.')
l3 = ax2.plot(rounds_p1, p1_drift, color='#d62728', linestyle=':', marker='s',
              markersize=6, linewidth=1.5, alpha=0.7, label='P1 Safety Drift')
l4 = ax2.plot(rounds_p2, p2_drift, color='#1f77b4', linestyle=':', marker='D',
              markersize=6, linewidth=1.5, alpha=0.7, label='P2 Safety Drift')

ax1.set_xlabel('Self-Improvement Round')
ax1.set_ylabel('Math Accuracy (%)', color='black')
ax2.set_ylabel('Safety Drift (%)', color='grey')
ax2.axhline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.4)
ax1.set_xticks(range(10))
ax1.grid(True, linestyle=':', alpha=0.4)

lines = l1 + l2 + l3 + l4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', fontsize=9, frameon=True, shadow=True)

ax1.set_title('Math Improvement vs Safety Drift Tradeoff', fontweight='bold', pad=12)
fig.tight_layout()
fig.savefig(OUT / "fig5_math_vs_safety_tradeoff.pdf")
fig.savefig(OUT / "fig5_math_vs_safety_tradeoff.png")
print(f"[SAVED] fig5_math_vs_safety_tradeoff")

# ══════════════════════════════════════════════════════════════
# Summary Table
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  EXPERIMENT SUMMARY")
print("="*60)
print(f"  {'Metric':<30} {'Phase 1':>12} {'Phase 2':>12}")
print(f"  {'-'*30} {'-'*12} {'-'*12}")
print(f"  {'Math Acc. (Round 0)':<30} {p1_math[0]:>11.1f}% {p2_math[0]:>11.1f}%")
print(f"  {'Math Acc. (Round 9)':<30} {p1_math[-1]:>11.1f}% {p2_math[-1]:>11.1f}%")
print(f"  {'Math Improvement':<30} {p1_math[-1]-p1_math[0]:>+11.1f}% {p2_math[-1]-p2_math[0]:>+11.1f}%")
print(f"  {'Safety Drift (Round 9)':<30} {p1_drift[-1]:>+11.1f}% {p2_drift[-1]:>+11.1f}%")
print(f"  {'Max Safety Drift':<30} {max(p1_drift):>+11.1f}% {max(p2_drift):>+11.1f}%")
print("="*60)
print(f"\nAll figures saved to: {OUT}")