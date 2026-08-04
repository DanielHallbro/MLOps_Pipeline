#!/usr/bin/env python3
"""
plot_hpa_scaling.py - turns the CSV log from k8s_stress_test.sh into
one chart: replica count (step line, left axis) and CPU utilization
(dashed line, right axis, with the 60% HPA target marked) over time,
with the load window shaded. This is the real evidence behind "it
scales up and back down", built from an actual run, not a mockup.

Usage: python3 scripts/plot_hpa_scaling.py <log_csv> <output_png>
"""
import csv
import sys

import matplotlib.pyplot as plt

log_path, output_path = sys.argv[1], sys.argv[2]

elapsed, replicas, cpu, phase = [], [], [], []
with open(log_path) as f:
    for row in csv.DictReader(f):
        elapsed.append(int(row["elapsed_seconds"]))
        replicas.append(int(row["replicas"]))
        raw_cpu = row["cpu_percent"]
        cpu.append(float(raw_cpu) if raw_cpu not in ("", "?") else None)
        phase.append(row["phase"])

load_end = max((e for e, p in zip(elapsed, phase) if p == "load"), default=0)

fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.axvspan(0, load_end, color="#ffe8b3", alpha=0.5, label="Load generator running")

ax1.step(elapsed, replicas, where="post", color="#1f77b4", linewidth=2.2, label="Replicas")
ax1.set_xlabel("Seconds since load test started")
ax1.set_ylabel("Replica count", color="#1f77b4")
ax1.set_ylim(0, max(replicas) + 1)
ax1.set_yticks(range(0, max(replicas) + 2))
ax1.tick_params(axis="y", labelcolor="#1f77b4")

ax2 = ax1.twinx()
cpu_x = [e for e, c in zip(elapsed, cpu) if c is not None]
cpu_y = [c for c in cpu if c is not None]
if cpu_x:
    ax2.plot(cpu_x, cpu_y, color="#d62728", linewidth=1.5, linestyle="--", label="CPU %")
ax2.axhline(60, color="#d62728", linewidth=0.8, linestyle=":", alpha=0.7)
ax2.text(elapsed[-1], 61, "60% target", color="#d62728", fontsize=8, ha="right")
ax2.set_ylabel("CPU utilization (%)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")

fig.suptitle("HPA scaling FastAPI under load")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(output_path, dpi=150)
print(f"Saved {output_path}")