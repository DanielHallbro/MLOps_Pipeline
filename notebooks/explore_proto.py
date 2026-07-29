"""
Investigates the distribution of the 'proto' column before deciding
how to group rare protocols for one-hot encoding.

Why this matters: 'proto' has 133 unique values (see explore.py).
One-hot encoding all 133 as-is would create 133 new columns, which
would make the feature space unnecessarily wide and hard to justify.
This script walks through the reasoning: first look at the overall
shape of the distribution, then use that to pick a concrete,
defensible threshold for grouping rare protocols into "other".
"""
import pandas as pd

train = pd.read_parquet("data/UNSW_NB15_training-set.parquet")
total_rows = len(train)

proto_counts = train['proto'].value_counts()

print("=== STEP 1: Raw frequency distribution ===")
print(proto_counts)
print(f"\nTotal unique protocols: {proto_counts.shape[0]}")

# STEP 2: Check how concentrated the distribution is. If a handful of
# protocols already account for most of the data, that's a strong
# signal we're dealing with a long-tail distribution, which is the
# standard case for grouping rare categories into "other".
print("\n=== STEP 2: Cumulative coverage ===")
cumulative_pct = (proto_counts.cumsum() / proto_counts.sum() * 100)
for threshold in [90, 95, 99, 99.9]:
    n_protocols = (cumulative_pct <= threshold).sum() + 1
    print(f"Protocols needed to cover {threshold}% of the data: {n_protocols}")

# STEP 3: Cumulative coverage tells us the shape, but not where to
# draw the line for an individual protocol. Here we check, for a few
# candidate thresholds, how many protocols individually clear that
# bar on their own. This gives a concrete number of resulting
# one-hot columns for each candidate threshold.
print("\n=== STEP 3: Protocols individually clearing each threshold ===")
for pct in [1.0, 0.5, 0.1]:
    row_threshold = total_rows * (pct / 100)
    n_above = (proto_counts >= row_threshold).sum()
    print(f"Threshold {pct}% of rows -> {n_above} protocols keep their own column")

# DECISION: 0.5% was chosen as the cutoff. At this threshold, ~6
# protocols keep their own one-hot column and everything else is
# grouped into "other". This is a large reduction from 133 raw
# categories while still keeping every protocol that has a
# meaningful presence in the data (>= ~880 rows out of 175,341).
# The threshold is applied as a percentage computed from the data,
# not a hardcoded list of protocol names, so the logic generalizes
# to other datasets or a different train/test split.
FINAL_THRESHOLD_PCT = 0.5
row_threshold = total_rows * (FINAL_THRESHOLD_PCT / 100)
kept_protocols = proto_counts[proto_counts >= row_threshold].index.tolist()

print(f"\n=== FINAL DECISION: {FINAL_THRESHOLD_PCT}% threshold ===")
print(f"Protocols kept as their own category: {kept_protocols}")
print(f"All other protocols will be grouped into 'other' in preprocess.py")