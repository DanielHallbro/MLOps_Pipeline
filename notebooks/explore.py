"""
explore.py - First look at the raw UNSW-NB15 dataset.

Before touching the data, the official UNSW-NB15 feature documentation
(published by UNSW Canberra Cyber, the dataset's creators) was reviewed
to understand what each column represents - e.g. that sbytes/dbytes and
spkts/dpkts represent traffic in opposite directions (source-to-destination
vs. destination-to-source). This script verifies real patterns in the
data itself, rather than assuming the documentation tells the whole story.
"""

import pandas as pd

# ---------------------------------------------------------------
# STEP 1: Load the raw data and get a general overview
# ---------------------------------------------------------------
print("STEP 1: Loading raw data...")
train = pd.read_parquet("data/UNSW_NB15_training-set.parquet")
test = pd.read_parquet("data/UNSW_NB15_testing-set.parquet")
print(f"  Train: {train.shape}, Test: {test.shape}")
print(f"  Columns: {list(train.columns)}\n")

# ---------------------------------------------------------------
# STEP 2: Check the class balance (attack vs normal)
# ---------------------------------------------------------------
print("STEP 2: Checking attack vs. normal split...")
# label = 1 means attack, 0 means normal. Checked early because a
# heavily imbalanced dataset would need different handling (e.g.
# class_weight='balanced').
print(train["label"].value_counts(normalize=True))
print()

# ---------------------------------------------------------------
# STEP 3: Check every categorical (text) column for cardinality and
# unusual values
# ---------------------------------------------------------------
print("STEP 3: Checking categorical columns...")
categorical_cols = train.select_dtypes(include=["category", "object"]).columns.tolist()
print(f"  Categorical columns: {categorical_cols}\n")

for col in categorical_cols:
    n_unique = train[col].nunique()
    print(f"  '{col}': {n_unique} unique values")
    if n_unique <= 15:
        print(train[col].value_counts())
    else:
        print(f"    (too many to list, showing top 5)")
        print(train[col].value_counts().head(5))
    print()

# ---------------------------------------------------------------
# STEP 4: Confirm whether attack_cat can be used as a feature
# ---------------------------------------------------------------
print("STEP 4: Checking if attack_cat leaks the label...")
# Categorical columns with few unique values (like attack_cat, with
# just 10) are worth checking against the target directly, since a
# column that closely tracks the label can't be used for training.
leakage_check = pd.crosstab(train["attack_cat"], train["label"])
print(leakage_check)
print()
print("  Every 'Normal' row has label=0, and every attack category has")
print("  label=1. attack_cat is just a more detailed version of the")
print("  answer we're trying to predict - it can't be used as a")
print("  feature, and it would never exist for a real, unseen")
print("  connection anyway.\n")

# ---------------------------------------------------------------
# STEP 5: Check for missing values across the whole dataset
# ---------------------------------------------------------------
print("STEP 5: Checking for missing values...")
missing = train.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("  No NaN values found. (Note: 'service' uses '-' instead of")
    print("  NaN for unknowns, seen in step 3, so this check alone")
    print("  doesn't catch it.)\n")
else:
    print(missing, "\n")

# ---------------------------------------------------------------
# STEP 6: sbytes/dbytes and spkts/dpkts represent traffic in opposite
# directions (per the dataset documentation). A ratio between them
# could be a useful signal - check if that's actually feasible first.
# ---------------------------------------------------------------
print("STEP 6: Checking feasibility of a source/destination traffic ratio...")
dbytes_zero = (train["dbytes"] == 0).sum()
dpkts_zero = (train["dpkts"] == 0).sum()
print(f"  dbytes == 0: {dbytes_zero} rows ({dbytes_zero / len(train) * 100:.1f}%)")
print(f"  dpkts == 0: {dpkts_zero} rows ({dpkts_zero / len(train) * 100:.1f}%)")
print("  Zero values are common enough that a plain division would")
print("  crash or produce infinities - any ratio feature needs a")
print("  safeguard against dividing by zero.\n")

# ---------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------
print("=" * 60)
print("EXPLORATION SUMMARY")
print("=" * 60)
print(f"1. Dataset: {len(train)} train / {len(test)} test rows, {len(train.columns)} columns")
print(f"2. Class balance: {train['label'].value_counts(normalize=True).round(3).to_dict()}")
print(f"3. attack_cat perfectly predicts label -> must be dropped (target leakage)")
print(f"4. 'service' uses '-' for {(train['service']=='-').sum()} rows -> a real category, not missing data")
print(f"5. 'proto' has {train['proto'].nunique()} unique values -> likely needs grouping")
print(f"6. dbytes/dpkts contain zeros -> any ratio feature needs a +1 safeguard")