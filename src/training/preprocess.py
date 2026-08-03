"""
preprocess.py - Cleans and prepares the UNSW-NB15 dataset for training.

Four decisions were made after exploring the data. Full reasoning for
each lives next to the code that implements it below. Quick summary:

1. Drop attack_cat and 6 other columns (attack_cat leaks the answer).
2. service '-' becomes 'unknown' (it's a real category, not missing data).
3. Rare protocols get grouped into 'other' (long-tail distribution).
4. Two ratio features added to capture traffic asymmetry.

Golden rule followed throughout: everything we "learn" from the data
(which protocols are common, average values for scaling) is learned
from the TRAINING set only, then applied to the test set unchanged.
This is like training for an exam using only old exams, never peeking
at tomorrow's exam questions. If we let information from the test set
leak into preparation, our results would look better than they'd
actually be in the real world.
"""

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# STEP 1: Load the raw data
# ---------------------------------------------------------------
print("STEP 1: Loading raw data...")
train = pd.read_parquet("data/UNSW_NB15_training-set.parquet")
test = pd.read_parquet("data/UNSW_NB15_testing-set.parquet")
print(f"  Loaded {len(train)} training rows and {len(test)} test rows.\n")

# ---------------------------------------------------------------
# STEP 2: Drop columns we don't want the model to see
# ---------------------------------------------------------------
print("STEP 2: Dropping columns...")
# attack_cat directly reveals whether a row is an attack or not - the
# model would just read the answer instead of learning real patterns.
# It also would never exist in a real, live prediction request.
# The other 6 columns were excluded per the original project plan.
COLUMNS_TO_DROP = [
    "attack_cat", "stcpb", "dtcpb", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "is_sm_ips_ports"
]
train = train.drop(columns=COLUMNS_TO_DROP)
test = test.drop(columns=COLUMNS_TO_DROP)
print(f"  Dropped: {COLUMNS_TO_DROP}\n")

# ---------------------------------------------------------------
# STEP 3: Fix the 'service' column
# ---------------------------------------------------------------
print("STEP 3: Cleaning up 'service' column...")
# '-' shows up in 54% of rows. It's not missing data - it means "no
# app-level protocol detected" (normal for things like ICMP or ARP).
# We rename it to 'unknown' so it reads clearly as its own category,
# instead of dropping over half our dataset.
train["service"] = train["service"].astype(str).replace("-", "unknown")
test["service"] = test["service"].astype(str).replace("-", "unknown")
print("  '-' renamed to 'unknown', kept as a real category.\n")

# ---------------------------------------------------------------
# STEP 4: Group rare network protocols together
# ---------------------------------------------------------------
print("STEP 4: Grouping rare protocols into 'other'...")
# 'proto' has 133 different values, but just a handful cover almost
# all the traffic (tcp, udp, etc). One-hot encoding all 133 as-is
# would create 133 mostly-empty columns. Instead, any protocol making
# up less than 0.5% of TRAINING rows gets grouped into 'other'.
# The threshold is calculated from train data only (see golden rule
# above) so this logic would work the same on a fresh dataset too.
PROTO_RARITY_THRESHOLD_PCT = 0.5
proto_counts = train["proto"].value_counts()
row_threshold = len(train) * (PROTO_RARITY_THRESHOLD_PCT / 100)
common_protocols = set(proto_counts[proto_counts >= row_threshold].index)

def group_rare_protocols(df, common_set):
    return df["proto"].astype(str).apply(
        lambda p: p if p in common_set else "other"
    )

train["proto"] = group_rare_protocols(train, common_protocols)
test["proto"] = group_rare_protocols(test, common_protocols)
print(f"  Protocols kept individually: {sorted(common_protocols)}")
# Save the list so it can be logged as an artifact and reused
# identically by the API later - without this, the API has no way
# to know which protocols were "common enough" at training time.
import json

os.makedirs("models/preprocessing", exist_ok=True)
with open("models/preprocessing/common_protocols.json", "w") as f:
    json.dump(sorted(common_protocols), f)
print("  Everything else grouped into 'other'.\n")

# ---------------------------------------------------------------
# STEP 5: Add two new features (feature engineering)
# ---------------------------------------------------------------
print("STEP 5: Adding traffic ratio features...")
# sbytes/spkts = data sent FROM source TO destination.
# dbytes/dpkts = data sent BACK from destination to source.
# Normal traffic sends little, gets a lot back (small ratio).
# Attacks (floods, scans) often break that pattern (large or
# near-zero ratio). These two ratios give the model that signal
# directly instead of making it infer it from raw counts.
#
# +1 in the denominator avoids dividing by zero (dbytes/dpkts is 0
# in ~48% of rows).
for df in (train, test):
    df["sbytes_dbytes_ratio"] = df["sbytes"] / (df["dbytes"] + 1)
    df["spkts_dpkts_ratio"] = df["spkts"] / (df["dpkts"] + 1)
print("  Added: sbytes_dbytes_ratio, spkts_dpkts_ratio\n")

# ---------------------------------------------------------------
# STEP 6: One-hot encode the text columns
# ---------------------------------------------------------------
print("STEP 6: One-hot encoding proto / service / state...")
# Switched from pd.get_dummies() to sklearn's OneHotEncoder. The old
# approach can't be reused later - it just reads whatever categories
# happen to exist in the dataframe you pass it. OneHotEncoder is a
# proper fit/transform object: fit once on train, then reused as-is
# on test now, and later reused by FastAPI to encode a single live
# request the exact same way.
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = ["proto", "service", "state"]

# handle_unknown="ignore" means if a live request someday has a
# category the encoder never saw during training, it gets encoded as
# all zeros instead of crashing the API.
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(train[CATEGORICAL_COLS])

def encode_categoricals(df, encoder, cat_cols):
    encoded = encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index,
    )
    return pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

train_encoded = encode_categoricals(train, encoder, CATEGORICAL_COLS)
test_encoded = encode_categoricals(test, encoder, CATEGORICAL_COLS)
print(f"  Done. Train shape: {train_encoded.shape}, test shape: {test_encoded.shape}\n")

# ---------------------------------------------------------------
# STEP 7: Scale numeric columns
# ---------------------------------------------------------------
print("STEP 7: Scaling numeric features...")
numeric_cols = train_encoded.select_dtypes(
    include=["int64", "int32", "int16", "int8", "float32", "float64"]
).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "label"]

scaler = StandardScaler()
train_encoded[numeric_cols] = scaler.fit_transform(train_encoded[numeric_cols])
test_encoded[numeric_cols] = scaler.transform(test_encoded[numeric_cols])
print(f"  Scaled {len(numeric_cols)} numeric columns.\n")

# ---------------------------------------------------------------
# STEP 8: Save the processed data AND the fitted preprocessing objects
# ---------------------------------------------------------------
print("STEP 8: Saving processed data and preprocessing objects...")
import joblib

os.makedirs("data/processed", exist_ok=True)
train_encoded.to_parquet("data/processed/train.parquet", index=False)
test_encoded.to_parquet("data/processed/test.parquet", index=False)

# Saved separately from data/processed/ since these aren't data, they're
# fitted transformation objects. Training scripts load these and log
# them as MLflow artifacts alongside whichever model they train, so
# FastAPI can later pull the exact same scaler/encoder used at
# training time for any given run.
os.makedirs("models/preprocessing", exist_ok=True)
joblib.dump(scaler, "models/preprocessing/scaler.pkl")
joblib.dump(encoder, "models/preprocessing/encoder.pkl")
print("  Saved data/processed/train.parquet and test.parquet")
print("  Saved models/preprocessing/scaler.pkl and encoder.pkl\n")
# ---------------------------------------------------------------
# SUMMARY - what actually happened, in plain language
# ---------------------------------------------------------------
print("=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print("1. Dropped 7 columns, including attack_cat (target leakage)")
print("2. 'service' = '-' relabeled as 'unknown' (a real category)")
print(f"3. Rare protocols grouped into 'other' ({len(common_protocols)} kept individually)")
print("4. Added 2 ratio features for traffic asymmetry")
print("5. One-hot encoded proto/service/state, scaled numeric columns")
print(f"6. Final shape -> train: {train_encoded.shape}, test: {test_encoded.shape}")