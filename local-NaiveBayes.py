"""
IST3134 Big Data Analytics - Group Assignment
Non-Big-Data baseline: single-threaded Naive Bayes on NYC Parking Violations
============================== HOW TO RUN IN VS CODE ==============================

1. Install Python if you don't already have it (python.org), then open this
   file's folder in VS Code.

2. Open a NEW TERMINAL in VS Code (Terminal menu -> New Terminal) and run:

       pip install pandas numpy scikit-learn

   (If you have both Python 2 and 3, or multiple versions installed, you may
   need "pip3 install ..." instead. Also make sure VS Code's selected Python
   interpreter, bottom-right corner, matches the one you installed packages
   into — click it to switch if needed.)

3. Edit the CSV_PATH variable just below these instructions so it points at
   your downloaded dataset file (see "WHERE TO GET THE DATA" below).

4. Click the "Run Python File" triangle button (top-right of the editor),
   or press Ctrl+F5 (Cmd+F5 on Mac). All output prints straight to the
   terminal — nothing is written to disk.

============================== WHERE TO GET THE DATA ==============================

Dataset: NYC Open Data — "Parking Violations Issued" (one CSV per fiscal
year). Search "NYC Parking Violations" on the NYC Open Data portal
(data.cityofnewyork.us) or on Kaggle for a mirrored copy.

The code in this script is written to work with the 2015 fiscal year .csv file.
Kaggle link: https://www.kaggle.com/datasets/new-york-city/nyc-parking-tickets

============================== NOTES ==============================

pandas.read_csv loads the ENTIRE file into memory on a single machine —
this is exactly the limitation that Big Data tools like Spark exist to
avoid. If this script is slow or runs out of memory on a full-size file,
that contrast (plain Python struggling vs. Spark handling it) is itself
useful material for your report's comparison/analysis section.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ============================================================================
# CONFIG — edit this before hitting Run
# ============================================================================
CSV_PATH = r"C:\Users\Chen Wei\Desktop\Uni Degree\Y3S2\IST3134 - Big Data Analytics\GrpAsg\ParkingTicket\Parking_Violations_Issued_-_Fiscal_Year_2015.csv"   # <-- put the path to your downloaded CSV here
TOP_N_CLASSES = 8                      # number of most-frequent violation codes kept as their own class
CURRENT_YEAR = 2026
# ============================================================================

CATEGORICAL_FEATURES = [
    "Registration State",
    "Plate Type",
    "Vehicle Body Type",
    "Vehicle Make",
    "Issuing Agency",
    "Violation County",
    "Vehicle Color",
    "Unregistered Vehicle?",
    "Violation Precinct",
]


def parse_hour_bucket(violation_time):
    """Violation Time is encoded like '0813A' / '1245P' (HHMM + AM/PM
    letter). Returns a coarse 4-way time-of-day bucket, or 'UNKNOWN' if the
    value is blank/malformed."""
    if not isinstance(violation_time, str):
        return "UNKNOWN"
    s = violation_time.strip().upper()
    if len(s) != 5 or not s[:4].isdigit() or s[4] not in ("A", "P"):
        return "UNKNOWN"
    hh = int(s[0:2])
    mm = int(s[2:4])
    if hh > 12 or mm > 59:
        return "UNKNOWN"
    hour24 = hh % 12
    if s[4] == "P":
        hour24 += 12
    if 0 <= hour24 < 6:
        return "LATE_NIGHT_0_6"
    elif 6 <= hour24 < 12:
        return "MORNING_6_12"
    elif 12 <= hour24 < 18:
        return "AFTERNOON_12_18"
    else:
        return "EVENING_18_24"


def clean_categorical(value):
    """Normalise a raw cell into a clean category string: missing/blank
    values become 'UNKNOWN', and whole-number floats (e.g. precinct codes
    that pandas read as 5.0) print as '5' instead of '5.0'."""
    if pd.isna(value) or str(value).strip() == "":
        return "UNKNOWN"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def bucket_vehicle_year(year):
    """Vehicle Year is frequently 0/blank/garbage in this dataset — bucket
    into decades, routing anything implausible to 'UNKNOWN'."""
    try:
        y = int(float(year))
    except (TypeError, ValueError):
        return "UNKNOWN"
    if y < 1970 or y > CURRENT_YEAR + 1:
        return "UNKNOWN"
    elif y < 1990:
        return "PRE_1990"
    elif y < 2000:
        return "1990s"
    elif y < 2010:
        return "2000s"
    elif y < 2020:
        return "2010s"
    else:
        return "2020s"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: could not find the file '{CSV_PATH}'.")
        print(f"Current working directory is: {os.getcwd()}")
        print("Edit the CSV_PATH variable near the top of this script so it "
              "points at your downloaded dataset (an absolute path is safest, "
              "e.g. C:/Users/you/Downloads/parking_violations.csv on Windows "
              "or /Users/you/Downloads/parking_violations.csv on Mac).")
        sys.exit(1)

    t0 = time.time()

    # ------------------------------------------------------------------
    # Load the CSV file
    # ------------------------------------------------------------------
    df = pd.read_csv(CSV_PATH)

    # View the first few rows — useful the first time you run this, to
    # confirm the columns loaded the way you expect.
    print(df.head())

    # Keep only the columns this script actually needs (everything else in
    # the raw file is dropped here, to keep the rest of the script simple).
    needed_cols = ["Violation Code", "Violation Time", "Vehicle Year"] + CATEGORICAL_FEATURES
    df = df[needed_cols]

    df["violation_code"] = pd.to_numeric(df["Violation Code"], errors="coerce")
    df = df.dropna(subset=["violation_code"])
    df["violation_code"] = df["violation_code"].astype(int)

    # 2. MAP (sequential, one row at a time): derive the same buckets used
    #    in the Spark version.
    df["violation_hour_bucket"] = df["Violation Time"].apply(parse_hour_bucket)
    df["vehicle_year_bucket"] = df["Vehicle Year"].apply(bucket_vehicle_year)

    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].apply(clean_categorical)

    # 3. Build the target: TOP_N_CLASSES most frequent codes + "OTHER".
    top_codes = df["violation_code"].value_counts().head(TOP_N_CLASSES).index.tolist()
    print(f"Top {TOP_N_CLASSES} violation codes kept as individual classes: {top_codes}")
    df["target_label"] = np.where(
        df["violation_code"].isin(top_codes), df["violation_code"].astype(str), "OTHER"
    )

    n_rows = len(df)
    print(f"Loaded {n_rows} usable rows.")
    if n_rows == 0:
        print("No usable rows — check CSV_PATH and that the column names match.")
        return

    # Sanity-check baseline: accuracy of always predicting the majority class.
    majority_baseline = df["target_label"].value_counts(normalize=True).max()
    print(f"Majority-class baseline accuracy: {majority_baseline:.4f}")

    feature_cols = CATEGORICAL_FEATURES + ["violation_hour_bucket", "vehicle_year_bucket"]
    X = df[feature_cols]
    y = df["target_label"]

    feat_prep_time = time.time() - t0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. One-hot encode the categorical features and fit Multinomial Naive Bayes.
    encoder = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), feature_cols)]
    )

    t1 = time.time()
    X_train_enc = encoder.fit_transform(X_train)
    clf = MultinomialNB()
    clf.fit(X_train_enc, y_train)
    train_time = time.time() - t1

    X_test_enc = encoder.transform(X_test)
    y_pred = clf.predict(X_test_enc)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted")

    total_time = time.time() - t0

    print("\n======== PLAIN PYTHON (NON-BIG-DATA) NAIVE BAYES — PARKING VIOLATIONS RESULTS ========")
    print(f"Rows used                : {n_rows}")
    print(f"Majority-class baseline  : {majority_baseline:.4f}")
    print(f"Feature prep time        : {feat_prep_time:.2f} s")
    print(f"Model training time      : {train_time:.2f} s")
    print(f"Total pipeline time      : {total_time:.2f} s")
    print(f"Test accuracy            : {accuracy:.4f}")
    print(f"Test F1 score            : {f1:.4f}")
    print(f"Test weighted precision  : {precision:.4f}")
    print(f"Test weighted recall     : {recall:.4f}")

    # ------------------------------------------------------------------
    # FINAL ANALYSIS - answering the research question directly: what does
    # the model actually predict, given a vehicle's characteristics and
    # where/when it was ticketed? Same two sections as the Spark version,
    # so the two reports read the same way side by side.
    # ------------------------------------------------------------------
    sample_display = X_test.copy()
    sample_display["target_label"] = y_test.values
    sample_display["predicted_label"] = y_pred

    print("\n================ SAMPLE PREDICTIONS ================")
    print("Vehicle/ticket characteristics alongside the ACTUAL vs PREDICTED violation category:\n")
    display_cols = [
        "Vehicle Body Type",
        "Vehicle Make",
        "Plate Type",
        "Violation County",
        "Violation Precinct",
        "violation_hour_bucket",
        "target_label",
        "predicted_label",
    ]
    print(sample_display[display_cols].head(20).to_string(index=False))

    print("\n================ PER-CLASS ACCURACY (which violation types predict well?) ================")
    results_df = pd.DataFrame({"target_label": y_test.values, "predicted_label": y_pred})
    results_df["correct"] = (results_df["target_label"] == results_df["predicted_label"]).astype(int)
    per_class = (
        results_df.groupby("target_label")
        .agg(class_accuracy=("correct", "mean"), n_test_examples=("correct", "size"))
        .sort_values("n_test_examples", ascending=False)
    )
    print(per_class.to_string())


if __name__ == "__main__":
    main()