"""
IST3134 Big Data Analytics - Group Assignment
Spark (Big Data) equivalent of local-NaiveBayes.py — built to run
via spark-submit on Amazon EMR, reading input from S3.

Same research question, same 9 features, same target construction (top 8
violation codes + "OTHER"), same Naive Bayes algorithm as the plain-Python
VS Code script — this is the "Big Data" side of the comparison your report
needs. The two scripts should be run against the SAME dataset so the
timing/accuracy numbers are directly comparable.

Results (accuracy, F1, timings, etc.) print straight to the console/logs —
nothing is written to S3 or anywhere else.

============================== YOUR S3 SETUP ==============================

Input dataset folder : s3://chenwei-asg/Dataset/

This is already set as the default in the CONFIG section below, so you can
just run the script with no arguments once your CSV is in that Dataset
folder. You can still override it from the command line if needed (see
"HOW TO RUN ON EMR").

On EMR specifically, s3:// paths work with Spark out of the box (no extra
JARs or config) — EMR ships with the S3 connector pre-installed.

============================== UPLOADING YOUR DATASET TO S3 ==============================

Two ways to get your CSV into s3://chenwei-asg/Dataset/ before running this:

OPTION A — AWS CLI (simplest, run once from wherever the file currently
sits — your own machine, or the EMR master node after you scp'd it there):

    aws s3 cp parking_violations.csv s3://chenwei-asg/Dataset/

OPTION B — let this script do it for you. Set LOCAL_CSV_TO_UPLOAD below to
the local path of your CSV (e.g. on the EMR master node's disk); the
upload_local_file_to_s3() function will upload it to INPUT_PATH via boto3
before Spark reads anything. Leave LOCAL_CSV_TO_UPLOAD as None if the file
is already sitting in the bucket (boto3 is pre-installed on EMR and
automatically uses the cluster's IAM role — no access keys to configure).

============================== HOW TO RUN ON EMR ==============================

1. Upload this script to S3 too, e.g.:
     s3://chenwei-asg/scripts/spark_parking_naive_bayes_emr.py

2. Make sure your dataset CSV is in s3://chenwei-asg/Dataset/ (see above).

3. Submit the job. Either as an EMR Step from the AWS Console:
     EMR -> your cluster -> Steps -> Add step
       Step type: Spark application
       Application location: s3://chenwei-asg/scripts/spark_parking_naive_bayes_emr.py

   Or via SSH into the cluster's primary (master) node:
     spark-submit s3://chenwei-asg/scripts/spark_parking_naive_bayes_emr.py

   To override the input path from the command line instead of editing the
   CONFIG section:
     spark-submit spark_parking_naive_bayes_emr.py s3://other-bucket/in/

4. Do NOT hardcode --master in this script (it deliberately doesn't call
   .master(...) below) - EMR's spark-submit sets the cluster master (YARN)
   automatically.

============================== TO TEST LOCALLY FIRST (recommended) ==============================

    pip install pyspark boto3
    spark-submit --master local[*] spark_parking_naive_bayes_emr.py /local/path/to/sample.csv
================================================================================================
"""

import os
import sys
import time
from urllib.parse import urlparse

import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, udf
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, IndexToString
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ============================================================================
# CONFIG
# ============================================================================
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "s3://chenwei-asg/Dataset/"

# Set this to a local file path (e.g. "/home/hadoop/parking_violations.csv")
# if your dataset still needs uploading to S3. Leave as None if it's
# already sitting in INPUT_PATH.
LOCAL_CSV_TO_UPLOAD = None

TOP_N_CLASSES = 8
CURRENT_YEAR = 2026

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
# ============================================================================


def upload_local_file_to_s3(local_path, s3_folder_uri):
    """Uploads a local file to an S3 folder (e.g. from the EMR master
    node's local disk) before Spark reads it. Uses boto3, which is
    pre-installed on EMR and automatically picks up the cluster's IAM
    role credentials - no access keys need to be configured.

    Returns the full s3:// URI of the uploaded object.
    """
    parsed = urlparse(s3_folder_uri)
    bucket = parsed.netloc
    folder_key = parsed.path.lstrip("/").rstrip("/")
    object_key = f"{folder_key}/{os.path.basename(local_path)}" if folder_key else os.path.basename(local_path)

    print(f"Uploading {local_path} -> s3://{bucket}/{object_key} ...")
    boto3.client("s3").upload_file(local_path, bucket, object_key)
    print("Upload complete.")
    return f"s3://{bucket}/{object_key}"


def parse_hour_bucket(violation_time):
    """Identical logic to the plain-Python version: Violation Time is
    encoded like '0813A' / '1245P' (HHMM + AM/PM letter) -> 4 time-of-day
    buckets, 'UNKNOWN' for anything blank/malformed."""
    if violation_time is None:
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


def bucket_vehicle_year(year):
    """Identical logic to the plain-Python version: bucket into decades,
    'UNKNOWN' for anything missing or implausible (0, blank, 2099, etc.)."""
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


def clean_categorical(value):
    """Identical logic to the plain-Python version: blank/null -> 'UNKNOWN'."""
    if value is None or value.strip() == "":
        return "UNKNOWN"
    return value.strip()


hour_bucket_udf = udf(parse_hour_bucket, StringType())
year_bucket_udf = udf(bucket_vehicle_year, StringType())
clean_categorical_udf = udf(clean_categorical, StringType())


def main():
    # Starts before the SparkSession itself, so this covers Spark startup
    # too - the true end-to-end runtime of the whole script.
    run_start = time.time()

    # No .master(...) call here on purpose - see the header comment above.
    spark = SparkSession.builder.appName("IST3134-ParkingNaiveBayes-EMR").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    t0 = time.time()

    # ------------------------------------------------------------------
    # (Optional) Upload the dataset to S3 first, then load it.
    # ------------------------------------------------------------------
    input_path = INPUT_PATH
    if LOCAL_CSV_TO_UPLOAD:
        input_path = upload_local_file_to_s3(LOCAL_CSV_TO_UPLOAD, INPUT_PATH)

    # Deliberately NOT using .option("inferSchema", "true") here - on a
    # large dataset that forces Spark to scan the whole file once just to
    # guess column types, then read it a second time to actually load it,
    # doubling the read cost. Every column loads as a string instead, and
    # the one column that actually needs to be numeric (Violation Code) is
    # cast explicitly a few lines down.
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .load(input_path)
    )
    df.show(5)

    needed_cols = ["Violation Code", "Violation Time", "Vehicle Year"] + CATEGORICAL_FEATURES
    df = df.select(*needed_cols)

    df = df.withColumn("violation_code", col("Violation Code").cast(IntegerType()))
    df = df.filter(col("violation_code").isNotNull())

    # MAP: derive the same two engineered buckets as the plain-Python version.
    df = df.withColumn("violation_hour_bucket", hour_bucket_udf(col("Violation Time")))
    df = df.withColumn("vehicle_year_bucket", year_bucket_udf(col("Vehicle Year")))

    # Same cleanup as clean_categorical() in the plain-Python version.
    for c in CATEGORICAL_FEATURES:
        df = df.withColumn(c, clean_categorical_udf(col(c)))

    # Build the target: TOP_N_CLASSES most frequent codes + "OTHER" - same
    # rule as the plain-Python version.
    top_codes = (
        df.groupBy("violation_code")
        .count()
        .orderBy(col("count").desc())
        .limit(TOP_N_CLASSES)
        .select("violation_code")
        .rdd.flatMap(lambda r: r)
        .collect()
    )
    print(f"Top {TOP_N_CLASSES} violation codes kept as individual classes: {top_codes}")

    df = df.withColumn(
        "target_label",
        when(col("violation_code").isin(top_codes), col("violation_code").cast(StringType()))
        .otherwise(lit("OTHER")),
    )

    n_rows = df.count()
    print(f"Loaded {n_rows} usable rows.")
    if n_rows == 0:
        print("No usable rows - check INPUT_PATH and column names.")
        spark.stop()
        return

    # Majority-class baseline, same sanity check as the plain-Python version.
    top_class_count = df.groupBy("target_label").count().agg({"count": "max"}).collect()[0][0]
    majority_baseline = top_class_count / n_rows
    print(f"Majority-class baseline accuracy: {majority_baseline:.4f}")

    feature_cols = CATEGORICAL_FEATURES + ["violation_hour_bucket", "vehicle_year_bucket"]

    # One-hot encode the categorical features. NOTE: this is the one real
    # structural difference from the plain-Python version - sklearn's
    # OneHotEncoder accepts raw strings directly, but Spark's OneHotEncoder
    # only accepts numeric column indices, so each feature needs a
    # StringIndexer first. The end result (a one-hot feature vector per
    # row) is functionally the same.
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in feature_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec") for c in feature_cols
    ]
    assembler = VectorAssembler(
        inputCols=[f"{c}_vec" for c in feature_cols], outputCol="features"
    )
    label_indexer = StringIndexer(inputCol="target_label", outputCol="label", handleInvalid="keep")

    nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")

    pipeline = Pipeline(stages=indexers + encoders + [assembler, label_indexer, nb])

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.cache()
    test_df = test_df.cache()
    train_df.count()
    test_df.count()

    feat_prep_time = time.time() - t0

    t1 = time.time()
    # REDUCE (distributed training): Naive Bayes fitting is a distributed
    # count-and-sum over partitions across the cluster - the same MapReduce
    # shape as a word-count job, just counting (class, feature) pairs
    # instead of words. On EMR this actually spreads across worker nodes.
    model = pipeline.fit(train_df)
    train_time = time.time() - t1

    predictions = model.transform(test_df)

    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    precision_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    recall_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )

    accuracy = acc_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    precision = precision_eval.evaluate(predictions)
    recall = recall_eval.evaluate(predictions)

    total_time = time.time() - t0

    print("\n================ SPARK NAIVE BAYES - PARKING VIOLATIONS RESULTS ================")
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
    # where/when it was ticketed?
    # ------------------------------------------------------------------
    # "prediction" and "label" are numeric indices - map them back to the
    # original violation-code text (e.g. "17", "98", "OTHER") using the
    # label mapping learned by the StringIndexer stage in the pipeline.
    label_indexer_model = model.stages[-2]  # the fitted StringIndexer for target_label
    index_to_label = IndexToString(
        inputCol="prediction", outputCol="predicted_label", labels=label_indexer_model.labels
    )
    predictions_readable = index_to_label.transform(predictions)

    print("\n================ SAMPLE PREDICTIONS ================")
    print("Vehicle/ticket characteristics alongside the ACTUAL vs PREDICTED violation category:\n")
    (
        predictions_readable.select(
            "Vehicle Body Type",
            "Vehicle Make",
            "Plate Type",
            "Violation County",
            "Violation Precinct",
            "violation_hour_bucket",
            "target_label",
            "predicted_label",
        ).show(20, truncate=False)
    )

    print("================ PER-CLASS ACCURACY (which violation types predict well?) ================")
    (
        predictions_readable.withColumn(
            "correct", (col("target_label") == col("predicted_label")).cast("int")
        )
        .groupBy("target_label")
        .agg({"correct": "avg", "target_label": "count"})
        .withColumnRenamed("avg(correct)", "class_accuracy")
        .withColumnRenamed("count(target_label)", "n_test_examples")
        .orderBy(col("n_test_examples").desc())
        .show(20, truncate=False)
    )

    spark.stop()

    total_code_runtime = time.time() - run_start
    print(f"\nTotal code runtime (incl. Spark startup/shutdown): {total_code_runtime:.2f} s")


if __name__ == "__main__":
    main()