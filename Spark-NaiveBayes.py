"""
IST3134 Big Data Analytics - Group Assignment
Spark (Big Data) equivalent of local-NaiveBayes.py — built to run via spark-submit on Amazon EMR, reading input from S3.

Input dataset folder : s3://chenwei-asg/Dataset/
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


INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "s3://chenwei-asg/Dataset/"

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



def upload_local_file_to_s3(local_path, s3_folder_uri):
    
    parsed = urlparse(s3_folder_uri)
    bucket = parsed.netloc
    folder_key = parsed.path.lstrip("/").rstrip("/")
    object_key = f"{folder_key}/{os.path.basename(local_path)}" if folder_key else os.path.basename(local_path)

    print(f"Uploading {local_path} -> s3://{bucket}/{object_key} ...")
    boto3.client("s3").upload_file(local_path, bucket, object_key)
    print("Upload complete.")
    return f"s3://{bucket}/{object_key}"


def parse_hour_bucket(violation_time):
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
   
    if value is None or value.strip() == "":
        return "UNKNOWN"
    return value.strip()


hour_bucket_udf = udf(parse_hour_bucket, StringType())
year_bucket_udf = udf(bucket_vehicle_year, StringType())
clean_categorical_udf = udf(clean_categorical, StringType())


def main():
    run_start = time.time()

   
    spark = SparkSession.builder.appName("IST3134-ParkingNaiveBayes-EMR").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    t0 = time.time()

   
    input_path = INPUT_PATH
    if LOCAL_CSV_TO_UPLOAD:
        input_path = upload_local_file_to_s3(LOCAL_CSV_TO_UPLOAD, INPUT_PATH)

   
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

    
    df = df.withColumn("violation_hour_bucket", hour_bucket_udf(col("Violation Time")))
    df = df.withColumn("vehicle_year_bucket", year_bucket_udf(col("Vehicle Year")))

    
    for c in CATEGORICAL_FEATURES:
        df = df.withColumn(c, clean_categorical_udf(col(c)))

    
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

    
    top_class_count = df.groupBy("target_label").count().agg({"count": "max"}).collect()[0][0]
    majority_baseline = top_class_count / n_rows
    print(f"Majority-class baseline accuracy: {majority_baseline:.4f}")

    feature_cols = CATEGORICAL_FEATURES + ["violation_hour_bucket", "vehicle_year_bucket"]

    # One-hot encode for  categorical features
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

    print("SPARK NAIVE BAYES - PARKING VIOLATIONS RESULTS")
    print(f"Rows used                : {n_rows}")
    print(f"Majority-class baseline  : {majority_baseline:.4f}")
    print(f"Feature prep time        : {feat_prep_time:.2f} s")
    print(f"Model training time      : {train_time:.2f} s")
    print(f"Total pipeline time      : {total_time:.2f} s")
    print(f"Test accuracy            : {accuracy:.4f}")
    print(f"Test F1 score            : {f1:.4f}")
    print(f"Test weighted precision  : {precision:.4f}")
    print(f"Test weighted recall     : {recall:.4f}")

    
    # FINAL ANALYSIS 
    index_to_label = IndexToString(
        inputCol="prediction", outputCol="predicted_label", labels=label_indexer_model.labels
    )
    predictions_readable = index_to_label.transform(predictions)

    print("\n SAMPLE PREDICTIONS")
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

    print("\n PER-CLASS ACCURACY (which violation types predict well?)")
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
