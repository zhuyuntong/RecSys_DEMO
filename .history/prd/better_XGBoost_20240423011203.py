import sys
import time
from datetime import datetime
import csv
import math
import os
import json
import numpy as np
import pandas as pd
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor
from better_features import (
    FeatureProcessor,
    read_json_data,
    transform_user_data,
    transform_business_data,
    extract_review_data,
)
from KMeans_user_cluster import KMeans_process_user_clusters


def initialize_spark_context(APP_NAME="RegressivePrediction: XGBModel"):
    spark_conf = (
        SparkConf()
        .setAppName(APP_NAME)
        .setAll(
            [
                ("spark.dynamicAllocation.enabled", "true"),
                ("spark.dynamicAllocation.maxExecutors", "10"),
                ("spark.executor.memory", "3g"),
                ("spark.executor.cores", "2"),
                ("spark.executor.memoryOverhead", "3000"),
                ("spark.driver.memory", "4g"),
                ("spark.driver.maxResultSize", "2g"),
                ("spark.python.worker.memory", "2g"),
                ("spark.sql.shuffle.partitions", "20"),
                ("spark.sql.sources.partitionOverWriteMode", "dynamic"),
                ("spark.network.timeout", "600s"),
                ("spark.executor.heartbeatInterval", "120s"),
            ]
        )
    )
    sc = SparkContext(conf=spark_conf)
    sc.setLogLevel("ERROR")
    return sc


def rdd_to_pandas(rdd):
    return pd.DataFrame(rdd.collect(), columns=rdd.first().keys())


sc = initialize_spark_context()


def main(args):
    start_time = time.time()

    if len(sys.argv) != 4:
        print("Usage: spark-submit competition.py data_folder_path test_filepath output_filepath")
        sys.exit(1)

    data_folder_path, test_data_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    # data_folder_path, test_data_file, output_file = '../data/', '../yelp_true.csv', 'prediction.csv'
    print(f"Parsed argv1: {data_folder_path}, argv2: {test_data_file}, argv3: {output_file}\n")

    user_rdd = sc.textFile(data_folder_path + '/user.json').map(transform_user_data)
    user_parsed_df = rdd_to_pandas(user_rdd)
    business_rdd = sc.textFile(data_folder_path + '/business.json').map(transform_business_data)
    business_parsed_df = rdd_to_pandas(business_rdd)
    review_rdd = read_json_data(data_folder_path + '/review_train.json', extract_review_data, sc)
    review_parsed_df = rdd_to_pandas(review_rdd)
    print("[1/4] Data loading completed!\n")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")

    feature_processor = FeatureProcessor(sc, data_folder_path, user_parsed_df, business_parsed_df, review_parsed_df)
    user_clusters = KMeans_process_user_clusters(feature_processor.map_reviews_with_categories(), business_parsed_df)
    # print("FeatureProcessor and Clustering Modules have initialized and processed user, business, review files.")

    print("[2/4] Processor init. and Cluster pre-processing completed!")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")

    return
