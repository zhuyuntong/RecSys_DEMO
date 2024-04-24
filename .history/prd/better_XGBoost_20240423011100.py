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