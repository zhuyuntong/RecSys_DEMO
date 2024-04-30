import os
import shutil
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
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor
from better_features import (
    FeatureProcessor,
    transform_user_data,
    transform_business_data,
    extract_review_data,
    read_json_data,
)
from KMeans_user_cluster import KMeans_process_user_clusters


def initialize_spark_context():
    spark_conf = (
        SparkConf()
        .setAppName("RegressivePrediction: XGBModel")
        .set("spark.dynamicAllocation.enabled", "true")
        .set("spark.dynamicAllocation.maxExecutors", "10")
        .set("spark.executor.memory", "8g")
        .set("spark.executor.cores", "2")
        .set("spark.executor.memoryOverhead", "1g")
        .set("spark.driver.memory", "8g")
        .set("spark.driver.maxResultSize", "2g")
        .set("spark.python.worker.memory", "2g")
        .set("spark.sql.shuffle.partitions", "200")
        .set("spark.sql.sources.partitionOverWriteMode", "dynamic")
        .set("spark.network.timeout", "800s")
        .set("spark.executor.heartbeatInterval", "120s")
    )
    sc = SparkContext(conf=spark_conf)
    sc.setLogLevel("ERROR")

    return sc


sc = initialize_spark_context()


def rdd_to_pandas(rdd):
    return pd.DataFrame(rdd.collect(), columns=rdd.first().keys())


# simplicity purpose
def process_json_files(data_folder_path, sc):
    user_rdd = sc.textFile(data_folder_path + '/user.json').map(transform_user_data)
    user_parsed_df = rdd_to_pandas(user_rdd)
    user_rdd.unpersist()

    business_rdd = sc.textFile(data_folder_path + '/business.json').map(transform_business_data)
    business_parsed_df = rdd_to_pandas(business_rdd)
    business_rdd.unpersist()

    review_rdd = read_json_data(data_folder_path + '/review_train.json', extract_review_data, sc)
    review_parsed_df = rdd_to_pandas(review_rdd)
    review_rdd.unpersist()

    return user_parsed_df, business_parsed_df, review_parsed_df


# simplicity purpose
def process_features(data_file_path, feature_processor, user_clusters):
    df = feature_processor.process_all_features(sc, pd.read_csv(data_file_path), data_file_path)
    df = df.merge(user_clusters, on='user_id', how='left')
    return df


def main(args):
    start_time = time.time()

    if len(sys.argv) != 4:
        print("Usage: spark-submit competition.py data_folder_path test_filepath output_filepath")
        sys.exit(1)

    data_folder_path, test_data_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"Parsed argv1: {data_folder_path}, argv2: {test_data_file}, argv3: {output_file}\n")

    # FIXME
    # user_parsed_df, business_parsed_df, review_parsed_df = process_json_files(data_folder_path, sc)
    # FIXME

    # FIXME Only for local DEBUG Purpose
    user_parsed_df = pd.read_csv('cache/user_df.csv')  # parsed from users.json
    business_parsed_df = pd.read_csv('cache/business_df.csv')  # parsed from business.json
    review_parsed_df = pd.read_csv('cache/review_df.csv')  # parsed from business.json
    # FIXME

    print("[1/4] Data loading completed!\n")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")

    feature_processor = FeatureProcessor(sc, data_folder_path, user_parsed_df, business_parsed_df, review_parsed_df)
    user_clusters = KMeans_process_user_clusters(feature_processor.map_reviews_with_categories(), business_parsed_df)

    print("[2/4] Processor init. and Cluster pre-processing completed!")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")

    ##############################################################################################################################
    print("------[3/4] Starting Collecting User-Biz Interaction-Level features for Train and Test Data-------\n")

    '''# Train data feature processing'''
    train_data_file = f"{data_folder_path}/yelp_train.csv"
    print("train_file: ", train_data_file)
    train_df = process_features(train_data_file, feature_processor, user_clusters)

    print("==== Training data User-Biz Interaction-Level features processed.")
    ################################################################################################
    '''# Test data feature processing'''
    print("test_file: ", test_data_file)
    val_df = process_features(test_data_file, feature_processor, user_clusters)

    print("==== Testing data User-Biz Interaction-Level features processed.")

    print("[3/4] All Data User-Biz Interaction-Level features processed.")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")

    ################################################################################################################################
    '''# MODEL TRAINING and VALIDATION PROCESS START'''
    print("------ [4/4] Starting Splitting Clusters and Training Models on 9 K-Means Clusters-------\n")
    large_clusters = [0, 2, 4, 3]
    moderate_clusters = [7, 6, 8]
    small_clusters = [5, 1]
    clusters = {'large': [0, 2, 4, 3], 'moderate': [7, 6, 8], 'small': [1, 5]}
    # clusters = {'small': [1, 5]}
    best_params = {
        'large': {
            0: {
                'colsample_bytree': 0.58,
                'gamma': 0.34,
                'learning_rate': 0.033,
                'max_depth': 9,
                'max_leaves': 119,
                'n_estimators': 482,
                'subsample': 0.9,
            },
            2: {
                'colsample_bytree': 0.58,
                'gamma': 0.4,
                'learning_rate': 0.029,
                'max_depth': 7,
                'max_leaves': 126,
                'n_estimators': 450,
                'subsample': 0.918,
            },
            4: {
                'colsample_bytree': 0.58,
                'gamma': 0.4,
                'learning_rate': 0.029,
                'max_depth': 7,
                'max_leaves': 118,
                'n_estimators': 450,
                'subsample': 0.93,
            },
            3: {
                'colsample_bytree': 0.61,
                'gamma': 0.4,
                'learning_rate': 0.029,
                'max_depth': 7,
                'max_leaves': 122,
                'n_estimators': 450,
                'subsample': 0.93,
            },
        },
        'moderate': {
            7: {'depth': 10, 'l2_leaf_reg': 1, 'learning_rate': 0.2, 'n_estimators': 634},
            6: {'depth': 9, 'l2_leaf_reg': 7, 'learning_rate': 0.187, 'n_estimators': 631},
            8: {'depth': 6, 'l2_leaf_reg': 8, 'learning_rate': 0.178, 'n_estimators': 638},
        },
        'small': {
            1: {'depth': 9, 'l2_leaf_reg': 1, 'learning_rate': 0.165, 'n_estimators': 705},
            5: {'depth': 10, 'l2_leaf_reg': 1, 'learning_rate': 0.2, 'n_estimators': 549},
        },
    }

    # rmse_scores = {}
    print("--------train and test, save predict result-----------")

    def prepare_data_for_prediction(df, columns_to_drop, keep_columns=['user_id', 'business_id']):
        if any(col not in df.columns for col in keep_columns):
            print(f"Missing one of the required columns: {keep_columns} in the DataFrame.")
            return None, None

        kept_data = df[keep_columns] if keep_columns else df
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        return df, kept_data

    def remove_constant_features(df, exclude_columns=['user_id', 'business_id']):
        # Identify constant features excluding specified columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1 and col not in exclude_columns]
        # Drop constant columns
        return df.drop(columns=constant_cols), constant_cols

    columns_to_drop = ['stars', 'user_id', 'business_id', 'Cluster']
    all_predictions = pd.DataFrame()
    models = {}
    removed_features = {}

    for size in clusters:
        for cluster in clusters[size]:
            # Preparing training data
            train_cluster_data = train_df[train_df['Cluster'] == cluster]
            train_cluster_data, removed_cols_train = remove_constant_features(
                train_cluster_data, exclude_columns=['user_id', 'business_id']
            )
            # removed_features[cluster] = removed_cols_train

            if train_cluster_data.empty:
                print(f"Skipping training for cluster {cluster} as it has no training data.")
                continue

            # # train data prep
            X_train, _ = prepare_data_for_prediction(train_cluster_data, columns_to_drop)
            y_train = train_cluster_data['stars']

            if y_train.empty:
                print(f"Skipping cluster {cluster} due to empty labels in training set.")
                continue

            # Train models
            params = best_params[size][cluster]
            model = xgb.XGBRegressor(**params, verbosity=0) if size == 'large' else CatBoostRegressor(**params, verbose=0)
            model.fit(X_train, y_train, verbose=False)
            models[cluster] = model

            ############################################################################
            # Preparing validation data
            val_cluster_data = val_df[val_df['Cluster'] == cluster]
            print("val cluster data: ", val_cluster_data.head(5))

            val_cluster_data, _ = remove_constant_features(val_cluster_data)

            if val_cluster_data.empty:
                print(f"Skipping validation for cluster {cluster} as it has no validation data.")
                continue

            X_val, kept_data = prepare_data_for_prediction(val_cluster_data, columns_to_drop)

            if set(X_train.columns) != set(X_val.columns):
                print(f"Feature mismatch in cluster {cluster}, skipping predictions.")
                continue

            # Retrieving training parameters
            predictions = models[cluster].predict(X_val)
            result_df = pd.DataFrame(
                {
                    'user_id': kept_data['user_id'],
                    'business_id': kept_data['business_id'],
                    'prediction': predictions,
                }
            )

            all_predictions = pd.concat([all_predictions, result_df], ignore_index=True)

    # final sanity check
    all_predictions['prediction'] = all_predictions['prediction'].astype(float)
    all_predictions['prediction'] = all_predictions['prediction'].clip(lower=1, upper=5)
    all_predictions.to_csv(output_file, columns=['user_id', 'business_id', 'prediction'], index=False)
    print(f"Predictions have been saved to {output_file}.")
    return


if __name__ == "__main__":
    main(sys.argv)
