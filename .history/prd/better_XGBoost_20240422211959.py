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


def main(args):
    sc = initialize_spark_context()
    start_time = time.time()

    if len(sys.argv) != 4:
        print("Usage: spark-submit competition.py folder_path test_filepath output_filepath")
        sys.exit(1)

    folder_path, test_data_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    train_data_file = f"{folder_path}/yelp_train.csv"
    print(
        f"Parsed {len(args)} args. argv1: {folder_path}, argv2: {test_data_file}, argv3: {output_file}, \ntrain file loc: {train_data_file}\n"
    )

    user_rdd = sc.textFile(folder_path + '/user.json').map(transform_user_data)
    user_parsed_df = rdd_to_pandas(user_rdd)
    business_rdd = sc.textFile(folder_path + '/business.json').map(transform_business_data)
    business_parsed_df = rdd_to_pandas(business_rdd)
    review_rdd = read_json_data(folder_path + '/review_train.json', extract_review_data, sc)
    review_parsed_df = rdd_to_pandas(review_rdd)
    print("[1/4] Data loading completed!\n")

    feature_processor = FeatureProcessor(sc, folder_path, user_parsed_df, business_parsed_df, review_parsed_df)
    user_clusters = KMeans_process_user_clusters(feature_processor.map_reviews_with_categories(), business_parsed_df)
    # print("FeatureProcessor and Clustering Modules have initialized and processed user, business, review files.")
    print("[2/4] Feature processing completed!")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")

    print("------Starting Collecting User-Biz Interaction-Level features for Train and Test Data-------\n")
    # Train data feature processing
    res_rdd = feature_processor.process_all_features(sc, pd.read_csv(train_data_file), folder_path, train_data_file)
    train_df = rdd_to_pandas(res_rdd)
    res_rdd.unpersist()
    train_df = train_df.merge(user_clusters, on='user_id', how='left')
    print("====(1/2)Training data User-Biz Interaction-Level features processed.")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")

    # Test data feature processing
    val_res_rdd = feature_processor.process_all_features(sc, pd.read_csv(test_data_file), folder_path, test_data_file)
    val_df = rdd_to_pandas(val_res_rdd)
    val_res_rdd.unpersist()
    val_df = val_df.merge(user_clusters, on='user_id', how='left')
    print("====(2/2)Feature extraction for train and test data completed.\n")
    print("[3/4]Training data User-Biz Interaction-Level features processed.")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")

    ## TRAINING PROCESS START
    print("------Starting Splitting Clusters and Training Models on 9 K-Means Clusters-------\n")
    large_clusters = [0, 2, 4, 3]
    moderate_clusters = [7, 6, 8]
    small_clusters = [5, 1]

    model_params = {
        'large': {
            0: {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8},
            2: {'learning_rate': 0.1, 'n_estimators': 120, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8},
            4: {'learning_rate': 0.1, 'n_estimators': 110, 'max_depth': 6, 'subsample': 0.7, 'colsample_bytree': 0.7},
            3: {'learning_rate': 0.09, 'n_estimators': 100, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8},
        },
        'moderate': {
            7: {'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3, 'iterations': 500},
            6: {'learning_rate': 0.05, 'depth': 5, 'l2_leaf_reg': 3, 'iterations': 450},
            8: {'learning_rate': 0.06, 'depth': 6, 'l2_leaf_reg': 2, 'iterations': 500},
        },
        'small': {
            5: {'learning_rate': 0.1, 'depth': 4, 'l2_leaf_reg': 1, 'iterations': 200},
            1: {'learning_rate': 0.1, 'depth': 3, 'l2_leaf_reg': 1, 'iterations': 150},
        },
    }

    models = {}

    # Training models for large clusters with XGBoost
    for cluster in large_clusters:
        cluster_data = train_df[train_df['cluster'] == cluster]
        features = cluster_data.drop(['stars', 'user_id', 'business_id', 'cluster'], axis=1)
        labels = cluster_data['stars']
        params = model_params['large'][cluster]
        model = xgb.XGBRegressor(**params)
        model.fit(features, labels)
        models[cluster] = model
        print(f"Trained XGBoost model for large cluster {cluster} with params {params}")

    # Training models for moderate clusters with CatBoost
    for cluster in moderate_clusters:
        cluster_data = train_df[train_df['cluster'] == cluster]
        features = cluster_data.drop(['stars', 'user_id', 'business_id', 'cluster'], axis=1)
        labels = cluster_data['stars']
        params = model_params['moderate'][cluster]
        model = CatBoostRegressor(**params, verbose=0)
        model.fit(features, labels)
        models[cluster] = model
        print(f"Trained CatBoost model for moderate cluster {cluster} with params {params}")

    # Training models for small clusters with CatBoost
    for cluster in small_clusters:
        cluster_data = train_df[train_df['cluster'] == cluster]
        features = cluster_data.drop(['stars', 'user_id', 'business_id', 'cluster'], axis=1)
        labels = cluster_data['stars']
        params = model_params['small'][cluster]
        model = CatBoostRegressor(**params, verbose=0)
        model.fit(features, labels)
        models[cluster] = model
        print(f"Trained CatBoost model for small cluster {cluster} with params {params}")

    ## TRAINING PROCESS END

    ## PREDICT RATINGS AND SAVE TO FILE
    print(
        f"------Starting Predicting on {test_data_file} and Merging Results from each Model on 9 K-Means Clusters-------\n"
    )
    # save all Prediction results from each cluster
    for cluster, model in models.items():
        # Predicting the validation/test set
        val_features = val_df[val_df['cluster'] == cluster].drop(['user_id', 'business_id', 'cluster'], axis=1)
        predictions = model.predict(val_features)
        val_df.loc[val_df['cluster'] == cluster, 'prediction'] = predictions

        print(f"Cluster {cluster} - Prediction completed. Duration: {time.time() - start_time:.4f} seconds")

    # Output to file
    val_df.to_csv(output_file, columns=['user_id', 'business_id', 'prediction'], index=False)
    print(f"Predictions have been saved to {output_file}.")


if __name__ == "__main__":
    main(sys.argv)
    # features = final_df.drop(['stars', 'user_id', 'business_id'], axis=1)  # 或其他目标变量列
    # labels = final_df['stars']
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    ### LOCAL DEBUG PURPOSE
    # user_parsed_df = pd.read_csv('cache/user_df.csv')
    # business_parsed_df = pd.read_csv('cache/business_df.csv')
    # review_parsed_df = pd.read_csv('cache/review_df.csv')
    ###
