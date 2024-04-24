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

    print("------Starting Collecting User-Biz Interaction-Level features for Train and Test Data-------\n")
    # Train data feature processing
    # merge two dataset
    train_data_file1 = f"{data_folder_path}/yelp_train.csv"
    train_data_file2 = f"{data_folder_path}/yelp_val.csv"
    df1 = pd.read_csv(train_data_file1)
    df2 = pd.read_csv(train_data_file2)
    merged_pairs_df = pd.concat([df1, df2], ignore_index=True)
    train_data_file = 'yelp_combined.csv'
    merged_pairs_df.to_csv(train_data_file, index=False)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")

    res_rdd = feature_processor.process_all_features(sc, merged_pairs_df, train_data_file)
    train_df = rdd_to_pandas(res_rdd)
    res_rdd.unpersist()
    train_df = train_df.merge(user_clusters, on='user_id', how='left')
    print("====(1/2)Training data User-Biz Interaction-Level features processed.")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds\n")

    # Test data feature processing
    val_res_rdd = feature_processor.process_all_features(sc, pd.read_csv(test_data_file), test_data_file)
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

    clusters = {'large': [0, 2, 4, 3], 'moderate': [7, 6, 8], 'small': [1, 5]}

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

    # print(best_params)

    models = {}
    rmse_scores = {}

    # train and test, save predict result
    for size in clusters:
        for cluster in clusters[size]:
            # Preparing training and validation data
            train_cluster_data = train_df[train_df['Cluster'] == cluster]
            X_train = train_cluster_data.drop(['stars', 'user_id', 'business_id', 'Cluster'], axis=1)
            y_train = train_cluster_data['stars']
            val_cluster_data = val_df[val_df['Cluster'] == cluster]
            X_val = val_cluster_data.drop(['stars', 'user_id', 'business_id', 'Cluster'], axis=1)
            y_val = val_cluster_data['stars']

            # Retrieving training parameters
            params = best_params[size][cluster]
            if size == 'large':
                # Set early_stopping_rounds during the model initialization
                # model = xgb.XGBRegressor(**params, verbosity=0, early_stopping_rounds=20)
                model = xgb.XGBRegressor(**params, verbosity=0, early_stopping_rounds=50)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model = CatBoostRegressor(**params, verbose=0)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=5, verbose=False)

            # 保存模型和计算预测
            models[cluster] = model
            predictions = model.predict(X_val)
            rmse_scores[cluster] = np.sqrt(mean_squared_error(y_val, predictions))

    # 汇总所有预测结果
    all_predictions = pd.DataFrame()
    for cluster in large_clusters + moderate_clusters + small_clusters:
        val_cluster_data = val_df[val_df['Cluster'] == cluster]
        predictions = models[cluster].predict(val_cluster_data.drop(['stars', 'user_id', 'business_id', 'Cluster'], axis=1))
        result_df = pd.DataFrame(
            {
                'user_id': val_cluster_data['user_id'],
                'business_id': val_cluster_data['business_id'],
                'prediction': predictions,
            }
        )
        all_predictions = pd.concat([all_predictions, result_df], ignore_index=True)
    all_predictions['prediction'] = all_predictions['prediction'].astype(float)

    merged_df = pd.merge(val_df, all_predictions, on=['user_id', 'business_id'])
    rmse = np.sqrt(mean_squared_error(merged_df['stars'], merged_df['prediction']))
    print("Overall RMSE on validation dataset:", rmse)

    all_predictions.to_csv('path_to_save_predictions.csv', index=False)
    print(f"Predictions have been saved to {output_file}.")
    return


if __name__ == "__main__":
    main(sys.argv)
