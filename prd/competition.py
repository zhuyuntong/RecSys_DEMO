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
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV


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
    # 0,2 no splitting val,
    # 3,4 using only catboost
    # test_size = 0.05
    large_xgb_params = {
        0: {'learning_rate': 0.04246778091879101, 'max_depth': 6, 'n_estimators': 491, 'subsample': 0.8689},
        2: {'learning_rate': 0.12350751460907078, 'max_depth': 3, 'n_estimators': 288, 'subsample': 0.8880},
        3: {'learning_rate': 0.08288064461501718, 'max_depth': 5, 'n_estimators': 305, 'subsample': 0.6839},
        4: {'learning_rate': 0.18299396047960448, 'max_depth': 4, 'n_estimators': 250, 'subsample': 0.9081},
    }

    large_catboost_params = {
        0: {'depth': 12, 'l2_leaf_reg': 18.15, 'learning_rate': 0.06229, 'n_estimators': 1486},
        2: {'depth': 6, 'l2_leaf_reg': 28.02, 'learning_rate': 0.1090, 'n_estimators': 176},
        3: {'depth': 5, 'l2_leaf_reg': 0.2358, 'learning_rate': 0.1968, 'n_estimators': 62},
        4: {'depth': 4, 'l2_leaf_reg': 0.3376, 'learning_rate': 0.1914, 'n_estimators': 46},
    }

    # 5, 6, 7, 8
    # test_size = 0.1
    medium_catboost_params = {
        5: {'depth': 3, 'l2_leaf_reg': 0.415375550656062, 'learning_rate': 0.9017553809036529},
        7: {'depth': 6, 'l2_leaf_reg': 18.3792029910461, 'learning_rate': 0.9372003134293689},
        8: {'depth': 1, 'l2_leaf_reg': 7.440376345058953, 'learning_rate': 0.4049807340854126},
        6: {'depth': 2, 'l2_leaf_reg': 1.1835517768726047, 'learning_rate': 0.38770112704134657},
    }
    # rmse_scores = {}
    print("--------train and test, save predict result-----------")

    def find_best_weight(cluster, X_train, y_train, X_test, y_test):
        # init
        cb_model = CatBoostRegressor(**large_catboost_params[cluster], verbose=False)
        xgb_model = XGBRegressor(**large_xgb_params[cluster], objective='reg:squarederror', verbosity=0)

        # train
        cb_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        # predict
        cb_preds = cb_model.predict(X_test).clip(1, 5)
        xgb_preds = xgb_model.predict(X_test).clip(1, 5)

        best_rmse = float('inf')
        best_ratio = 0

        for ratio in np.linspace(0, 1, 21):  # test different porpotion
            final_preds = ratio * cb_preds + (1 - ratio) * xgb_preds
            rmse = np.sqrt(mean_squared_error(y_test, final_preds))
            if rmse < best_rmse:
                best_rmse = rmse
                best_ratio = ratio

        return best_ratio, best_rmse

    def find_best_weight(cluster, X_train, y_train, X_test, y_test):
        # init
        cb_model = CatBoostRegressor(**large_catboost_params[cluster], verbose=False)
        xgb_model = XGBRegressor(**large_xgb_params[cluster], objective='reg:squarederror', verbosity=0)

        # train
        cb_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        # predict
        cb_preds = cb_model.predict(X_test).clip(1, 5)
        xgb_preds = xgb_model.predict(X_test).clip(1, 5)

        best_rmse = float('inf')
        best_ratio = 0

        for ratio in np.linspace(0, 1, 21):  # test different porpotion
            final_preds = ratio * cb_preds + (1 - ratio) * xgb_preds
            rmse = np.sqrt(mean_squared_error(y_test, final_preds))
            if rmse < best_rmse:
                best_rmse = rmse
                best_ratio = ratio

        return best_ratio, best_rmse

    def train_model(cluster, train_df, test_df):
        def prepare_data(df, cluster, drop_cols):
            if cluster == -1:
                cluster_data = df
            else:
                cluster_data = df[df['Cluster'] == cluster]

            drop_cols = [col for col in drop_cols if col in cluster_data.columns]
            X = cluster_data.drop(columns=drop_cols, errors='ignore')
            y = cluster_data['stars']
            return X, y

        if cluster in [5]:
            drop_cols = ['stars', 'user_id', 'business_id', 'log_affinity_score']  # For cluster 5, do not drop 'score'
        elif cluster in [6, 7, 8]:
            drop_cols = [
                'stars',
                'user_id',
                'business_id',
                'binary_affinity_score',
                'log_affinity_score',
            ]  # Drop both 'score' and 'log_affinity_score'
        else:
            drop_cols = ['stars', 'user_id', 'business_id']  # Default columns to drop

        if cluster in [0, 2]:
            X_train, y_train = prepare_data(train_df, cluster, drop_cols)
            X_test, y_test = prepare_data(test_df, cluster, drop_cols)

            cb_model = CatBoostRegressor(**large_catboost_params[cluster], verbose=False)
            xgb_model = XGBRegressor(**large_xgb_params[cluster], objective='reg:squarederror', verbosity=0)

            cb_model.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)

            cb_preds = cb_model.predict(X_test)
            xgb_preds = xgb_model.predict(X_test)

            # Applying best weight ratio and clipping
            #  if cluster == 0 else (0.65 * cb_preds + 0.35 * xgb_preds)
            best_ratio, best_rmse = find_best_weight(cluster, X_train, y_train, X_test, y_test)
            final_preds = best_ratio * cb_preds + (1 - best_ratio) * xgb_preds
            final_preds = np.clip(final_preds, 1, 5)

        elif cluster in [1]:
            if pd.isnull(train_df['score']).any() or pd.isnull(test_df['score']).any():
                drop_cols.append('score')

            X_train, y_train = prepare_data(train_df, cluster, drop_cols)
            X_test, y_test = prepare_data(test_df, cluster, drop_cols)

            # model = ElasticNet(**small_ES_params[cluster])
            model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1], alphas=np.logspace(-6, 2, 100), cv=5, random_state=42
            )
            model.fit(X_train, y_train)
            final_preds = model.predict(X_test).clip(1, 5)

        elif cluster in [3]:
            test_size = 0.25
            X_train, y_train = prepare_data(train_df, cluster, drop_cols)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
            X_test, y_test = prepare_data(test_df, cluster, drop_cols)
            model = CatBoostRegressor(**large_catboost_params[cluster], verbose=False)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
            final_preds = model.predict(X_test).clip(1, 5)

        elif cluster in [4, 5, 6, 7, 8]:
            test_size = 0.1
            X_train, y_train = prepare_data(test_df, cluster, drop_cols)
            X_test, y_test = prepare_data(test_df, cluster, drop_cols)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

            model = CatBoostRegressor(
                **(large_catboost_params[cluster] if cluster in [3, 4] else medium_catboost_params[cluster]), verbose=False
            )
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10)
            final_preds = model.predict(X_test).clip(1, 5)

        # merge a prediction df
        id_and_stars = test_df[test_df['Cluster'] == cluster][['user_id', 'business_id', 'stars']]
        predictions_df = pd.DataFrame(final_preds, index=id_and_stars.index, columns=['predicted_stars'])
        results_df = pd.concat([id_and_stars, predictions_df], axis=1)

        rmse = np.sqrt(mean_squared_error(y_test, final_preds))
        # rmse = np.sqrt(mean_squared_error(results_df['stars'], results_df['predicted_stars']))
        return results_df, rmse

    clusters_rmse = {}
    all_results = pd.DataFrame()

    for cluster in range(9):
        # for cluster in [0, 2, 1, 3, 4, 5, 6, 7, 8]:
        result_df, rmse = train_model(cluster, train_df, val_df)
        clusters_rmse[cluster] = rmse
        all_results = pd.concat([all_results, result_df])

    print("RMSE from each cluster:", clusters_rmse)

    # final sanity check
    all_results['prediction'] = all_results['prediction'].astype(float)
    all_results['prediction'] = all_results['prediction'].clip(lower=1, upper=5)
    all_results.to_csv(output_file, columns=['user_id', 'business_id', 'prediction'], index=False)
    print(f"Predictions have been saved to {output_file}.")
    return


if __name__ == "__main__":
    main(sys.argv)
