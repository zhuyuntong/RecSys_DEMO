import json
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS as mllib_ALS, MatrixFactorizationModel, Rating
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

''' 
PREREQ 

# category: category md5 mapping
> category_md5_df = create_category_md5_mapping(feature_processor.business_df)

# historical reviews 
> final_mapped_review_df = integrate_mapping_user_bus_cat_data(feature_processor.df_conn, category_md5_df, review_data)
'''


# DEBUG PURPOSE
def test(sc, data_folder_path, pair_file_path='../yelp_combined.csv', rank=10, reg_param=0.5, alpha=1520.0):
    # Helper function to parse JSON data
    def parse_json(line):
        try:
            data = json.loads(line)
            return ((data['user_id'], data['business_id']), 1)
        except json.JSONDecodeError:
            return None

    # Load data
    rdd = sc.textFile(data_folder_path + '/tip.json')
    interaction_counts = rdd.map(parse_json).filter(lambda x: x is not None)

    interaction_counts.persist()  # Caching this RDD as it is used to create multiple RDDs

    # 对所有交互进行计数, 后续使用；
    user_business_interaction_counts = interaction_counts.reduceByKey(lambda a, b: a + b)

    # Map data to numeric indices
    user_ids = interaction_counts.map(lambda x: x[0][0]).distinct().zipWithUniqueId().collectAsMap()
    business_ids = interaction_counts.map(lambda x: x[0][1]).distinct().zipWithUniqueId().collectAsMap()

    user_ids_bcast = sc.broadcast(user_ids)
    business_ids_bcast = sc.broadcast(business_ids)

    # # Convert interactions to rating format
    ratings_rdd = user_business_interaction_counts.map(
        lambda p: Rating(
            int(user_ids_bcast.value[p[0][0]]),
            int(business_ids_bcast.value[p[0][1]]),
            float(np.float16(log_transform_interaction(p[1]))),
        )  # FIXME
    )
    return ratings_rdd


def decompose_matrix_factorization_als(sc, data_folder_path, rank=51, reg_param=30, alpha=1530.0):  # FIXME:
    # Helper function to parse JSON data
    def parse_json(line):
        try:
            data = json.loads(line)
            return ((data['user_id'], data['business_id']), 1)
        except json.JSONDecodeError:
            return None

    # Load data
    rdd = sc.textFile(data_folder_path + '/tip.json')
    interaction_counts = rdd.map(parse_json).filter(lambda x: x is not None)

    interaction_counts.persist()  # Caching this RDD as it is used to create multiple RDDs

    # 对所有交互进行计数, 后续使用；
    user_business_interaction_counts = interaction_counts.reduceByKey(lambda a, b: a + b)

    # Map data to numeric indices
    user_ids = interaction_counts.map(lambda x: x[0][0]).distinct().zipWithUniqueId().collectAsMap()
    business_ids = interaction_counts.map(lambda x: x[0][1]).distinct().zipWithUniqueId().collectAsMap()

    user_ids_bcast = sc.broadcast(user_ids)
    business_ids_bcast = sc.broadcast(business_ids)

    # # Convert interactions to rating format
    ratings_rdd = user_business_interaction_counts.map(
        lambda p: Rating(
            int(user_ids_bcast.value[p[0][0]]),
            int(business_ids_bcast.value[p[0][1]]),
            float(np.float16(log_transform_interaction(p[1]))),
        )  # FIXME
        # lambda p: Rating(
        #     int(user_ids_bcast.value[p[0][0]]), int(business_ids_bcast.value[p[0][1]]), float(p[1])
        # )  # debugging to see if this works better # FIXME:
    )
    ratings_rdd.cache()  # use later in ALS algo

    spark = SparkSession.builder.appName("ALS Example").getOrCreate()
    schema = StructType(
        [
            StructField("user", IntegerType(), True),
            StructField("product", IntegerType(), True),  # Changed from 'business' to 'product'
            StructField("rating", FloatType(), True),
        ]
    )
    # NOTE: temporary conver to sparkDF for input START HERE
    ratings_df = spark.createDataFrame(ratings_rdd, schema)
    ratings_df = ratings_df.withColumnRenamed("product", "business")
    # Train ALS model
    als = ALS(
        rank=rank,
        regParam=reg_param,
        alpha=alpha,
        userCol="user",
        itemCol="business",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=True,
    )

    # model = ALS.trainImplicit(ratings_rdd, rank, alpha=alpha, lambda_=reg_param) # buggy!
    model = als.fit(ratings_df)

    # Collect user and product features
    # user_features_rdd = model.userFeatures().cache()  # use later in affinity calc. # repartition(1000)
    # biz_features_rdd = model.productFeatures().cache()

    user_factors_df = model.userFactors
    item_factors_df = model.itemFactors
    # NOTE: temporary usage END HERE
    user_features_rdd = user_factors_df.rdd.map(lambda row: (row['id'], row['features']))
    biz_features_rdd = item_factors_df.rdd.map(lambda row: (row['id'], row['features']))

    user_features_rdd = user_features_rdd.cache()
    biz_features_rdd = biz_features_rdd.cache()

    user_features_bcast = sc.broadcast(user_features_rdd.collectAsMap())
    biz_features_bcast = sc.broadcast(biz_features_rdd.collectAsMap())

    return (
        user_features_bcast,
        biz_features_bcast,
        user_ids,
        business_ids,
        user_features_rdd,
        biz_features_rdd,
        user_ids_bcast,
        business_ids_bcast,
    )


def predict_aff_score_als(
    sc,
    user_features_bcast,
    biz_features_bcast,
    user_ids,
    business_ids,
    user_features_rdd,
    biz_features_rdd,
    user_ids_bcast,
    business_ids_bcast,
    pair_file_path='../yelp_combined.csv',
):
    ##################################################################
    pair_rdd = (
        sc.textFile(pair_file_path)
        .filter(lambda x: not x.startswith("user_id"))
        .map(lambda line: (line.split(',')[0].strip(), line.split(',')[1].strip()))
    )
    pair_rdd.cache()
    #######################################################################

    ##### PRD: better deal with cold start issue and map num=>string
    reverse_user_ids = {v: k for k, v in user_ids.items()}
    reverse_business_ids = {v: k for k, v in business_ids.items()}

    reverse_user_ids_bcast = sc.broadcast(reverse_user_ids)
    reverse_business_ids_bcast = sc.broadcast(reverse_business_ids)

    # 过滤掉有-1的pairs进行计算
    tracked_pairs_rdd = pair_rdd.map(lambda line: map_and_track_ids(line, user_ids_bcast, business_ids_bcast))
    # user_ids_bcast.unpersist()
    # business_ids_bcast.unpersist()

    valid_pairs_rdd = tracked_pairs_rdd.filter(lambda x: x[0][0] != -1 and x[0][1] != -1).cache()
    missing_pairs_rdd = tracked_pairs_rdd.filter(lambda x: x[0][0] == -1 or x[0][1] == -1).map(
        lambda x: (x[1], 1.0)
    )  # FIXME:  # tune this

    tracked_pairs_rdd.cache()
    valid_pairs_rdd.cache()
    missing_pairs_rdd.cache()

    # 计算有效num_id pairs的scores
    valid_num_id_pairs = valid_pairs_rdd.map(lambda x: x[0])

    batch_size = 1000
    valid_batches = valid_num_id_pairs.mapPartitions(lambda x: [list(x)]).flatMap(
        lambda x: [x[i : i + batch_size] for i in range(0, len(x), batch_size)]
    )
    valid_scores_rdd = valid_batches.flatMap(
        lambda batch: calculate_affinity_for_batch(batch, user_features_bcast, biz_features_bcast)
    )
    # map back to id
    mapped_scores_rdd = valid_scores_rdd.map(lambda x: map_ids_back(x, reverse_user_ids_bcast, reverse_business_ids_bcast))

    # 合并计算结果和无效pairs
    mapped_missing_scores_rdd = missing_pairs_rdd.map(lambda x: (x[0], x[1]))
    final_scores_rdd = mapped_scores_rdd.union(mapped_missing_scores_rdd)
    # return pair_rdd, user_ids_bcast, business_ids_bcast # debug
    # return mapped_scores_rdd, mapped_missing_scores_rdd

    # 进一步处理得到yelp_pairs.csv中的用户和商家向量
    final_user_features_rdd, final_biz_features_rdd = process_vector_features(
        sc,
        valid_pairs_rdd,
        user_features_rdd,
        biz_features_rdd,
        missing_pairs_rdd,
        reverse_user_ids_bcast,
        reverse_business_ids_bcast,
    )

    reverse_user_ids_bcast.unpersist()
    reverse_business_ids_bcast.unpersist()

    final_scores_rdd.cache()
    final_user_features_rdd.cache()
    final_biz_features_rdd.cache()

    return final_scores_rdd, final_user_features_rdd, final_biz_features_rdd


'''
Usage:
def matrix_factorization_als(sc, data_folder_path, pair_file_path='../yelp_val.csv', rank=10, reg_param=0.5, alpha=1520.0)

final_scores_rdd, final_user_features_rdd, final_biz_features_rdd = matrix_factorization_als(sc, data_path, '../yelp_val.csv')
'''


def process_vector_features(
    sc,
    valid_pairs_rdd,
    user_features_rdd,
    biz_features_rdd,
    missing_pairs_rdd,
    reverse_user_ids_bcast,
    reverse_business_ids_bcast,
):
    # Extract and broadcast IDs
    user_ids_bcast, business_ids_bcast = extract_and_broadcast_ids(sc, valid_pairs_rdd)

    # Filter features RDDs
    valid_user_features_rdd = filter_features_by_ids(user_features_rdd, user_ids_bcast)
    valid_biz_features_rdd = filter_features_by_ids(biz_features_rdd, business_ids_bcast)

    # Define default vector and broadcast
    default_vector = [0.0] * len(user_features_rdd.first()[1])
    default_vector_bcast = sc.broadcast(default_vector)

    # Handle missing features
    missing_user_features_rdd = create_missing_features_rdd(missing_pairs_rdd, 0, default_vector_bcast)
    missing_biz_features_rdd = create_missing_features_rdd(missing_pairs_rdd, 1, default_vector_bcast)

    # Map numerical IDs to string IDs
    valid_user_features_with_str_ids_rdd = valid_user_features_rdd.map(
        lambda x: map_num_to_string_id(x, reverse_user_ids_bcast)
    )
    valid_biz_features_with_str_ids_rdd = valid_biz_features_rdd.map(
        lambda x: map_num_to_string_id(x, reverse_business_ids_bcast)
    )

    # Combine into final RDDs
    final_user_features_rdd = combine_features(valid_user_features_with_str_ids_rdd, missing_user_features_rdd)
    final_biz_features_rdd = combine_features(valid_biz_features_with_str_ids_rdd, missing_biz_features_rdd)

    # len(default_vector_bcast.value)
    default_vector_bcast.unpersist()
    user_ids_bcast.unpersist()
    business_ids_bcast.unpersist()
    return final_user_features_rdd, final_biz_features_rdd


def extract_and_broadcast_ids(sc, rdd):
    user_ids = rdd.map(lambda x: x[0][0]).distinct().collect()
    business_ids = rdd.map(lambda x: x[0][1]).distinct().collect()
    return sc.broadcast(user_ids), sc.broadcast(business_ids)


def filter_features_by_ids(features_rdd, ids_bcast):
    return features_rdd.filter(lambda x: x[0] in ids_bcast.value)


def create_missing_features_rdd(pairs_rdd, id_index, default_vector_bcast):
    return pairs_rdd.map(lambda x: (x[0][id_index], default_vector_bcast.value))


def map_num_to_string_id(features_tuple, reverse_id_bcast):
    """
    Maps a tuple of (numerical_id, features) to (string_id, features) using a broadcasted reverse ID map.
    """
    num_id, features = features_tuple
    str_id = reverse_id_bcast.value.get(num_id, "Unknown")  # Use "Unknown" if no mapping exists
    return (str_id, features)


def combine_features(valid_features_rdd, missing_features_rdd):
    return valid_features_rdd.union(missing_features_rdd)


#######################
def map_and_track_ids(line, user_ids_bcast, business_ids_bcast):
    user_id, biz_id = line
    user_num = user_ids_bcast.value.get(user_id.strip(), -1)
    biz_num = business_ids_bcast.value.get(biz_id.strip(), -1)
    return (user_num, biz_num), (user_id, biz_id)


def map_ids_back(pair_score, reverse_user_ids_bcast, reverse_business_ids_bcast):
    (user_num, biz_num), score = pair_score
    user_id = reverse_user_ids_bcast.value.get(user_num, "Unknown")
    biz_id = reverse_business_ids_bcast.value.get(biz_num, "Unknown")
    return ((user_id, biz_id), score)


#######################
def calculate_affinity_for_batch(batch, user_features_bcast, biz_features_bcast, default_score=1.0):  # tune this # FIXME
    """计算一批用户-商户对的亲和力得分,batch中包含的是数值ID对,对无效映射使用默认得分(不应该存在)"""
    results = []
    for user_num, biz_num in batch:
        if user_num == -1 or biz_num == -1:
            # 对于无效的数值ID，使用默认得分
            results.append(((user_num, biz_num), default_score))
        else:
            user_features = user_features_bcast.value.get(user_num)
            biz_features = biz_features_bcast.value.get(biz_num)
            if user_features is not None and biz_features is not None:
                score = np.dot(user_features, biz_features)
                results.append(((user_num, biz_num), score))
            else:
                # 如果找不到特征向量，也使用默认得分
                results.append(((user_num, biz_num), default_score))
    return results


# def calculate_batch_affinity_scores(batch, user_features_bcast, biz_features_bcast):
#     """计算一批用户-商户对的亲和力得分"""
#     results = []
#     for user_id, biz_id in batch:
#         user_features = user_features_bcast.value.get(user_id, None)
#         biz_features = biz_features_bcast.value.get(biz_id, None)
#         if user_features is not None and biz_features is not None:
#             score = np.dot(user_features, biz_features)
#             results.append(((user_id, biz_id), score))
#     return results


# pair_rdd = sc.parallelize([(user_id, biz_id) for user_id, biz_id in zip(list_of_user_ids, list_of_business_ids)])
'''
# # Example of function usage
# data_path = '../data/'  # Update with the actual data path
affinity_scores_rdd, user_features_rdd, biz_features_rdd = matrix_factorization_als(data_path)

print("Example user features:", list(user_features.items())[:1])
print("Example product features:", list(product_features.items())[:1])
'''

#### HELPERS ####
from pyspark.mllib.recommendation import Rating


def log_transform_interaction(interaction_count):
    """对交互次数应用对数转换并标准化到1到5的评分范围"""
    return 1 + 4 * (np.log(interaction_count + 1) - np.log(1 + 1)) / (np.log(284 + 1) - np.log(1 + 1))


'''
Example: 
user_to_num_dict = user_ids_bcast.value
business_to_num_dict = business_ids_bcast.value
user_to_num_bc = sc.broadcast(user_to_num_dict)
business_to_num_bc = sc.broadcast(business_to_num_dict)

# 创建经过对数转换的Rating对象的RDD
ratings_rdd = user_business_interaction_counts.map(
    lambda p: Rating(int(user_to_num_bc.value[p[0][0]]), int(business_to_num_bc.value[p[0][1]]), float(np.float64(log_transform_interaction(p[1]))))
)
'''


####### ratings = interaction_counts.map(lambda x: (user_ids_bcast.value[x[0][0]], business_ids_bcast.value[x[0][1]], x[1]))
# return user_features, biz_features_bcast, user_ids_bcast, business_ids_bcast

# def calculate_affinity_scores(user_features_rdd, biz_features_bcast, user_ids_bcast, business_ids_bcast):
#     """Calculate affinity scores using broadcasted business features and map back IDs to strings"""
#     def calculate_scores(partition):
#         results = []
#         for user_id, user_vec in partition:
#             # Convert numeric ID back to string ID, with error handling
#             user_id_str = user_ids_bcast.value.get(user_id, "Unknown User")
#             for biz_id, biz_vec in biz_features_bcast.value.items():
#                 # Convert numeric ID back to string ID, with error handling
#                 biz_id_str = business_ids_bcast.value.get(biz_id, "Unknown Business")
#                 score = np.dot(user_vec, biz_vec)
#                 results.append(((user_id_str, biz_id_str), score))
#         return results
#     return user_features_rdd.mapPartitions(calculate_scores)
