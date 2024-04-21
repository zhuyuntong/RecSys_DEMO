import pyspark
import json
import sys
from pyspark.sql import Window
import random
import csv
from io import StringIO
import time
from itertools import combinations

APP_NAME = "Task1: MinHash+LSH"

SPARK_CONF = [
    ("spark.dynamicAllocation.maxExecutors", "100"),
    ("spark.executor.memoryOverhead", "2000"),
    ("spark.executor.memory", "8g"),
    ("spark.executor.cores", "4"),
    ("spark.driver.maxResultSize", "0"),
    ("spark.python.worker.memory", "8g"),
    ("spark.executor.memoryOverhead", "3000"),
    ("spark.sql.shuffle.partitions", "10000"),
    ("spark.sql.sources.partitionOverWriteMode", "dynamic"),
    ("spark.driver.memory", "8g"),
]

spark_conf = pyspark.SparkConf()
spark_conf.setAppName(APP_NAME)
spark_conf.setAll(SPARK_CONF)

sc = pyspark.SparkContext(conf=spark_conf)
spark = pyspark.sql.SparkSession(sc)
sc.setLogLevel("ERROR")


def process_csv():
    def parse_arguments():
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        return input_file_path, output_file_path

    input_file, output_file = parse_arguments()
    rdd = sc.textFile(input_file).filter(lambda x: not x.startswith("user_id"))
    # raw_rdd = sc.textFile(input_file_path).map(parse_csv)

    # remove headers, get a clean rdd
    # headers = raw_rdd.first()
    # rdd = raw_rdd.filter(lambda x: x != headers)

    # create business-user pair => [(b1, u1), (b1, u2), ...)]
    bus_user_pairs = rdd.map(lambda x: x.split(",")).map(lambda x: (x[1], x[0]))

    # and a dict for user/bus => e.g. {"b1": 0, "b2": 1, "b3": 2..}, {"u1": 0, "u2": 1, ..}
    bus_dict = bus_user_pairs.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
    user_dict = bus_user_pairs.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()

    return bus_user_pairs, bus_dict, user_dict, output_file


def binary_matrix(bus_user_pairs, bus_dict, user_dict):
    '''
    transform bus-user pairs into a binary matrix
    return a binary matrix, bus_id as row, user_id as col.
    '''
    # [(b1, u1), (b1, u2), ...)] -> [("b1", {"u1", "u2"}), ("b2", {"u3", "u1, u4"}), ...]
    bus_user_group = bus_user_pairs.groupByKey().mapValues(set)

    def extract_row_indexing(x):
        business_index = bus_dict[x[0]]
        user_indices = [user_dict[user] for user in x[1]]
        return (business_index, user_indices)

    business_user_indexed_RDD = bus_user_group.map(extract_row_indexing)
    return business_user_indexed_RDD


# FIXME
def hash_functions(user_index):
    global hash_params, m
    return [((a * user_index + b) % m) for (a, b) in hash_params]


def hash_fn(x, a, b, m):
    return (a * x + b) % m


# FIXME
def min_hash(user_indices):
    '''
    find business's signature; apply on RDD

    scan through sparse matrix (binary_matrix_rdd) row by row,
    apply min_hash fn(s) => sign matrix
    each row will be given a same set of hash functions, m hash functions (a, b)
    '''
    # initially, Sign matrix val = inf;
    # Sign(h_k, S_j) = min(Input(i, h_k), Sign(h_k, S_j))
    global min_hash_params, m

    sig = []
    for a, b in min_hash_params:
        # for each min-hash fn, rehash a list of user indices, pick the lowest rehashed index
        lowest_sig_vals = min([hash_fn(i, a, b, m) for i in user_indices])
        sig.append(lowest_sig_vals)
    return sig


if __name__ == "__main__":
    # TODO:
    start_time = time.time()
    # Step 1: Preprocess csv data
    # RETURN a dict for indexing user/bus => e.g. {"b1": 0, "b1": 1, ..}, {"u1": 0, "u2": 1, ..}
    bus_user_pairs, bus_dict, user_dict, output_file = process_csv()

    # Step 2: Define a group of hash functions
    num_hash_functions = 50
    m = len(user_dict)

    # generate # min_hash (a,b) hash pairs by given num
    min_hash_params = [
        (random.randint(0, m), random.randint(0, m)) for _ in range(num_hash_functions)
    ]

    # Step 3: Create a sparse matrix RDD for bus-user, business as row, user as col.
    binary_matrix_rdd = binary_matrix(bus_user_pairs, bus_dict, user_dict)

    # Step 4: min_hash impl
    sign_matrix_rdd = binary_matrix_rdd.mapValues(min_hash).cache()

    # LSH
    rows_per_band = 2  # b, adjust, b*r = n
    num_bands = num_hash_functions // rows_per_band

    # Step 5: divide matrix -> b bands w/ r rows
    def split_into_bands(signature):
        '''slice columns to bands'''
        return [
            signature[i * rows_per_band : (i + 1) * rows_per_band] for i in range(num_bands)
        ]

    bus_bands_rdd = sign_matrix_rdd.mapValues(split_into_bands)

    # Step 6: candidates pairs
    candidate_pairs_rdd = (
        bus_bands_rdd.flatMap(
            lambda x: [((i, tuple(band)), x[0]) for i, band in enumerate(x[1])]
        )
        .groupByKey()
        .filter(lambda x: len(x[1]) > 1)  # hash to same buckets
        .flatMap(lambda x: combinations(x[1], 2))  # generate candidate pairs
        .distinct()
    ).cache()

    # FIXME: TODO:

    def create_business_to_users_map(bus_user_pairs):
        return bus_user_pairs.groupByKey().mapValues(set).collectAsMap()

    business_to_users = create_business_to_users_map(bus_user_pairs)  # test
    business_to_users_broadcast = sc.broadcast(business_to_users)

    index_to_business_dict = {v: k for k, v in bus_dict.items()}
    index_business_broadcast = sc.broadcast(index_to_business_dict)

    # Calculate Jaccard similarity
    def jaccard_similarity(pair):
        business_to_users = business_to_users_broadcast.value
        index_business = index_business_broadcast.value

        business_id1 = index_business[pair[0]]
        business_id2 = index_business[pair[1]]

        user_set1 = business_to_users[business_id1]
        user_set2 = business_to_users[business_id2]
        intersection = len(user_set1.intersection(user_set2))
        union = len(user_set1.union(user_set2))

        return intersection / union if union != 0 else 0

    # Filter and map similar pairs RDD
    similarity_threshold = 0.5
    similar_pairs_rdd = (
        candidate_pairs_rdd.map(lambda pair: (pair, jaccard_similarity(pair)))
        .filter(lambda x: x[1] >= similarity_threshold)
        .map(lambda x: (x[0][0], x[0][1], x[1]))
    )

    # Prepare output data
    index_business = index_business_broadcast.value
    output_data = (
        similar_pairs_rdd.map(
            lambda x: (
                index_business[x[0]],
                index_business[x[1]],
                x[2],
            )
        )
        .map(lambda x: (min(x[0], x[1]), max(x[0], x[1]), x[2]))
        .sortBy(lambda x: (x[0], x[1]))
    )

    # Write to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["business_id_1", " business_id_2", " similarity"])
        for row in output_data.collect():
            writer.writerow(row)

    print("Duration: ", time.time() - start_time)
    sc.stop()


'''
    # print(
    #     "{} minhash fn params: {},  len: {}".format(
    #         len(min_hash_params), min_hash_params, len(min_hash_params)
    #     )
    # )

    # Create a dictionary representation of the binary matrix
    matrix_dict = create_matrix_dict(binary_matrix_rdd)

    # Example usage
    business_id_example = 0  # replace with actual business index
    indice_id_example = 63  # replace with actual user index
    value = get_matrix_value(matrix_dict, business_id_example, indice_id_example)
    print(f"Value at matrix[{business_id_example}][{indice_id_example}]: {value}")
    
    # with open(output_file, "w") as f:
    #     f.write(sign_matrix_rdd.collect())
'''


# FIXME: For DEBUG Purpose
def create_matrix_dict(binary_matrix_rdd):
    matrix_dict = {}
    for business_index, user_indices in binary_matrix_rdd.collect():
        matrix_dict[business_index] = {user_index: 1 for user_index in user_indices}
    return matrix_dict


# FIXME: For DEBUG Purpose
def get_matrix_value(matrix_dict, business_id, indice_id):
    return matrix_dict.get(business_id, {}).get(indice_id, 0)
