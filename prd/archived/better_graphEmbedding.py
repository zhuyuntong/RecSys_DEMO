import sys
import time
import numpy as np
from pyspark import SparkContext
import json
from operator import add
import pandas as pd
from node2vec import Node2Vec as n2v
import networkx as nx
from hashlib import md5
import pickle
import pyspark


def PartHash(string):
    seed = 131
    hash = 0
    for ch in string:
        hash = hash * seed + ord(ch)
    return hash & 0x7FFFFFFF


def initialize_spark_session(APP_NAME="Train: GraphEmbedding"):
    SPARK_CONF = [
        ("spark.executor.memory", "16g"),
        ("spark.executor.cores", "8"),
        ("spark.python.worker.memory", "8g"),
        ("spark.sql.sources.partitionOverWriteMode", "dynamic"),
        ("spark.driver.memory", "8g"),
    ]

    spark_conf = pyspark.SparkConf()
    spark_conf.setAppName(APP_NAME)
    spark_conf.setAll(SPARK_CONF)

    sc = pyspark.SparkContext(conf=spark_conf)
    sc.setLogLevel("ERROR")

    return sc


if __name__ == '__main__':
    # set the path for reading and outputting files
    # folder_path = sys.argv[1]
    # test_filepath = sys.argv[2]
    # output_filepath = sys.argv[3]

    # Uncommon when run at local machine
    folder_path = "../data/"
    test_filepath = "../yelp_val.csv"
    # output_filepath = "../output_task2_2.csv"

    # TODO:
    # connect the spark and set the environment
    sc = initialize_spark_session()
    # sc = SparkContext('local[*]', 'graphembedding').getOrCreate()
    sc.setLogLevel("ERROR")

    # prepare datasets
    yelp_train = folder_path + "/yelp_train.csv"
    ############## only uncomment if train on local (yelp_test is a random dataset created on business_train)
    # yelp_test = folder_path + "yelp_test.csv"
    user = folder_path + "/user.json"
    business = folder_path + "/business.json"

    start_time = time.time()
    # '''
    # read the dataset
    # train_dataset (uid_bid_stars)
    rdd1 = sc.textFile(yelp_train)
    head = rdd1.first()
    train_rdd = rdd1.filter(lambda x: x != head).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    # val_dataset
    rdd2 = sc.textFile(test_filepath)
    head2 = rdd2.first()
    val_rdd = rdd2.filter(lambda x: x != head2).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    # test_dataset
    business_rdd = (
        sc.textFile(business)
        .map(lambda line: json.loads(line))
        .map(lambda x: (x['business_id'], x["categories"], x['city']))
    )
    # user_dataset
    user_rdd = sc.textFile(user).map(lambda line: json.loads(line)).map(lambda x: (x["user_id"], x["friends"]))

    # for all the ids we currently have
    # business_ids
    business_list = list(
        set(train_rdd.map(lambda x: x[1] + "_bid").collect() + val_rdd.map(lambda x: x[1] + "_bid").collect())
    )

    user_list = list(set(train_rdd.map(lambda x: x[0] + "_uid").collect() + val_rdd.map(lambda x: x[0] + "_uid").collect()))

    # deal with business and user dataset
    remaining_business = (
        business_rdd.partitionBy(10, PartHash)
        .map(lambda x: (x[0] + "_bid", x[1]))
        .filter(lambda x: x[0] in business_list)
        .map(
            lambda x: (
                x[0],
                (
                    x[1].replace("&", " ").replace("/", " ").replace("(", "").replace(")", "").replace("  ", "").split(",")
                    if x[1] is not None
                    else ""
                ),
            )
        )
        .map(lambda x: (x[0], [h.strip().lower() for h in x[1]]))
    )

    remaining_user = (
        user_rdd.partitionBy(10, PartHash)
        .map(lambda x: (x[0] + "_uid", [single.strip() + "_uid" for single in x[1].split(",")]))
        .filter(lambda x: x[0] in user_list)
        .map(lambda x: (x[0], [i for i in x[1] if i in user_list]))
    )

    print("BEFORE JOIN PAIRS HAVE FINISHED!")

    # build business pairs and friend pairs
    remaining_business_list = remaining_business.collect()
    business_label_pair = set()
    for single_ele in remaining_business_list:
        b_id = single_ele[0]
        corr_labels = single_ele[1]
        if corr_labels != []:
            for label in corr_labels:
                if label != "":
                    business_label_pair.add(tuple(sorted([b_id, md5(label.encode(encoding='UTF-8')).hexdigest()])))

    remaining_user_list = remaining_user.collect()
    user_friend_pair = set()
    for single_ele in remaining_user_list:
        u_id = single_ele[0]
        u_friends = single_ele[1]
        if u_friends != []:
            for friend in u_friends:
                if friend != "" and friend != "_uid":
                    user_friend_pair.add(tuple(sorted([u_id, friend])))
    new_final_list = list(business_label_pair) + list(user_friend_pair)

    # for all original pairs
    train_pair = train_rdd.map(lambda x: (x[0], x[1])).collect()
    val_pair = val_rdd.collect()
    ################### only uncomment if train on local.
    # test_pair = test_rdd.collect()
    # final_list = train_pair+val_pair+test_pair
    final_list = train_pair + val_pair
    print("FINISH BUILDING THE PAIRS.")

    # generating relationship
    with open('GraphEdgeInfo.txt', 'w+') as f:
        for pair_original in final_list:
            f.write(f'{pair_original[0]+"_uid"} {pair_original[1]+ "_bid"}\n')
        for pair_new in new_final_list:
            f.write(f'{pair_new[0]} {pair_new[1]}\n')
    print("FINISH WRITING THE PAIRS.", "TIME:", time.time() - start_time)

    # generate graph
    # TODO
    start_time = time.time()

    graph = nx.read_edgelist('GraphEdgeInfo.txt', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    print("FINISH BUILDING THE GRAPH", "TIME:", time.time() - start_time)
    # embeddings
    # model = n2v(graph, dimensions=200, walk_length=25, num_walks=250, p=0.25, q=4, workers=16)
    model = n2v(graph, dimensions=300, walk_length=30, num_walks=300, p=0.25, q=2, workers=16)
    n2v_file = "models/Node2Vec_dim300_l30_n300_q2_w15"

    # model = n2v(graph, dimensions=10, walk_length=5, num_walks=5, p=0.25, q=4, workers=10)  # FIXME: 和底下保持一致
    model2 = model.fit(window=15)  # 15, min_count=1, batch_words=4)  # train model
    with open(n2v_file, "wb") as model_file:
        pickle.dump(model2, model_file)

    print("FINISH TRAINING THE MODEL", "TIME:", time.time() - start_time)

    # '''
    # model2 = pickle.load(open("Node2Vec", "rb"))
    # '''

    # TODO
    n2v_file = "models/Node2Vec_dim300_l30_n300_q2_w15"
    start_time = time.time()
    with open(n2v_file, "rb") as model_file:
        model2 = pickle.load(model_file)

    info_dict = {}
    with open("GraphEdgeInfo.txt", "r") as f:
        for line in f.readlines():
            for single in line.split():
                if single in info_dict:
                    continue
                else:
                    info_dict[single] = list(model2.wv.get_vector(single))

    df_embeddings = (
        pd.DataFrame.from_dict(
            info_dict, orient='index', columns=[f'E_{_}' for _ in range(300)]  # FIXME
        )  # FIXME:根据dimension设置为200
        .reset_index()
        .rename(columns={'index': 'id'})
    )
    df_embeddings.to_csv('models/VectorizedFeatures_dim300_l30_n300_q2_w15.csv', index=False)

    print("FINISH vectorized feature.", "TIME:", time.time() - start_time)
