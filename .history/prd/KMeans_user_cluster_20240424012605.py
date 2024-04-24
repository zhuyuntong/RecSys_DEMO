import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

'''
Prereq: utils.py category_md5_df mapping_review_df business_df

使用用户评论数据 (processed_review_df from utils.py) 和商户数据 (feature_processor.self.business_df) 进行用户的聚类分析。
计算每个用户访问过的商户ID集合,并创建了用户与商户城市的映射。
创建了一个城市矩阵，包含用户访问不同城市的分布。
使用 StandardScaler 对城市矩阵数据进行了标准化。
应用了 KMeans 聚类算法,并设置了9个聚类中心, random_seed=42, (k=9, random_state=42)经过elbow,Silhouette, 具体城市簇的含义，实验得出。
计算了每个聚类中城市的平均访问次数，并创建了聚类和城市的映射。

最后，函数返回了两个 DataFrame:user_clusters 包含用户ID和对应的聚类标签;clusters_cities 包含每个聚类及其对应的城市列表。

后续(4/16):可能可以考虑别的数据结构输出，看实际情况；
更新:clusters_cities筛选了代表城市
'''


# user_clusters, clusters_important_cities, business_df = KMeans_process_user_clusters(mapping_review_df, business_df)
def KMeans_process_user_clusters(review_df, business_df):
    '''
    根据用户和城市间的访问模式进行聚类，为后续**本地化**用户群组和商家进行针对性学习 (验证了用户的确有这个倾向);

    使用全量的review, business数据进行用户聚类分析
    加载和处理数据,然后应用K-means聚类算法,最后组织结果
    返回两个DataFrame:一个包含用户和对应聚类,另一个包含聚类中的城市列表
    '''

    # Business ID Sets visited by Users
    user_business_sets = review_df.groupby('user_id')['business_id'].agg(lambda x: set(x)).reset_index()
    user_business_sets.columns = ['user_id', 'visited_business_ids']

    # Business-City Mapping
    business_city_map = business_df.set_index('business_id')['city'].to_dict()

    # Num. City Visited
    def count_city_visits(business_ids):
        city_counts = {}
        for bid in business_ids:
            city = business_city_map.get(bid, None)
            if city:
                city_counts[city] = city_counts.get(city, 0) + 1
        return city_counts

    user_business_sets['city_visit_distribution'] = user_business_sets['visited_business_ids'].apply(count_city_visits)

    # City Matrix
    all_cities = set()
    for dist in user_business_sets['city_visit_distribution']:
        all_cities.update(dist.keys())
    all_cities = list(all_cities)

    def create_city_vector(city_distribution):
        return [city_distribution.get(city, 0) for city in all_cities]

    city_matrix = pd.DataFrame(
        user_business_sets['city_visit_distribution'].apply(create_city_vector).tolist(), columns=all_cities
    )
    city_matrix.insert(0, 'user_id', user_business_sets['user_id'])

    # Standardize counts
    scaler = StandardScaler()
    city_matrix_scaled = scaler.fit_transform(city_matrix.drop('user_id', axis=1))

    # K-means聚类
    kmeans = KMeans(n_clusters=9, random_state=42)  # fix seed=42, this achieves the best performance => 8/9 (9 better score)
    clusters = kmeans.fit_predict(city_matrix_scaled)
    city_matrix['Cluster'] = clusters

    # 用户聚类结果
    user_clusters = city_matrix[['user_id', 'Cluster']].copy()

    ''' NOTE: for future use e.g. city analysis 
    # 计算每个聚类的城市访问均值
    # cluster_city_means = city_matrix.groupby('Cluster')[all_cities].mean()
    # 聚类中的城市
    # clusters_cities = pd.DataFrame(columns=['Cluster', 'Cities'])
    # for cluster_num in cluster_city_means.index:
    #     cities_in_cluster = cluster_city_means.loc[cluster_num][cluster_city_means.loc[cluster_num] > 0]
    #     clusters_cities = pd.concat([clusters_cities, pd.DataFrame({'Cluster': [cluster_num], 'Cities': [cities_in_cluster.index.tolist()]})], ignore_index=True)

    # # 筛选代表性的城市
    # important_cities_per_cluster = {}
    # for cluster_num in cluster_city_means.index:
    #     # 获取该聚类中城市的平均访问次数
    #     cities_in_cluster = cluster_city_means.loc[cluster_num]
    #     # 计算重要城市的阈值
    #     threshold = cities_in_cluster.max() * 0.05
    #     # 筛选出重要的城市
    #     important_cities = cities_in_cluster[cities_in_cluster >= threshold].index.tolist()
    #     # 将结果存储在字典中
    #     important_cities_per_cluster[cluster_num] = important_cities

    # clusters_important_cities = pd.DataFrame(
    #     list(important_cities_per_cluster.items()), columns=['Cluster', 'Important_Cities']
    # )

    # # 集成城市和商户信息, business_df增加两列cluster和important cities;
    # test_df_exploded = clusters_important_cities.explode('Important_Cities')
    # business_df = pd.merge(test_df_exploded, business_df, left_on='Important_Cities', right_on='city', how='left')

    # return user_clusters, clusters_important_cities, business_df
    '''
    return user_clusters


## Example:
from utils import create_category_md5_mapping, integrate_mapping_user_bus_cat_data

# category_md5_df = create_category_md5_mapping(feature_processor.business_df)
# mapping_review_df = integrate_mapping_user_bus_cat_data(feature_processor.df_conn, category_md5_df, review_data)
# business_df = feature_processor.business_df || pd.read_csv('../well-trained/cache/business_df.csv')
# user_clusters, clusters_important_cities, business_df = KMeans_process_user_clusters(mapping_review_df, business_df)
