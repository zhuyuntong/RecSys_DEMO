import pandas as pd
from hashlib import md5
from collections import defaultdict


## category: category md5 mapping, df = feature_processor.business_df
def create_category_md5_mapping(df):
    """
    This function creates a mapping table for business's category strings to their corresponding MD5 hash.

    Args:
    df (pd.DataFrame): DataFrame containing a column 'categories' with category strings.

    Returns:
    pd.DataFrame: A DataFrame with original business's category strings and their corresponding MD5 hashes.
    """
    # Ensure the input column is exploded to have one category per row
    exploded_df = df.explode('categories').dropna(subset=['categories']).drop_duplicates(subset=['categories'])

    # Calculate MD5 hash for each category
    exploded_df['category_md5'] = exploded_df['categories'].apply(lambda x: md5(x.encode('UTF-8')).hexdigest())

    # Create a DataFrame to store the mapping from category to MD5
    mapping_df = exploded_df[['categories', 'category_md5']].drop_duplicates()
    return mapping_df


### HELPERS
def __process_bid_tail(cat):
    """process business_id, remove ['cat'] col's ending w/ '_bid'."""
    if cat.endswith('_bid'):
        return cat[:-4]
    return cat


def __add_user_ids(enhanced_category_md5_df, review_df):
    """extract user_id from review_df, merge it with enhanced_category_md5_df."""
    final_df = enhanced_category_md5_df.merge(review_df[['user_id', 'business_id']], on='business_id', how='left')
    return final_df


def __add_business_ids(category_md5_df, df_bus_conn):
    """将category_md5_df与df_bus_conn进行合并, 添加business_id."""
    merged_df = category_md5_df.merge(
        df_bus_conn[['business_id', 'type']], left_on='category_md5', right_on='type', how='left'
    )
    merged_df.drop(columns=['type'], inplace=True)  # drop temp col
    return merged_df


### HELPERS
def integrate_mapping_user_bus_cat_data(df_conn, business_df, review_df):
    """combine all steps, get a final DataFrame including categories、corresponding MD5 num、business_id and user_id."""
    # Step 1: Generate connection DataFrame with categories and business IDs
    # Step 2: Process business IDs to remove '_bid' tail
    df_conn['business_id'] = df_conn['bid_cat'].apply(__process_bid_tail)
    df_conn.drop(columns=['bid_cat'], inplace=True)

    # Step 3: Merge to add business IDs to category MD5 DataFrame
    # merged_df = __add_business_ids(category_md5_df, df_conn)

    # category: category md5 mapping
    category_md5_df = create_category_md5_mapping(business_df)

    merged_df = category_md5_df.merge(
        df_conn[['business_id', 'type']], left_on='category_md5', right_on='type', how='left'
    ).drop(columns=['type'])

    # Step 4: Add user IDs using the review data
    final_mapping_df = __add_user_ids(merged_df, review_df)
    # Return the final DataFrame
    return final_mapping_df


### DF ==> Dict: {(category, md5 category val) , (biz_id, user_id)}
################################ USE EFFICIENT DICT FOR FUTURE SEARCH
def dataframe_to_rdd_dict(sc, final_mapping_df):
    # 将DataFrame转换为RDD
    data_rdd = sc.parallelize(final_mapping_df.to_dict(orient='records'))

    # 映射RDD以创建所需的键值对格式
    mapped_rdd = data_rdd.map(
        lambda row: (
            (row['categories'], row['category_md5']),
            (set([row['business_id']]), set([x for x in [row['user_id']] if pd.notna(x)])),
        )
    )
    # 使用reduceByKey来合并具有相同键的值 # 可以cache或者persist
    reduced_rdd = mapped_rdd.reduceByKey(lambda a, b: (a[0].union(b[0]), a[1].union(b[1]))).persist()

    # 收集结果回本地字典
    result_dict = reduced_rdd.collectAsMap()
    return result_dict


def analyze_top_business_categories(result_dict, business_df, reverse=True, top_K=10):
    """
    Find top 10 categories with the most unique businesses and users, and calculate their proportions in the business dataset.

    Args:
    result_dict (dict): Mapping table with (categories, category_md5) as keys and (business_set, user_set) as values.
    business_df (pd.DataFrame): Processed business dataset with categories and other business-related info.

    Returns:
    pd.DataFrame: DataFrame containing the analysis results.
    """
    # Calculate counts for businesses and users per category
    business_count = {key: len(value[0]) for key, value in result_dict.items()}
    user_count = {key: len(value[1]) for key, value in result_dict.items()}

    # Sort and get top 10 categories for businesses and users
    top_business_categories = sorted(business_count.items(), key=lambda item: item[1], reverse=reverse)[:top_K]
    top_user_categories = sorted(user_count.items(), key=lambda item: item[1], reverse=True)[:10]

    # Flatten category data and prepare for proportion calculation
    total_businesses = len(business_df['business_id'].unique())
    category_list = [item[0][0] for item in top_business_categories] + [item[0][0] for item in top_user_categories]
    category_set = set(category_list)

    # Calculate category proportions
    category_proportions = {}
    for category in category_set:
        num_businesses = business_df['categories'].explode().eq(category).sum()
        category_proportions[category] = num_businesses / total_businesses

    # Prepare final DataFrame
    result_data = {
        "Category": [cat[0][0] for cat in top_business_categories],
        "Business Count": [cat[1] for cat in top_business_categories],
        "User Count": [user_count[cat[0]] for cat in top_business_categories],
        "Proportion in Dataset": [category_proportions[cat[0][0]] for cat in top_business_categories],
    }
    result_df = pd.DataFrame(result_data)

    return result_df


def analyze_top_categories(result_dict, business_df, dimension, total_user=None, reverse=True, top_K=10):
    """
    Find top 10 categories with the most unique entities based on the specified dimension (businesses or users),
    and calculate their proportions in the business dataset.

    Args:
    result_dict (dict): Mapping table with (categories, category_md5) as keys and (business_set, user_set) as values.
    business_df (pd.DataFrame): Processed business dataset with categories and other business-related info.
    dimension (str): The dimension to analyze ("business" or "user").

    Returns:
    pd.DataFrame: DataFrame containing the analysis results.
    """
    # Calculate counts for businesses and users per category
    entity_count = {key: len(value[0]) if dimension == "business" else len(value[1]) for key, value in result_dict.items()}

    # Sort and get top 10 categories for the specified dimension
    top_categories = sorted(entity_count.items(), key=lambda item: item[1], reverse=reverse)[:top_K]

    # Flatten category data and prepare for proportion calculation
    if dimension == "business":
        total_entities = len(business_df['business_id'].unique())
    else:
        # # Correcting the set union operation for user sets
        # all_user_sets = [value[1] for value in result_dict.values()]  # extract all user sets
        # total_entities = len(set.union(*all_user_sets))
        if total_user is None:
            raise ValueError("Total user count must be provided for 'user' dimension analysis.")
        total_entities = total_user

    category_list = [item[0][0] for item in top_categories]
    category_set = set(category_list)

    # Calculate category proportions
    category_proportions = {}
    for category in category_set:
        if dimension == "business":
            num_entities = business_df['categories'].explode().eq(category).sum()
        else:
            num_entities = entity_count[(category, md5(category.encode('UTF-8')).hexdigest())]

        category_proportions[category] = num_entities / total_entities

    # Prepare final DataFrame
    result_data = {
        "Category": [cat[0][0] for cat in top_categories],
        f"{dimension.capitalize()} Count": [cat[1] for cat in top_categories],
        "Proportion in Dataset": [category_proportions[cat[0][0]] for cat in top_categories],
    }
    result_df = pd.DataFrame(result_data)

    return result_df


# def integrate_mapping_user_bus_cat_data(df_conn, category_md5_df, review_data):
#     """整合所有步骤, 生成最终的DataFrame包含categories、MD5散列、business_id和user_id."""
#     # Step 1: Generate connection DataFrame with categories and business IDs
#     # df_conn = feature_processor.df_conn

#     # # Step 2: Process business IDs to remove '_bid' tail
#     df_conn['business_id'] = df_conn['bid_cat'].apply(__process_bid_tail)
#     df_conn.drop(columns=['bid_cat'], inplace=True)

#     # Step 3: Merge to add business IDs to category MD5 DataFrame
#     merged_df = __add_business_ids(category_md5_df, df_conn)

#     # Step 4: Add user IDs using the review data
#     final_mapping_df = __add_user_ids(merged_df, review_data)

#     # Return the final DataFrame
#     return final_mapping_df

# def __add_user_ids(enhanced_category_md5_df, review_data):
#     """从review_data中提取user_id, 并与enhanced_category_md5_df合并."""
#     review_list = [item[0] for item in review_data]  # 提取字典
#     df_reviews = pd.DataFrame(review_list)
#     final_df = enhanced_category_md5_df.merge(df_reviews[['user_id', 'business_id']], on='business_id', how='left')
#     return final_df
