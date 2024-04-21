# DEMO

Regressive hybrid Recommendation Model: An offline, scalable, supervised recommendation model for user-to-business ratings 

<img width="404" alt="image" src="https://github.com/zhuyuntong/Rating-Predictive-Hybrid-Recommendation-Model/assets/45145164/f258eed9-6a19-4ca3-a266-615bc70432ae">


Mainly SparkRDD implemented. Pandas DF as feature combination and as Model input.

<img width="1728" alt="image" src="https://github.com/zhuyuntong/Rating-Predictive-Hybrid-Recommendation-Model/assets/45145164/53dd222f-89a4-425c-b2bc-27f2395efd8f">

## Table of Contents
- [Model Taxonomy](#model-taxonomy)
- [Matrix Factorization](#1-matrix-factorization-with-als-and-leveraging-block-matrix-calculation-using-rdd)
- [Scaling](#2-scaling)
   - [Using KMeans for User-Business Clustering](#using-kmeans-for-user-business-clustering)
- [Usage](#usage)

# Model Taxonomy
Hybrid: Supervised learning to combine both approaches:
- Content-based
- Collaborative

Terms:
- Models: ALS Matrix Factorization approach, K-means, XGBRegressor, (Graph Node2Vec, Universal Sentence Encoder)
- Packages/framework: SparkRDD, spark.ml, (gensim, spacy, tensorflow, ... )
- Scaling: K-means Clusters identifying localized user-business groups
- Fine-Tunig: Bayesian Optimization Search
- Evaluation Metric: RMSE

# 1. Matrix Factorization with ALS and leveraging Block Matrix Calculation using RDD
- Implicit Interaction Matrix (tip.json: short reviews)
![image](https://github.com/zhuyuntong/DEMO/assets/45145164/8ccb6958-9f1b-4c70-ba86-6fec27d7d36d)

### MF.py Description

This script implements ALS (Alternating Least Squares) for matrix factorization to derive feature vectors for users and businesses based on implicit interactions. Additionally, it includes a logarithmic transformation to standardize raw interaction counts into a more normalized rating scale (from 1 to 5).

#### Key Components:

1. **Matrix Factorization**:
   - Uses `ALS.trainImplicit` to perform matrix decomposition and obtain feature vectors for users and businesses.
   - Defaults scores are used for user-business pairs where feature vectors are not available.

2. **Logarithmic Transformation**:
   - The `log_transform_interaction` function standardizes the interaction counts to a rating scale of 1 to 5.
   - The transformation is applied using the formula:

     <img width="756" alt="image" src="https://github.com/zhuyuntong/Rating-Predictive-Hybrid-Recommendation-Model/assets/45145164/b276d652-dd5b-403e-9659-9a401e7975e1">


### better_features.py Description

This script enhances feature extraction by integrating category information into the user-business scoring data, preparing it for clustering algorithms like KMeans.

#### Key Components:

1. **Category Processing**:
   - The `process_unique_category` function handles and integrates unique category and city data.
   - Merges review and category data for further processing.

2. **Inverse Logarithmic Transformation**:
   - The `inverse_log_transform` function converts normalized scores back to original affinity scores.
   - With a `reverse transformation` in **better_features.py** to enhance XGBRegressor's sensitivity;
   - Both Functions are as following:

      <img width="741" alt="image" src="https://github.com/zhuyuntong/Rating-Predictive-Hybrid-Recommendation-Model/assets/45145164/22bbeb15-dfff-4878-8218-b131515ab921">

     
### Score Scaling Adjustment

To enhance model sensitivity and adjust the scoring scale for specific applications, scores are magnified post-transformation. For instance:

```python
score = float(50 * inverse_log_transform(x[1][1])) if x[1][1] else 0
```

# 2. Scaling
![image](https://github.com/zhuyuntong/DEMO/assets/45145164/ed536903-c224-47d7-bc12-899eb20c32f8)

## Using KMeans for User-Business Clustering

### Overview

In this project, I employ KMeans clustering to understand and segment users based on their geographical interaction patterns with businesses. This clustering approach helps in tailoring marketing strategies and enhancing model accuracy by adapting to localized user behaviors.

### Scripts Analysis

#### KMeans_user_cluster.py

This script performs clustering based on the cities visited by the users. By analyzing these patterns, Model can group users with similar geographical preferences which is valuable for localized marketing and personalized recommendation systems.

**Steps Involved:**
1. **City Visits Extraction**: Identify each city visited by users for each business interaction.
2. **Visit Count Calculation**: Compute how often each city is visited and create a distribution vector of city visits.
3. **Data Standardization**: Standardize the city visit distribution using `StandardScaler` to prepare for clustering.
4. **Cluster Formation**: Implement KMeans clustering with a predefined number of clusters (in this case, 9) to segment users based on their city visit patterns.
5. **Cluster Analysis**: Analyze mean visit frequencies per cluster to understand geographical tendencies of user segments.

#### better_features.py

Complements the clustering by integrating categorical data from user interactions with businesses, providing a richer set of features that can be used to refine user profiles and personalize experiences further.

### Model Scaling and Ensembling

**Scaling Models**
- Clustering allows for the segmentation of users into meaningful groups that can have models tailored to their specific characteristics.
- By adjusting model parameters or strategies for different clusters, performance can be optimized on a per-group basis, enhancing overall efficiency.

**Model Ensembling**
- Combining predictions from multiple models can lead to more accurate and robust predictions.
- Clusters can serve as a meta-feature in larger machine learning frameworks (e.g., input to neural networks or decision trees), helping to predict user behaviors or preferences more effectively.
- Alternatively, different predictive models can be employed for each cluster, and their outputs merged to form a composite prediction, thereby leveraging strengths of various approaches.

### Example Code

```python
# Scaling city visit data
scaler = StandardScaler()
city_matrix_scaled = scaler.fit_transform(city_matrix.drop('user_id', axis=1))

# Applying KMeans
kmeans = KMeans(n_clusters=9, random_state=42)
clusters = kmeans.fit_predict(city_matrix_scaled)

# Analyzing clusters
cluster_city_means = city_matrix.groupby('Cluster')[all_cities].mean()
```

# Usage:
- **better_XGB.py** is the exact script for running everything and output a prediction.csv, in which **KMeans_user_cluster.py** further process features from **better_features.py** in **better_XGB.py**, preparing for Model Training and Evaluation.
- **better_features.py** is a Class for User-Business Interaction Data ETL and provides a interface for train and test data,
- **utils.py** and **KMeans_user_cluster.py** are encapsulated and served as functions. 
  
Deploy on your local Env using the **Dockerfile** provided. 

My workstation is M3 Chip Macbook, you may want to uncomment the 1st line in Dockerfile for building upon AMD-based architecture.

```Dockerfile
# FROM --platform=linux/amd64 ubuntu:20.04 AS builder

# If M-chip Mac, use arm-64; Intel-Chip => amd64
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-arm64
# ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

```






