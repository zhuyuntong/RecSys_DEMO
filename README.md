# Update Note:
## Timeline
- [2024/08-09: Implementation of Multimodal embeddings and Two-Tower Model](##202408-09)
- [2024/07: Design of Migrating to Multimodal Top-K Personalized Ranking Recommendation](#20240728)
- [2024/03-05: Legacy Score Prediction Hybrid Model](#personal-project-demo)
## 2024/08-09
### Progress
Implemented notebook `multimodalRanking.ipynb`, covers several aspects mentioned in the `TODO_EN.md`.
Showcased the usage of multimodal methods (such as BERT/USE/CLIP) to handle various data sources and use these features for downstream tasks like rank-based recommendations and classification.

### Methodology
#### How Review and Image Embeddings Are Fed into the Recommendation Model
#### 1. Embedding Extraction
- **Review Embeddings**: Extracted using the BERT model. Each review is processed by the `get_text_embedding` function, generating a 768-dimensional vector.
- **Image Embeddings**: Extracted using the CLIP model. Each image is processed by the `get_image_embedding` function, generating a 512-dimensional vector.

#### 2. Feature Combination
During training, review and image embeddings are combined with other user and business features. The specific steps are as follows:
- **User Features**: Extracted from user data (e.g., behavioral characteristics).
- **Business Features**: Extracted from business data (e.g., geographic location, categories).

#### 3. Dataset Construction
The `YelpDataset` class builds the dataset by combining user indices, positive business indices, and negative business indices. Each sample consists of:
- **User Features**: Retrieved from the user feature dictionary.
- **Positive Business Features**: Retrieved from the business feature dictionary.
- **Negative Business Features**: Retrieved from the business feature dictionary.

#### 4. Model Input
During training, the `get_user_features` and `get_business_features` functions are called to retrieve user and business features, including the review and image embeddings. The steps are:
- **User Features**: Retrieved using the `get_user_features` function, returning a feature vector for the user.
- **Business Features**: Retrieved using the `get_business_features` function, which includes business features, review embeddings, and image embeddings.

#### 5. Model Forward Pass
In the `forward` method of the `TwoTowerModel`, user and business features are fed into the model, passing through the user tower and business tower respectively. These features generate user vectors and business vectors, which are then used to compute contrastive loss for training.

The review and image embeddings are fed into the recommendation model through a multi-step process of extraction, feature combination, and dataset construction. These embeddings, along with other user and business features, serve as input features for model training and inference.

### Overview of the notebook process
#### 1. **Data Preparation Module**
- **Function**: Load and preprocess data to provide foundational input for model training and recommendations.
- **Relevant Code Blocks**: 
  - in[0]: Imports and Device Setup
  - in[1]: Data Loading and Preprocessing
  - in[2]: Feature Scaling and Category Processing
  - in[3]: Business Feature Extraction

#### 2. **Embedding Extraction Module**
- **Function**: Extract embeddings from reviews and images, using BERT for text and CLIP for image data processing.
- **Relevant Code Blocks**: 
  - in[2]: BERT Text Embedding
  - in[14]: Image Embedding Extraction
  - in[17]: Text Embedding Extraction

#### 3. **Model Definition Module**
- **Function**: Define the structure of the recommendation model, including user and business towers, and implement the contrastive loss function.
- **Relevant Code Blocks**: 
  - in[5]: Model Definition
  - in[6]: Contrastive Loss Function

#### 4. **Training and Evaluation Module**
- **Function**: Train the recommendation model and evaluate its performance using metrics like loss, NDCG, and MAP.
- **Relevant Code Blocks**: 
  - in[4]: Distance Calculation (Dataset Creation)
  - in[7]: Training Function
  - in[8]: Evaluation Function

#### 5. **Ranking and Recommendation Module**
- **Function**: Generate business recommendations for users based on the trained model and rank them.
- **Relevant Code Blocks**: 
  - in[10]: Recommendation Function
  - in[13]: Business Recommendations

#### 6. **Similarity Analysis Module**
- **Function**: Analyze the similarity between reviews and businesses to improve recommendation quality.
- **Relevant Code Blocks**: 
  - in[14]: Similarity Heatmap
  - in[19]: Text Similarity Heatmap

#### 7. **Visualization Module**
- **Function**: Visualize user features, recommendation results, and similarity analysis to better understand model performance.
- **Relevant Code Blocks**: 
  - in[11]: User Feature Visualization

#### 8. **Image Classification Module**
- **Function**: Process image data, extract image embeddings, train classification models, and plot confusion matrices to support the recommendation system.
- **Relevant Code Blocks**: 
  - in[15]: Image Classification
  - in[16]: Home Services Classification
---

This notebook covers several aspects, including recommendation systems, text analysis, and image classification. It demonstrates how to leverage deep learning models (such as BERT and CLIP) to handle various data types and use these features for downstream tasks like recommendation and classification.

## Technical challenges
 1. Define the Objectives
    - Aim: personalized ranked order of business for each user
    - Metric: NDCG relevance level: strength of interaction (2.0 > 1.0)
    - Group Definition: based on both user and location
    - ISSUE: lack of geo/implicit interaction data;
 2. Negative Sampling for the hybrid user-business model
    - ISSUE: not considering all non-interacted businesses as negatives, lack of a **recall step** with a selection criteria to pick only the recalled candidates for model training.
    - ISSUE: intensive analysis such as location radius, user preferences, etc., are required. 
 3. Scaling predictions
    - BLAS, file-based processing, etc.

### TODO:
Further work shall focus on integrating with existing ETL feature-eng scripts using max/avg pooling, evaluating XGB LambdaMART approach, refined negative sampling with Recall (more interaction data release by Yelp), model fine-tuning or comparison. 

Others:
- Features & Negative sampling: consider Word2Vec (https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- Text Similarity: Contextual / Semantic-Matching / Attention-based Model
- Image Classification: EfficientNet/ImageNet

> ## Visualization
> Graph1: Cosine Similarity Heatmap of Review Embeddings across various categories applied on a sample of Yelp Review data
> ![image](https://github.com/user-attachments/assets/d2bcb6b9-cf1c-4d79-bab6-2ca484cb959c)
> This plot showcases the **cosine similarity** between review embeddings from various categories. By computing the similarities between these embeddings and visualizing them as a heatmap, we can observe the following:
> - **Same-category reviews**: Reviews within the same category tend to be closer to each other in vector space, reflected by brighter areas in the heatmap.
> - **Different-category reviews**: Reviews from different categories are relatively farther apart in vector space, shown as darker areas in the heatmap.
> This indicates that the model successfully captures the similarity between reviews, especially within the same category.
> 
> Graph 2: CLIP Photo Cliassification Precision Table
> ![image](https://github.com/user-attachments/assets/1e61aea0-dd89-443b-86fb-64cace99017d)
> **Content**: This plot is a **confusion matrix**. It shows the precision with which the CLIP model predicted the hand-labeled Yelp dataset. More importantly, it shows which categories get mixed up together
> - **X-axis**: Represents the predicted labels by the model.
> - **Y-axis**: Represents the true labels.
> 
> Graph 3: Home Services Contractor Classifier ('Inside' and 'Outside')
> ![image](https://github.com/user-attachments/assets/29b4262e-95f9-4ee4-aad6-4409404218e2)
> Assess how accurately the model recognizes images related to home services. CLIP offers the possibility to remove the restraint of needing a large number of examples for each category the model infers. A review of the possible output classes from CLIP will lead to more diversified content tags on Yelp.
>
> Graph 4: A more complete overview of text-based cosine similarity of Yelp's review embeddings
> ![image](https://github.com/user-attachments/assets/a2d32350-e5c2-4809-9b6c-eca0ef47b71c)
> Similar to Graph 1.


## 2024/07/28
The new document `TODO_EN.md` introduces a ranking-focused mixed recommendation system that utilizes multimodal data (text and images) to generate embeddings and employs XGBoost's LambdaMART for ranking optimization. Current version is a regressive model, and the new version will be a ranking model. `README.md` explains the previous version, and `TODO_EN.md` explains the new version. Here is a snapshot of the new version:


>   ## Project Goals
>   The project aims to implement a **ranking-focused** mixed recommendation system that utilizes multimodal data (text and images) to generate embeddings and employs XGBoost's LambdaMART for ranking optimization.
> The model's task is to recommend personalized ranked order of business for each user. The main optimization goal is to enhance recommendation accuracy with multi-modal content-based embeddings, especially in cold-start user scenarios and visually-driven business contexts.

>   ## Main Technology Stack
>   - **Python, Spark**: For data processing and model training
>   - **XGBoost (LambdaMART)**: For ranking optimization
>   - **Universal Sentence Encoder (USE)**: For generating text embeddings
>   - **CLIP**: For generating image embeddings
>   - **ALS (Alternating Least Squares)**: For implicit matrix factorization
>   ## System Architecture
>   ### Ranking Optimization Goal
>   Switch the existing regression target to a ranking optimization target, using **XGBoost**'s **LambdaMART** to implement a ranking-based recommendation system.
>    - **Evaluation Metrics**: Use **NDCG** and **MAP** as the main evaluation metrics, replacing the original RMSE (Root Mean Square Error).
>    - **Grouped Processing**: Define a group of businesses for each user and ensure optimal sorting of business results generated for the same user during training.
>
---

# Personal Project DEMO

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
- Fine-Tuning: Bayesian Optimization Search (hyperopt version for spark-based)
- Evaluation Metric: RMSE

# 1. Matrix Factorization with ALS and leveraging Block Matrix Calculation using RDD
- Implicit Interaction Matrix (tip.json: short reviews)
![image](https://github.com/zhuyuntong/DEMO/assets/45145164/8ccb6958-9f1b-4c70-ba86-6fec27d7d36d)

### MF.py Description

This script implements ALS (Alternating Least Squares) for matrix factorization to derive feature vectors for users and businesses based on implicit interactions from `tips.json`. Additionally, it includes a logarithmic transformation to standardize raw interaction counts into a more normalized rating scale (from 1 to 5).

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






