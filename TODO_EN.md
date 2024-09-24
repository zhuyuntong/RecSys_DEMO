# TODO
# Mixed Recommendation System Project Documentation
## Existing Project Summary

This project implements a regression-based mixed recommendation model primarily used for predicting ratings between users and businesses. Below are the main contents of the project and the algorithms used:

### Implementation Content
- **Regression Mixed Recommendation Model**: An offline, scalable supervised learning model that combines the advantages of content-based recommendations and collaborative filtering.
- **Feature Extraction and Processing**: Enhanced feature extraction by integrating user and business category information, preparing for subsequent clustering algorithms (e.g., KMeans).
- **Model Training and Evaluation**: Utilizes XGBoost and ALS (Alternating Least Squares) for model training and evaluates using RMSE (Root Mean Square Error).

### Algorithms and Functions Used
- **Matrix Factorization**: Uses ALS for implicit interaction matrix factorization, extracting feature vectors for users and businesses.
- **KMeans Clustering**: Clusters users based on geographical interaction patterns to better understand and segment users.
- **Bayesian Optimization**: Used for hyperparameter tuning of the model to improve performance.
- **Inverse Log Transformation**: Converts standardized ratings back to original affinity scores to enhance model sensitivity.

### Code Structure
- `MF.py`: Implements the main logic for ALS matrix factorization.
- `better_features.py`: Enhances feature extraction and processes category information.
- `KMeans_user_cluster.py`: Executes the logic for user clustering.
- `better_XGB.py`: Integrates all functionalities for model training and prediction.

The project is containerized using Docker for easy deployment and execution in local environments.
---

## Project Goals
The project aims to implement a **ranking-focused** mixed recommendation system that utilizes multimodal data (text and images) to generate embeddings and employs XGBoost's LambdaMART for ranking optimization. The main goal of the system is to enhance recommendation accuracy, especially in cold-start user scenarios and visually-driven business contexts.

## Main Technology Stack
- **Python, Spark**: For data processing and model training
- **XGBoost (LambdaMART)**: For ranking optimization
- **Universal Sentence Encoder (USE)**: For generating text embeddings
- **CLIP**: For generating image embeddings
- **ALS (Alternating Least Squares)**: For implicit matrix factorization

## System Architecture
### 1. Ranking Optimization Goal
Switch the existing regression target to a ranking optimization target, using **XGBoost**'s **LambdaMART** to implement a ranking-based recommendation system.
- **Evaluation Metrics**: Use **NDCG** and **MAP** as the main evaluation metrics, replacing the original RMSE (Root Mean Square Error).
- **Grouped Processing**: Define a group of businesses for each user and ensure optimal sorting of business results generated for the same user during training.

#### XGBoost Ranking Code Example:
```python
import xgboost as xgb

params = {
    'objective': 'rank:ndcg',
    'eta': 0.1,
    'max_depth': 6,
    'eval_metric': 'ndcg',
}

group = [len(items) for user, items in user_business_interactions.items()]
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtrain.set_group(group)

bst = xgb.train(params, dtrain, num_boost_round=100)
```
### 2. Text Embedding (Universal Sentence Encoder)

Extract text embeddings from Yelp business reviews using **Universal Sentence Encoder (USE)**. Integrate the embeddings with other user-business features to enhance recommendations for cold-start users.

#### USE Embedding Generation Example:
```python
import tensorflow_hub as hub
import numpy as np

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
business_reviews = load_business_reviews()

business_embeddings = {business_id: embed([review_text]).numpy().mean(axis=0)
                       for business_id, review_text in business_reviews.items()}

def add_use_embeddings_to_features(features, business_id):
    if business_id in business_embeddings:
        embedding = business_embeddings[business_id]
        return np.concatenate([features, embedding])
    return features
```

### 3. Image Embedding (CLIP Model)

Generate image embeddings for Yelp businesses using the **CLIP** model and integrate them with text embeddings to create a multimodal recommendation system.

#### CLIP Embedding Generation and Integration Example:
```python
import torch
import clip
from PIL import Image
import numpy as np

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# load and process image
image = preprocess(Image.open("path_to_yelp_image.jpg")).unsqueeze(0).to(device)

# get image embedding
with torch.no_grad():
    image_embedding = model.encode_image(image).cpu().numpy()

# integrate text and image embedding
combined_embedding = np.concatenate([text_embedding, image_embedding], axis=1)
```

### 4. Multimodal Feature Integration
- **Text Embeddings**: Use **Universal Sentence Encoder** to extract semantic embeddings from review texts.
- **Image Embeddings**: Use **CLIP** model to generate image embeddings for businesses.
- **Multimodal Integration**: Concatenate text and image embeddings and use them as input for the XGBoost model for ranking optimization.

### 5. Model Training and Evaluation
Train the XGBoost LambdaMART ranking model, using NDCG and MAP as evaluation criteria.

```python
from sklearn.metrics import ndcg_score

# Predict and calculate NDCG
y_pred = bst.predict(dtest)
ndcg = ndcg_score(y_true, y_pred)

print(f"NDCG Score: {ndcg}")
```

---

## Further Improvement Directions
1. **Fine-tuning USE**: Fine-tune the Universal Sentence Encoder model based on Yelp data to further enhance text embedding performance.
2. **Multimodal Optimization**: Explore the relationship between image and text embeddings, such as generating image-text similarity through CLIP to enhance multimodal relevance in recommendations.
3. **Model Optimization**: Utilize Bayesian optimization to further adjust hyperparameters of the model, improving the ranking performance of the LambdaMART model.

---

## Project Objective
- **Ranking Optimization**: Achieved ranking optimization through XGBoost's LambdaMART model, enhancing the recommendation system's performance in cold-start and multimodal scenarios.
- **Text Embeddings (USE)**: Generated business review embeddings using the Universal Sentence Encoder, improving recommendation effectiveness for cold-start users, with an increase in relevance.
- **Image Embeddings (CLIP)**: Generated embeddings for business images using CLIP and integrated them into the multimodal recommendation system, achieving an increase in accuracy in visually intensive recommendation scenarios.
- **Multimodal Feature Integration**: Combined text and image embeddings of businesses to create a more robust multimodal recommendation system, enhancing user engagement.

---