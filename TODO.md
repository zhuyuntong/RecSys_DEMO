# TODO
# 混合推荐系统项目文档
## 现有项目总结

该项目实现了一个回归混合推荐模型，主要用于用户与商家之间的评分预测。以下是项目的主要内容和使用的算法：

### 实现内容
- **回归混合推荐模型**：一个离线、可扩展的监督学习模型，结合了内容推荐和协同过滤的优点。
- **特征提取与处理**：通过整合用户与商家的类别信息，增强了特征提取的效果，为后续的聚类算法（如KMeans）做准备。
- **模型训练与评估**：使用XGBoost和ALS（交替最小二乘法）进行模型训练，并通过RMSE（均方根误差）进行评估。

### 使用的算法与功能
- **矩阵分解**：使用ALS进行隐式交互的矩阵分解，提取用户和商家的特征向量。
- **KMeans聚类**：对用户进行地理交互模式的聚类，以便更好地理解和细分用户。
- **贝叶斯优化**：用于模型的超参数调优，提升模型性能。
- **逆对数变换**：将标准化的评分转换回原始的亲和力评分，以增强模型的敏感性。

### 代码结构
- `MF.py`：实现ALS矩阵分解的主要逻辑。
- `better_features.py`：增强特征提取，处理类别信息。
- `KMeans_user_cluster.py`：执行用户聚类的逻辑。
- `better_XGB.py`：整合所有功能，进行模型训练和预测。

该项目通过Docker容器化，便于在本地环境中部署和运行。
---

## 项目目标
该项目旨在实现一个**以排名为目标**的混合推荐系统，利用多模态数据（文本和图像）生成嵌入，并使用XGBoost的LambdaMART进行排名优化。系统的主要目标是提升推荐的准确性，特别是在冷启动用户和视觉驱动业务场景中。

## 主要技术栈
- **Python, Spark**：用于数据处理和模型训练
- **XGBoost (LambdaMART)**：用于排名优化
- **Universal Sentence Encoder (USE)**：用于生成文本嵌入
- **CLIP**：用于生成照片嵌入
- **ALS (交替最小二乘法)**：用于隐式矩阵分解

## 系统架构
### 1. Ranking 优化目标
将现有的回归目标切换为排名优化目标，使用 **XGBoost** 的 **LambdaMART** 实现基于排名的推荐系统。
- **评估指标**：使用 **NDCG** 和 **MAP** 作为主要评估指标，替代原有的RMSE（均方根误差）。
- **分组处理**：为每个用户定义一组商家，并确保在训练中为同一用户生成的商家结果排序最优。

#### XGBoost Ranking 代码示例：
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
### 2. 文本嵌入（Universal Sentence Encoder）

通过 **Universal Sentence Encoder (USE)** 提取Yelp商家评论的文本嵌入。将嵌入与用户-商家的其他特征整合，提升冷启动用户的推荐效果。

#### USE 嵌入生成示例：
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

### 3. 照片嵌入（CLIP 模型）

通过 **CLIP** 模型生成Yelp商家的照片嵌入，并将其与文本嵌入整合，创建一个多模态推荐系统。

#### CLIP 嵌入生成与整合示例：
```python
import torch
import clip
from PIL import Image
import numpy as np

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 加载并处理图片
image = preprocess(Image.open("path_to_yelp_image.jpg")).unsqueeze(0).to(device)

# 获取图像嵌入
with torch.no_grad():
    image_embedding = model.encode_image(image).cpu().numpy()

# 整合文本和图片嵌入
combined_embedding = np.concatenate([text_embedding, image_embedding], axis=1)
```

### 4. 多模态特征整合
- **文本嵌入**：使用 **Universal Sentence Encoder** 提取评论文本的语义嵌入。
- **图片嵌入**：使用 **CLIP** 模型为商家生成图片嵌入。
- **多模态整合**：将文本和图片的嵌入特征拼接，并作为XGBoost模型的输入，用于排名优化。

### 5. 模型训练与评估
训练XGBoost的 **LambdaMART** 排名模型，使用 NDCG 和 MAP 作为评估标准。

```python
from sklearn.metrics import ndcg_score

# 预测并计算 NDCG
y_pred = bst.predict(dtest)
ndcg = ndcg_score(y_true, y_pred)

print(f"NDCG Score: {ndcg}")
```

---

## 进一步改进方向
1. **Fine-tuning USE**：根据Yelp的数据细调Universal Sentence Encoder模型，以进一步提升文本嵌入的表现。
2. **多模态优化**：探索图像和文本嵌入之间的关系，例如通过CLIP生成图片-文本相似度，提升推荐的多模态关联性。
3. **模型优化**：利用贝叶斯优化进一步调整模型的超参数，提升LambdaMART模型的排名性能。

---

## 项目目标
- **Ranking优化**：通过XGBoost的LambdaMART模型进行排名优化，提升推荐系统在冷启动和多模态场景下的表现。
- **文本嵌入（USE）**：使用Universal Sentence Encoder生成商家评论嵌入，提升冷启动用户的推荐效果，提升相关性。
- **照片嵌入（CLIP）**：通过CLIP生成商家照片的嵌入，并整合到多模态推荐系统中，视觉密集型推荐场景中的准确性提升。
- **多模态特征整合**：将商家的文本和图片嵌入结合，实现一个更强大的多模态推荐系统，提升用户参与度。

---

## 思考方向
1. **Q1**: 是否需要进一步优化用户和商家之间的个性化匹配逻辑？
2. **Q2**: 是否有其他数据模态（如音频或地理位置）可以进一步丰富推荐系统？
3. **Q3**: 是否有计划对图像分类或图像-文本匹配任务进行细调和增强？

