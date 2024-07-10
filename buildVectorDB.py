import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from pinecone import Pinecone, ServerlessSpec

# 读取清洗后的数据
data = pd.read_csv('cleaned_restaurant_data.csv', encoding='UTF-8-SIG')

# 检查缺失值
print(data.columns)  # 输出列名
data.columns = data.columns.str.strip()  # 去除列名的首尾空格

# 合并文本数据列
text_columns = ['Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Cuisines', 'Rating text']
data['combined_text'] = data[text_columns].astype(str).agg(' '.join, axis=1)

# 增加 TF-IDF 向量的特征数量
vectorizer = TfidfVectorizer(max_features=1361)
tfidf_matrix = vectorizer.fit_transform(data['combined_text'])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# 类别数据列
categorical_columns = ['Country Code', 'City', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Rating color']
one_hot_encoder = OneHotEncoder()
one_hot_encoded = one_hot_encoder.fit_transform(data[categorical_columns]).toarray()
print("One-hot encoded shape:", one_hot_encoded.shape)

# 数值数据列
numeric_columns = ['Longitude', 'Latitude', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(data[numeric_columns])
print("Scaled numeric shape:", scaled_numeric.shape)

# 组合所有特征
combined_features = np.hstack((tfidf_matrix.toarray(), one_hot_encoded, scaled_numeric))
print("Combined features shape:", combined_features.shape)

# 打印向量的维度
vector_dimension = combined_features.shape[1]
print(f"Vector dimension: {vector_dimension}")

# 初始化 Pinecone
api_key = 'your-pinecone-api-key'  # 替换为你的 Pinecone API 密钥
pc = Pinecone(api_key=api_key)

# 删除现有索引（如果存在）
index_name = 'restaurant-index'
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# 创建新索引
pc.create_index(
    name=index_name,
    dimension=vector_dimension,  # 使用实际的向量维度
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# 连接到索引
index = pc.Index(index_name)

# 批量插入向量数据到 Pinecone 索引中，包含元数据
batch_size = 100  # 每次插入100条数据
for i in range(0, len(combined_features), batch_size):
    batch_vectors = [
        (
            str(data.iloc[j]['Restaurant ID']),
            combined_features[j],
            {
                "Restaurant Name": str(data.iloc[j]['Restaurant Name']),
                "Address": str(data.iloc[j]['Address']),
                "Locality": str(data.iloc[j]['Locality']),
                "Locality Verbose": str(data.iloc[j]['Locality Verbose']),
                "Cuisines": str(data.iloc[j]['Cuisines']),
                "Average Cost for two": str(data.iloc[j]['Average Cost for two']),
                "Currency": str(data.iloc[j]['Currency']),
                "Has Table booking": str(data.iloc[j]['Has Table booking']),
                "Has Online delivery": str(data.iloc[j]['Has Online delivery']),
                "Is delivering now": str(data.iloc[j]['Is delivering now']),
                "Switch to order menu": str(data.iloc[j]['Switch to order menu']),
                "Price range": str(data.iloc[j]['Price range']),
                "Aggregate rating": str(data.iloc[j]['Aggregate rating']),
                "Rating color": str(data.iloc[j]['Rating color']),
                "Rating text": str(data.iloc[j]['Rating text']),
                "Votes": str(data.iloc[j]['Votes'])
            }
        )
        for j in range(i, min(i + batch_size, len(combined_features)))
    ]
    index.upsert(batch_vectors)

print("向量数据库已成功建立并插入数据！")
