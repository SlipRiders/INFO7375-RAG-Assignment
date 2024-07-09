import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from pinecone import Pinecone, ServerlessSpec

# read cleaned data
data = pd.read_csv('cleaned_restaurant_data.csv', encoding='UTF-8-SIG')

# check for missing values
print(data.columns)  # output the column names
data.columns = data.columns.str.strip()  # remove leading/trailing whitespaces from column names


text_columns = ['Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Cuisines', 'Rating text']
data['combined_text'] = data[text_columns].astype(str).agg(' '.join, axis=1)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=200)
tfidf_matrix = vectorizer.fit_transform(data['combined_text'])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# categorical columns
categorical_columns = ['Country Code', 'City', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Rating color']
one_hot_encoder = OneHotEncoder()
one_hot_encoded = one_hot_encoder.fit_transform(data[categorical_columns]).toarray()
print("One-hot encoded shape:", one_hot_encoded.shape)

# numeric columns
numeric_columns = ['Longitude', 'Latitude', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(data[numeric_columns])
print("Scaled numeric shape:", scaled_numeric.shape)

# combine all features
combined_features = np.hstack((tfidf_matrix.toarray(), one_hot_encoded, scaled_numeric))
print("Combined features shape:", combined_features.shape)

# Initialize Pinecone
api_key = 'pinecone-api-key'
pc = Pinecone(api_key=api_key)

# delete existing index if it exists
index_name = 'restaurant-index'
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# create a new index
pc.create_index(
    name=index_name,
    dimension=combined_features.shape[1],
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# connect to the index
index = pc.Index(index_name)

# insert data into the index
batch_size = 100  # batch size for inserting vectors
for i in range(0, len(combined_features), batch_size):
    batch_vectors = [(str(data.iloc[j]['Restaurant ID']), combined_features[j]) for j in range(i, min(i + batch_size, len(combined_features)))]
    index.upsert(batch_vectors)

print("Data inserted successfully!")
