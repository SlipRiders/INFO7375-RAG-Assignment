# INFO7375-RAG-Assignment
# Restaurant Recommendation Chatbot

This project is a Restaurant Recommendation Chatbot that utilizes local data to provide personalized restaurant recommendations based on user queries. The chatbot is built using Streamlit for the frontend, OpenAI for natural language processing, and Pinecone for the vector database.

## Features

- **Natural Language Understanding**: Uses OpenAI's GPT-3.5-turbo model to understand user queries and generate responses.
- **Vector Database**: Utilizes Pinecone to store and query restaurant data vectors.
- **Asynchronous Processing**: Handles requests and responses asynchronously for better performance.
- **Streamlit Interface**: Provides an easy-to-use web interface for user interaction.

## Video Demonstration
link: [Youtube](https://youtu.be/94NzZf6JlV8) 

## Project Structure

- `main.py`: The main script that runs the Streamlit application and handles user interactions.
- `data_cleaning.py`: Script for cleaning the restaurant data.
- `vector_database.py`: Script for building and populating the Pinecone vector database.
- `requirements.txt`: A file listing all the required Python packages for the project.
## Dataset Description

### About the Dataset
The dataset contains information about 10,000 restaurants in India, with 80% of the entries being from New Delhi. Thus, this chatbot primarily recommends restaurants in New Delhi.

### Dataset Source
The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/mohdshahnawazaadil/restaurant-dataset/data).

### Dataset Content
This dataset is designed for developing a machine learning model to classify restaurants based on their cuisines. It includes various attributes related to restaurants such as location, average cost, ratings, and services offered. The primary objective is to predict the cuisine type of a restaurant using these attributes.

### Attributes
- **Restaurant ID**: Unique identifier for each restaurant.
- **Restaurant Name**: Name of the restaurant.
- **Country Code**: Country code where the restaurant is located.
- **City**: City where the restaurant is situated.
- **Address**: Address of the restaurant.
- **Locality**: General locality of the restaurant.
- **Locality Verbose**: Detailed locality description.
- **Longitude**: Longitude coordinate of the restaurant's location.
- **Latitude**: Latitude coordinate of the restaurant's location.
- **Cuisines**: Type of cuisines offered by the restaurant (target variable).
- **Average Cost for Two**: Average cost for two people dining at the restaurant.
- **Currency**: Currency used for pricing.
- **Has Table Booking**: Binary variable indicating if the restaurant accepts table bookings.
- **Has Online Delivery**: Binary variable indicating if the restaurant offers online delivery.
- **Is Delivering Now**: Binary variable indicating if the restaurant is currently delivering.
- **Switch to Order Menu**: Binary variable indicating if the restaurant has an online menu ordering option.
- **Price Range**: Range indicating the price level of the restaurant's menu items.
- **Aggregate Rating**: Average rating of the restaurant based on customer reviews.
- **Rating Color**: Color code representing the rating level.
- **Rating Text**: Textual representation of the rating level.
- **Votes**: Total number of votes received by the restaurant.

### Acknowledgements
This dataset was collected from various sources and curated for research purposes. Gratitude is extended to all contributors who made this dataset available for analysis and experimentation.

### Usage Policy
The dataset is intended for research and educational purposes only. Commercial use or redistribution of the dataset requires explicit permission from the data owners.

### Disclaimer
While efforts were made to ensure the accuracy and reliability of the dataset, the creators do not guarantee its correctness or suitability for any particular purpose. Users are advised to validate the data before making any decisions based on it.



## Getting Started

### Prerequisites

- Python 3.7 or higher
- OpenAI API key
- Pinecone API key

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SlipRiders/INFO7375-RAG-Assignment.git
    cd restaurant-recommendation-chatbot
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your API keys:
    - Create a `.env` file in the project root directory.
    - Add your OpenAI and Pinecone API keys to the `.env` file:
      ```bash
      OPENAI_API_KEY=your_openai_api_key
      PINECONE_API_KEY=your_pinecone_api_key
      ```

### Data Cleaning

Before building the vector database, the restaurant data needs to be cleaned to remove any garbled characters.

```python
import pandas as pd
import numpy as np
import re

# Attempt to read the file with UTF-8-SIG encoding
data = pd.read_csv('restaurant_data.csv', encoding='UTF-8-SIG')
pd.set_option('display.max_columns', None)  # Display all columns

# Define a function to check for garbled characters
def has_garbled_chars(text):
    if isinstance(text, str):
        # Check for non-ASCII characters
        return bool(re.search(r'[^\x00-\x7F]+', text))
    return False

# Remove rows with garbled characters
def remove_garbled_rows(df):
    for column in df.columns:
        df = df[~df[column].apply(has_garbled_chars)]
    return df

# Apply the function to remove rows with garbled characters
cleaned_data = remove_garbled_rows(data)

# Print the first 50 rows to check the cleaned data
print(cleaned_data.head(50))

# Save the cleaned data to a new CSV file
cleaned_data.to_csv('cleaned_restaurant_data.csv', index=False, encoding='UTF-8-SIG')
```

### Building the Vector Database
Once the data is cleaned, we can build the vector database using Pinecone.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from pinecone import Pinecone, ServerlessSpec

# Read the cleaned data
data = pd.read_csv('cleaned_restaurant_data.csv', encoding='UTF-8-SIG')

# Check for missing values
print(data.columns)  # Output column names
data.columns = data.columns.str.strip()  # Remove leading and trailing spaces from column names

# Combine text data columns
text_columns = ['Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Cuisines', 'Rating text']
data['combined_text'] = data[text_columns].astype(str).agg(' '.join, axis=1)

# Increase the number of features for the TF-IDF vector
vectorizer = TfidfVectorizer(max_features=1361)
tfidf_matrix = vectorizer.fit_transform(data['combined_text'])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Categorical data columns
categorical_columns = ['Country Code', 'City', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Rating color']
one_hot_encoder = OneHotEncoder()
one_hot_encoded = one_hot_encoder.fit_transform(data[categorical_columns]).toarray()
print("One-hot encoded shape:", one_hot_encoded.shape)

# Numerical data columns
numeric_columns = ['Longitude', 'Latitude', 'Average Cost for two', 'Price range', 'Aggregate rating', 'Votes']
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(data[numeric_columns])
print("Scaled numeric shape:", scaled_numeric.shape)

# Combine all features
combined_features = np.hstack((tfidf_matrix.toarray(), one_hot_encoded, scaled_numeric))
print("Combined features shape:", combined_features.shape)

# Print the vector dimension
vector_dimension = combined_features.shape[1]
print(f"Vector dimension: {vector_dimension}")

# Initialize Pinecone
api_key = 'your-pinecone-api-key'  # Replace with your Pinecone API key
pc = Pinecone(api_key=api_key)

# Delete existing index (if any)
index_name = 'restaurant-index'
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# Create a new index
pc.create_index(
    name=index_name,
    dimension=vector_dimension,  # Use the actual vector dimension
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

# Connect to the index
index = pc.Index(index_name)

# Batch insert vector data into the Pinecone index with metadata
batch_size = 100  # Insert 100 records at a time
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

print("Vector database successfully built and populated!")
```
### Running the Application
To start the Streamlit application, run the following command in the project root directory:
```bash
streamlit run main.py
```

### How It Works
User Query Input: The user enters their preferences or needs into the text input box.      
Generate Vector: The chatbot generates a vector for the user query using OpenAI's text embedding model.     
Query Pinecone: The generated vector is used to query Pinecone for the most relevant restaurant data.     
Filter Results: The results are filtered to include only restaurants with sufficient ratings and votes.     
Generate Response: A detailed response is generated using GPT-3 based on the filtered recommendations.     
Display Recommendations: The recommendations are displayed to the user in the Streamlit interface.     
     
### Example
Below is an example of how the chatbot interaction might look:

User: "I am looking for an Italian restaurant in downtown with a cozy atmosphere."
Bot: "Based on your query, here are several restaurant recommendations..."

### License
This project is licensed under the MIT License. See the LICENSE file for details.
