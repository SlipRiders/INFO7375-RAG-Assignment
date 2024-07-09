import streamlit as st
import openai
import os
from pinecone import Pinecone, ServerlessSpec

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
api_key = os.getenv("pinecone-api-key")
environment = "us-east-1"
pc = Pinecone(api_key=api_key)

index_name = "restaurant-index"

# Check if the index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=225,  # Make sure this matches your vector dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=environment
        )
    )

index = pc.Index(index_name)


def process_user_query(user_query):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Extract key information from the following query: {user_query}"}
        ],
        max_tokens=50
    )
    return response.choices[0].message.content


def generate_vector(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']


def get_recommendations(processed_query):
    query_vector = generate_vector(processed_query)
    results = index.query(query_vector, top_k=10, include_metadata=True)
    recommendations = [
        {
            "Restaurant ID": res["id"],
            "Restaurant Name": res["metadata"]["Restaurant Name"],
            "Address": res["metadata"]["Address"],
            "Locality": res["metadata"]["Locality"],
            "Cuisines": res["metadata"]["Cuisines"],
            "Average Cost for Two": res["metadata"]["Average Cost for two"],
            "Aggregate Rating": res["metadata"]["Aggregate rating"],
            "Votes": res["metadata"]["Votes"],
            "Rating Text": res["metadata"]["Rating text"]
        }
        for res in results["matches"]
    ]
    return recommendations


def generate_natural_language_description(recommendation):
    description_prompt = (
        f"Provide a detailed and natural language description for the following restaurant recommendation:\n"
        f"Name: {recommendation['Restaurant Name']}\n"
        f"Address: {recommendation['Address']}\n"
        f"Locality: {recommendation['Locality']}\n"
        f"Cuisines: {recommendation['Cuisines']}\n"
        f"Average Cost for Two: {recommendation['Average Cost for Two']}\n"
        f"Aggregate Rating: {recommendation['Aggregate Rating']}\n"
        f"Votes: {recommendation['Votes']}\n"
        f"Rating Text: {recommendation['Rating Text']}\n"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": description_prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message["content"].strip()


# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Restaurant Recommendation Chatbot")

user_query = st.text_input("Enter your preferences or needs:")

if st.button("Get Recommendations"):
    if user_query:
        processed_query = process_user_query(user_query)
        recommendations = get_recommendations(processed_query)

        if recommendations:
            for rec in recommendations:
                description = generate_natural_language_description(rec)
                st.session_state.history.append({"user": user_query, "bot": description})
        else:
            st.session_state.history.append({"user": user_query, "bot": "No recommendations found."})
    else:
        st.write("Please enter a query.")

# Display conversation history
for entry in st.session_state.history:
    st.write(f"**User:** {entry['user']}")
    st.write(f"**Bot:** {entry['bot']}")
    st.write("---")

# Button to clear conversation history
if st.button("Clear History"):
    st.session_state.history = []
