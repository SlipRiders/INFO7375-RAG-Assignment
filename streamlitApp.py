import streamlit as st
import openai
import os
from pinecone import Pinecone, ServerlessSpec
import asyncio
from openai import AsyncOpenAI

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=openai.api_key)
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


async def process_user_query(user_query):
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Extract key information from the following query: {user_query}"}
        ]
    )
    return response.choices[0].message.content


async def generate_vector(text):
    response = await client.embeddings.create(
        input=[text],  # Embedding API expects a list of inputs
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


async def get_recommendations(processed_query):
    query_vector = await generate_vector(processed_query)
    results = index.query(vector=query_vector, top_k=10, include_metadata=True)
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


async def generate_natural_language_description(recommendation):
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
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": description_prompt}
        ]
    )
    return response.choices[0].message.content


# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Restaurant Recommendation Chatbot")

user_query = st.text_input("Enter your preferences or needs:")

if st.button("Get Recommendations"):
    if user_query:
        processed_query = asyncio.run(process_user_query(user_query))
        recommendations = asyncio.run(get_recommendations(processed_query))

        if recommendations:
            for rec in recommendations:
                description = asyncio.run(generate_natural_language_description(rec))
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