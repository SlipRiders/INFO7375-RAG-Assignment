import streamlit as st
import openai
import os
import asyncio
import pinecone
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec


# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=openai.api_key)

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
environment = "us-east-1"
pc = Pinecone(api_key=api_key)
index_name = "restaurant-index"

# Ensure the index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region=environment
        )
    )

index = pc.Index(index_name)

# Function to generate vector and get recommendations
async def get_recommendations(user_query):
    # Generate vector for user query
    response = await client.embeddings.create(input=[user_query], model="text-embedding-3-small")
    query_vector = response.data[0].embedding

    # Query Pinecone for relevant information
    results = index.query(vector=query_vector, top_k=10, include_metadata=True)

    # Extract relevant information from the results
    recommendations = []
    for res in results["matches"]:
        metadata = res['metadata']
        recommendations.append({
            "Restaurant ID": res["id"],
            "Restaurant Name": metadata.get("Restaurant Name", "N/A"),
            "Address": metadata.get("Address", "N/A"),
            "Locality": metadata.get("Locality", "N/A"),
            "Cuisines": metadata.get("Cuisines", "N/A"),
            "Average Cost for two": metadata.get("Average Cost for two", "N/A"),
            "Aggregate rating": metadata.get("Aggregate rating", "N/A"),
            "Votes": metadata.get("Votes", "N/A"),
            "Rating text": metadata.get("Rating text", "N/A"),
        })

    # Use GPT-3 to generate a response based on the recommendations
    if recommendations:
        response = await generate_response(user_query, recommendations)
        return response
    else:
        return "No relevant recommendations found."

# Function to generate response using GPT-3
async def generate_response(user_query, recommendations):
    prompt = (
        f"User query: {user_query}\n"
        f"Based on the user query, here are several restaurant recommendations. Please provide a detailed description of the best match:\n"
    )
    for i, recommendation in enumerate(recommendations):
        prompt += (
            f"\nRecommendation {i + 1}:\n"
            f"Name: {recommendation['Restaurant Name']}\n"
            f"Address: {recommendation['Address']}\n"
            f"Locality: {recommendation['Locality']}\n"
            f"Cuisines: {recommendation['Cuisines']}\n"
            f"Average Cost for two: {recommendation['Average Cost for two']}\n"
            f"Aggregate rating: {recommendation['Aggregate rating']}\n"
            f"Votes: {recommendation['Votes']}\n"
            f"Rating text: {recommendation['Rating text']}\n"
        )
    response = await client.chat.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content'].strip()

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Restaurant Recommendation Chatbot")

user_query = st.text_input("Enter your preferences or needs:")

async def get_recommendation_and_description(user_query):
    recommendation = await get_recommendations(user_query)
    return recommendation

if st.button("Get Recommendations"):
    if user_query:
        description = asyncio.run(get_recommendation_and_description(user_query))
        st.session_state.history.append({"user": user_query, "bot": description})
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
