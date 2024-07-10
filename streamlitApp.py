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
api_key = os.getenv("PINECONE_API_KEY")
environment = "us-east-1"
pc = Pinecone(api_key=api_key)
index_name = "restaurant-index"

# Check if the index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Updated to match the vector dimension of the model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=environment
        )
    )

index = pc.Index(index_name)

async def generate_vector(text):
    response = await client.embeddings.create(
        input=[text],  # Embedding API expects a list of inputs
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

async def select_best_recommendation(user_query, recommendations):
    prompt = (
        f"User query: {user_query}\n"
        f"Here are several restaurant recommendations. Please select the one that best matches the user's query and provide a detailed description:\n"
    )
    for i, recommendation in enumerate(recommendations):
        prompt += (
            f"\nRecommendation {i+1}:\n"
            f"Name: {recommendation['Restaurant Name']}\n"
            f"Address: {recommendation['Address']}\n"
            f"Locality: {recommendation['Locality']}\n"
            f"Cuisines: {recommendation['Cuisines']}\n"
            f"Average Cost for two: {recommendation['Average Cost for Two']}\n"
            f"Aggregate rating: {recommendation['Aggregate Rating']}\n"
            f"Votes: {recommendation['Votes']}\n"
            f"Rating text: {recommendation['Rating Text']}\n"
        )
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

async def get_recommendations(user_query):
    query_vector = await generate_vector(user_query)
    results = index.query(vector=query_vector, top_k=10, include_metadata=True)

    recommendations = []
    for res in results["matches"]:
        recommendations.append({
            "Restaurant ID": res["id"],
            "Restaurant Name": res["metadata"]["Restaurant Name"],
            "Address": res["metadata"]["Address"],
            "Locality": res["metadata"]["Locality"],
            "Cuisines": res["metadata"]["Cuisines"],
            "Average Cost for Two": res["metadata"]["Average Cost for Two"],
            "Aggregate Rating": res["metadata"]["Aggregate Rating"],
            "Votes": res["metadata"]["Votes"],
            "Rating Text": res["metadata"]["Rating Text"]
        })

    if recommendations:
        best_recommendation = await select_best_recommendation(user_query, recommendations)
        return best_recommendation
    else:
        return "No relevant recommendations found."

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Restaurant Recommendation Chatbot")

user_query = st.text_input("Enter your preferences or needs:")

def get_recommendation_and_description(user_query):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    recommendation = loop.run_until_complete(get_recommendations(user_query))
    return recommendation

if st.button("Get Recommendations"):
    if user_query:
        description = get_recommendation_and_description(user_query)
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
