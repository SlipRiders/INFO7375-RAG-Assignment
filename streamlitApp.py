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
        dimension=1536,  # Updated to match the vector dimension of the model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=environment
        )
    )

index = pc.Index(index_name)

# Helper function to parse key elements from the user query
def parse_key_elements(query_response):
    # Assuming the response is in the format "location: X, rating: Y"
    elements = query_response.split(',')
    key_elements = {}
    for element in elements:
        key, value = element.split(':')
        key_elements[key.strip()] = value.strip()
    return key_elements

async def process_user_query(user_query):
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Extract key elements from the following query: {user_query}"}
        ]
    )
    query_response = response.choices[0].message.content.strip()
    return parse_key_elements(query_response)

async def generate_vector(text):
    response = await client.embeddings.create(
        input=[text],  # Embedding API expects a list of inputs
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

async def filter_results_by_query(results, key_elements):
    filtered_results = []
    for res in results["matches"]:
        metadata = res["metadata"]
        if 'location' in key_elements and 'rating' in key_elements:
            if key_elements['location'].lower() in metadata['Locality'].lower() and float(metadata['Aggregate Rating']) >= float(key_elements['rating']):
                filtered_results.append(res)
    return filtered_results

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
    return response.choices[0].message.content.strip()

async def check_relevance(user_query, description):
    prompt = (
        f"User query: {user_query}\n"
        f"Recommendation description: {description}\n"
        "Does the recommendation description match the user query? Reply with 'yes' or 'no' and explain why."
    )
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    relevance_response = response.choices[0].message.content.strip().lower()
    relevance = "yes" in relevance_response
    explanation = relevance_response if not relevance else ""
    return relevance, explanation

async def get_recommendations(processed_query, user_query):
    query_vector = await generate_vector(processed_query)
    results = index.query(vector=query_vector, top_k=10, include_metadata=True)
    key_elements = await process_user_query(user_query)  # 提取用户查询中的关键元素

    filtered_results = await filter_results_by_query(results, key_elements)

    recommendations = []
    for res in filtered_results:
        recommendation = {
            "Restaurant ID": res["id"],
            "Restaurant Name": res["metadata"]["Restaurant Name"],
            "Address": res["metadata"]["Address"],
            "Locality": res["metadata"]["Locality"],
            "Cuisines": res["metadata"]["Cuisines"],
            "Average Cost for Two": res["metadata"]["Average Cost for Two"],
            "Aggregate Rating": res["metadata"]["Aggregate Rating"],
            "Votes": res["metadata"]["Votes"],
            "Rating Text": res["metadata"]["Rating Text"]
        }
        description = await generate_natural_language_description(recommendation)
        relevance, explanation = await check_relevance(user_query, description)
        if relevance:
            recommendations.append(recommendation)
            break
        else:
            print(f"Recommendation discarded: {explanation}")

    return recommendations[0] if recommendations else None

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Restaurant Recommendation Chatbot")

user_query = st.text_input("Enter your preferences or needs:")

def get_recommendation_and_description(user_query):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    processed_query = loop.run_until_complete(process_user_query(user_query))
    recommendation = loop.run_until_complete(get_recommendations(processed_query, user_query))

    if recommendation:
        description = loop.run_until_complete(generate_natural_language_description(recommendation))
        return description
    else:
        return "No relevant recommendations found."

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
