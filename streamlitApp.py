import streamlit as st
import openai
import os
import asyncio
from langchain import OpenAI, Pinecone, LangChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import PineconeVectorStore

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
environment = "us-east-1"
pinecone.init(api_key=api_key, environment=environment)
index_name = "restaurant-index"

# Ensure the index exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

index = pinecone.Index(index_name)

# Initialize LangChain
embeddings = OpenAIEmbeddings(api_key=openai.api_key)
vector_store = PineconeVectorStore(pinecone_index=index, embedding=embeddings)
chain = LangChain(vector_store)

# Function to generate vector and get recommendations
async def get_recommendations(user_query):
    # Generate vector for user query
    query_vector = await embeddings.embed_text_async(user_query)
    # Query Pinecone for relevant information
    results = vector_store.similarity_search_by_vector(query_vector, top_k=10)

    # Extract relevant information from the results
    recommendations = []
    for res in results:
        metadata = res.metadata
        recommendations.append({
            "Restaurant ID": res.id,
            "Restaurant Name": metadata.get("Restaurant Name", "N/A"),
            "Address": metadata.get("Address", "N/A"),
            "Locality": metadata.get("Locality", "N/A"),
            "Cuisines": metadata.get("Cuisines", "N/A"),
            "Average Cost for two": metadata.get("Average Cost for two", "N/A"),
            "Aggregate rating": metadata.get("Aggregate rating", "N/A"),
            "Votes": metadata.get("Votes", "N/A"),
            "Rating text": metadata.get("Rating text", "N/A")
        })

    # Use GPT-3 to generate a response based on the recommendations
    if recommendations:
        response = await generate_response(user_query, recommendations)
        return response
    else:
        return "No relevant recommendations found."


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
    response = await openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()


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
