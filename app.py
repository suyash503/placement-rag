import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from pymongo import MongoClient
import certifi

# 1. LOAD ENVIRONMENT VARIABLES
load_dotenv() 
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. AI SETUP
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file!")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# 3. MONGODB CONNECTION
@st.cache_resource # Keeps the connection alive and fast
def init_rag():
    if not MONGO_URI:
        st.error("❌ MONGO_URI not found in .env file!")
        st.stop()
        
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db_name = "placement_rag"
    collection_name = "knowledge_base"
    collection = client[db_name][collection_name]
    
    # Must match the model used in Colab during ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index"
    )

vector_store = init_rag()

# 4. STREAMLIT UI
st.set_page_config(page_title="AKGEC T&P Assistant", page_icon="🎓")
st.title("🎓 AKGEC Placement AI")
st.markdown("Your 24/7 assistant for company criteria, packages, and eligibility.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ex: Show me companies with CTC > 10 LPA"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- THE RAG LOGIC ---
    with st.spinner("Searching placement records..."):
        # 1. Retrieve the top 4 most relevant matches
        docs = vector_store.similarity_search(prompt, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 2. Build the System Prompt
        system_instructions = f"""
        You are the official AKGEC T&P Assistant. 
        Use the following data extracted from the placement records to answer the student's question.
        If the data doesn't contain the answer, politely say you don't have that info.
        Keep your response professional, structured, and helpful.

        RELEVANT DATA:
        {context}
        """

        # 3. Generate response using Gemini
        try:
            response = model.generate_content(system_instructions + "\n\nStudent Question: " + prompt)
            full_response = response.text
        except Exception as e:
            full_response = f"⚠️ Error generating response: {str(e)}"

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})