import os
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, CSVLoader, UnstructuredExcelLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_to_atlas(folder_path):
    # 1. Define loaders (Keep exactly as you had them)
    loader_mapping = {
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".txt": TextLoader,
    }

    documents = []
    print(f"🔍 Scanning folder: {folder_path}")
    
    for ext, loader_cls in loader_mapping.items():
        loader = DirectoryLoader(
            folder_path, 
            glob=f"**/*{ext}", 
            loader_cls=loader_cls,
            show_progress=True
        )
        try:
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            print(f"✅ Loaded {len(loaded_docs)} documents with extension {ext}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load {ext} files. Error: {e}")

    if not documents:
        print("❌ No documents found.")
        return

    # 2. Smart Splitting (Keep your logic)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(final_chunks)} chunks.")

    # 3. MongoDB Atlas Integration
    print("🚀 Connecting to MongoDB Atlas...")
    
    # Replace with your actual Connection String
    MONGO_URI = "mongodb+srv://suyash:<db_password>@cluster0.6avycwa.mongodb.net/?appName=Cluster0"
    client = MongoClient(MONGO_URI)
    collection = client["placement_rag"]["knowledge_base"]

    # Use your HuggingFace model (Dimensions = 384)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Push to Atlas
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=final_chunks, 
        embedding=embeddings, 
        collection=collection,
        index_name="vector_index" # This must match the index you create in Atlas UI
    )
    
    print(f"✨ SUCCESS: Data pushed to MongoDB Atlas!")

if __name__ == "__main__":
    DATA_PATH = r"C:\placement-rag\data" 
    ingest_to_atlas(DATA_PATH)
    szAYDae6gKhtRsNF