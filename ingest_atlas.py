import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, CSVLoader, UnstructuredExcelLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
# 1. Setup Atlas Connection
DB_NAME = "placement_rag"
COLLECTION_NAME = "knowledge_base"

import ssl
import certifi

# ... inside your ingest_to_mongodb function ...

print(" Connecting to MongoDB Atlas...")

client = MongoClient(
    MONGO_URI,
    tls=True,                                
    tlsCAFile=certifi.where(),               
    tlsAllowInvalidCertificates=True,        
    serverSelectionTimeoutMS=30000           
)

def ingest_to_mongodb(data):
    # Define which loader to use for each file type
    loader_mapping = {
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".txt": TextLoader,
    }

    documents = []
    print(f" Scanning folder: {data}")
    
    # --- THE MISSING LOOP: This actually loads your files ---
    for ext, loader_cls in loader_mapping.items():
        loader = DirectoryLoader(
            data, 
            glob=f"**/*{ext}", 
            loader_cls=loader_cls,
            show_progress=True
        )
        try:
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            print(f" Loaded {len(loaded_docs)} documents with extension {ext}")
        except Exception as e:
            print(f" Warning: Could not load {ext} files. Error: {e}")

    if not documents:
        print(" ERROR: No documents were found. Check if your /data folder has files!")
        return

    # 2. Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(docs)} chunks.")

    # 3. Use HuggingFace Embeddings (Dimension: 384)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Connect and Push to Atlas
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    print(f" Pushing {len(docs)} chunks to MongoDB Atlas...")
    
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        collection=collection,
        index_name="vector_index" 
    )
    
    print(" SUCCESS: Your placement data is now live in the cloud!")

# This part actually starts the script
if __name__ == "__main__":
    # Point this to your data folder
    DATA_PATH = r"C:\placement-rag\data" 
    ingest_to_mongodb(DATA_PATH)

