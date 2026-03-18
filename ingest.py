import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest_all_docs(folder_path):
    # 1. Define loaders for different extensions
    loader_mapping = {
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".txt": TextLoader,
    }

    documents = []
    
    print(f" Scanning folder: {folder_path}")
    
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
            print(f" Loaded {len(loaded_docs)} documents with extension {ext}")
        except Exception as e:
            print(f" Warning: Could not load files with extension {ext}. Error: {e}")

    if not documents:
        print(" No documents found. Please check your folder path.")
        return

    # 2. Smart Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    final_chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split documents into {len(final_chunks)} chunks.")

    # 3. Store in Vector DB (Chroma)
    print("Generating embeddings and updating ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # We use .from_documents to create/overwrite the DB
    vector_db = Chroma.from_documents(
        documents=final_chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print(f"🚀 SUCCESS: System updated with {len(final_chunks)} searchable units!")

if __name__ == "__main__":
    DATA_PATH = r"C:\placement-rag\data" 
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created folder at {DATA_PATH}. Put your files there and run again!")
    else:
        ingest_all_docs(DATA_PATH)