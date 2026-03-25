import certifi
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup Connection (Use your LONG connection string here)
MONGO_URI = "mongodb+srv://suyashsingh2711_db_user:KeS5yYYNpztOWwRX@placementcluster.sicsdzv.mongodb.net/?appName=PlacementCluster"
DB_NAME = "placement_rag"
COLLECTION_NAME = "knowledge_base"

# 2. Connect to the Brain
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
collection = client[DB_NAME][COLLECTION_NAME]

# 3. Load the same Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Initialize Vector Store
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index"
)

# 5. TEST QUERY
query = "Tell me about companies hiring for software roles"
print(f"🔍 Searching for: {query}")

# Find top 3 matches
results = vector_store.similarity_search(query, k=3)

print("\n✨ TOP MATCHES FOUND:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)