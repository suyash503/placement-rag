import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# MODERN IMPORTS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup & Safety Check
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    print(" ERROR: GROQ_API_KEY not found in .env file!")
    exit()

print("Initializing AI Engine...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to Vector DB
if not os.path.exists("./chroma_db"):
    print(" ERROR: Database not found! Please run ingest.py first.")
    exit()

vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. Connect to Groq (Llama-3.3)
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 4. Professional RAG Prompt
system_prompt = (
    "You are a professional Placement Assistant. "
    "Use the provided context to answer the user's question accurately. "
    "If the answer isn't in the context, say: 'I don't have that specific record in my database.'\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Build the Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 5}), question_answer_chain)

def ask_question(question):
    print(f"\n Searching records...")
    
    response = qa_chain.invoke({"input": question}) 
    
    print(f"\n AI Answer: {response['answer']}")
    
    print("\n Sources Cited:")
    sources = set()
    for doc in response["context"]:
        # Extract filename from metadata path
        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
        sources.add(source_file)
    
    for s in sources:
        print(f"   • {s}")
    print("-" * 30)

if __name__ == "__main__":
    print(" Placement RAG System Online! (Type 'exit' to quit)")
    while True:
        user_input = input("\nAsk about placements: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input.strip():
            try:
                ask_question(user_input)
            except Exception as e:
                print(f" An error occurred: {e}")