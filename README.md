# 🎓CollegePlacement RAG Assistant

An AI-powered conversational agent designed to help students navigate complex placement data. Instead of manually searching through thousands of spreadsheet rows, students can query eligibility, packages, and company details using natural language.

---

## 🚀 Features
- **Semantic Search:** Understands user intent (e.g., "Web Dev roles") beyond simple keyword matching.
- **Eligibility Filtering:** Instantly filters companies based on CGPA, branch, and 10th/12th percentages.
- **Real-time Insights:** Provides summaries of selection processes and recruitment timelines.
- **Natural Language Interface:** Powered by Gemini 2.5 Flash for human-like interaction.

## 🛠️ Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/) (Python-based Web UI)
- **Orchestration:** [LangChain](https://www.langchain.com/) (RAG Pipeline)
- **Vector Database:** [MongoDB Atlas Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search)
- **LLM:** [Google Gemini 2.5 Flash](https://aistudio.google.com/)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (384 Dimensions)

## 🏗️ Architecture
The system follows a standard **RAG (Retrieval-Augmented Generation)** pipeline:
1. **Ingestion:** 3,000+ placement records are chunked, embedded, and stored in MongoDB Atlas.
2. **Retrieval:** User queries are converted to vectors; Cosine Similarity is used to find the most relevant data chunks.
3. **Generation:** The retrieved context is passed to Gemini 2.5 to generate a structured, professional response.



## 💻 Local Setup

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/suyash503/placement-rag.git](https://github.com/suyash503/placement-rag.git)
   cd placement-rag

2. Install dependencies:
   uv pip install -r requirements.txt

3. Create a .env file with your credentials:
   
   MONGO_URI=your_mongodb_connection_string
   GEMINI_API_KEY=your_google_api_key

4. Run the App:

   streamlit run app.py

👨‍💻 Author
Suyash 3rd Year Coordinator, Training & Placement Cell Ajay Kumar Garg Engineering College (AKGEC)

Disclaimer: This tool is intended for informational purposes. Please verify final details on the official college placement portal.
