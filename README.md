# 💊 PharmacologyGPT

> **An Enterprise-Style Hybrid Retrieval-Augmented Generation (Hybrid RAG) Assistant for Pharmacology and Drug Knowledge**

PharmacologyGPT is a production-oriented Hybrid RAG application that answers pharmacology-related questions using trusted textbook PDFs. It combines **semantic vector search (ChromaDB)** and **keyword search (BM25)** with **Reciprocal Rank Fusion (RRF)** to improve retrieval accuracy before generating grounded responses with **OpenAI GPT-5 Nano**.

---

# 🚀 Features

## ✅ Current Features (v1.1)

- 📚 Multiple PDF knowledge base
- 📄 Dynamic PDF upload
- ✂️ Automatic document chunking
- 🧠 OpenAI Embeddings (`text-embedding-3-small`)
- 🗄️ Persistent Chroma Vector Database
- 🔍 Vector Search using Maximum Marginal Relevance (MMR)
- 🔎 BM25 Keyword Retrieval
- 🔀 Hybrid Retrieval (Vector + BM25)
- ⭐ Reciprocal Rank Fusion (RRF)
- 🚫 Duplicate PDF Detection (SHA-256)
- 🤖 GPT-5 Nano Answer Generation
- 📖 Source Citation (PDF + Page Number)
- 💻 Streamlit User Interface

---

# 🏗️ System Architecture

```text
                          User
                            │
                            ▼
                    Streamlit Web App
                            │
                            ▼
                    User Question
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
        Chroma Vector Search      BM25 Keyword Search
            (MMR Retriever)         (Exact Matching)
                │                       │
                └───────────┬───────────┘
                            ▼
             Reciprocal Rank Fusion (RRF)
                            │
                  Top Relevant Chunks
                            │
                            ▼
                  GPT-5 Nano (OpenAI)
                            │
                            ▼
               Grounded Answer + Citations
```

---

# 📂 Project Structure

```text
PharmacologyGPT-RAG/
│
├── docs/                          # Pharmacology PDFs
│
├── vector/                        # Persistent Chroma database
│
├── test/
│   ├── __init__.py
│   ├── test_bm25_retrieval.py
│   └── test_hybrid_retrieval.py
│
├── app.py                         # Streamlit application
├── rag.py                         # Production Hybrid RAG pipeline
├── hybrid_retriever.py            # BM25 + Vector + RRF
├── build_vector_db.py             # Vector database builder
├── pdf_utils.py                   # PDF processing utilities
├── inspect_vector_db.py           # Database inspection utility
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# ⚙️ Technologies Used

## Programming Language

- Python 3.13

## LLM

- OpenAI GPT-5 Nano

## Embeddings

- text-embedding-3-small

## Vector Database

- ChromaDB

## Retrieval

- Vector Search (MMR)
- BM25
- Reciprocal Rank Fusion (RRF)

## Frameworks

- LangChain
- Streamlit

## Document Processing

- PyPDF
- Recursive Character Text Splitter

---

# 📦 Installation

## 1. Clone Repository

```bash
git clone https://github.com/marufasumi/PharmacologyGPT-RAG.git

cd PharmacologyGPT-RAG
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate

### macOS/Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment

Create a `.env` file.

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## 5. Build Vector Database

```bash
python build_vector_db.py
```

---

## 6. Launch Application

```bash
streamlit run app.py
```

---

# 💬 Example Questions

- What is the mechanism of action of metformin?
- Why do ACE inhibitors cause cough?
- How does succinylcholine produce neuromuscular blockade?
- What are the adverse effects of digoxin?
- What drugs inhibit CYP3A4?
- Explain beta-2 agonists.

---

# 🔍 Retrieval Pipeline

## Step 1

User submits a question.

↓

## Step 2

Vector Retriever (Chroma MMR)

↓

## Step 3

BM25 Keyword Retriever

↓

## Step 4

Reciprocal Rank Fusion

↓

## Step 5

Top 5 unique chunks

↓

## Step 6

GPT-5 Nano generates a grounded response.

↓

## Step 7

Display answer with source citations.

---

# 📈 Current Version

## ✅ v1.1.0

Implemented:

- Hybrid Retrieval
- BM25
- Reciprocal Rank Fusion
- Hybrid Retriever Module
- End-to-End Testing
- Streamlit Integration

---

# 🛣️ Future Roadmap

## v1.2

- Web Search Integration
- Local + Web Context Fusion
- Intelligent Question Routing

---

## v1.3

- RAG Evaluation
- RAGAS
- DeepEval
- Retrieval Metrics

---

## v1.4

- LangGraph Agent Workflow
- Multi-step Reasoning
- Tool Calling

---

## v1.5

- Multi-turn Conversation Memory
- Session Context
- Chat History

---

## v1.6

- FastAPI Backend
- REST API
- Swagger Documentation

---

## v1.7

- Docker
- Cloud Deployment
- CI/CD
- Production Monitoring

---

# 📊 Project Highlights

✔ Enterprise-style Hybrid RAG

✔ Production-oriented Architecture

✔ Modular Python Codebase

✔ Persistent Vector Database

✔ Hybrid Search

✔ Explainable Retrieval

✔ Source Citations

✔ Retrieval Evaluation Ready

---

# 👨‍💻 Author

**Marufa Sultana Sumi**

MS in Information, University of Michigan

Healthcare Data Science | AI | Data Engineering | Generative AI

GitHub:
https://github.com/marufasumi

LinkedIn:
https://www.linkedin.com/in/marufasumi/

---

# ⭐ Acknowledgements

This project was built as part of my learning journey in:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Information Retrieval
- Generative AI
- Production AI System Design

The architecture is designed to evolve into an enterprise-grade AI assistant capable of combining local knowledge bases, web search, evaluation frameworks, agent workflows, and cloud deployment.
