# 💊 PharmacologyGPT

> Enterprise-grade Hybrid RAG (Retrieval-Augmented Generation) assistant for pharmacology, drug information, FDA safety updates, and clinical knowledge.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM-black)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Overview

PharmacologyGPT is a production-oriented Hybrid Retrieval-Augmented Generation (RAG) application that combines textbook knowledge with real-time web search to answer pharmacology-related questions.

The system intelligently routes user queries to:
- 📚 Local knowledge base (pharmacology textbooks)
- 🌐 Live web search
- 🔀 Hybrid retrieval (local + web)

---

## ✨ Features

- Hybrid Retrieval (Vector + BM25)
- Intelligent Query Router
- Query Intent Detection
- Query Rewriting
- Tavily Web Search
- Context Fusion
- OpenAI GPT Response Generation
- Chroma Persistent Vector Database
- User PDF Upload
- Lazy Initialization
- Modular LangChain Architecture
- Streamlit Web Interface

---

## 🏗 System Architecture

```text
                 User
                   │
                   ▼
              Streamlit UI
                   │
                   ▼
              Query Router
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
     Local       Web       Hybrid
        │          │          │
        └──────┬───┴──────────┘
               ▼
        Context Fusion
               ▼
        OpenAI GPT Model
               ▼
          Final Response
```

---

## 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.13 |
| Framework | LangChain |
| UI | Streamlit |
| Vector Database | ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI GPT |
| Web Search | Tavily |
| Hybrid Search | Vector + BM25 |
| PDF Processing | PyPDF |
| Environment | Python Dotenv |

---

## 📂 Project Structure

```text
PharmacologyGPT/
│
├── app.py
├── rag.py
├── vector_store.py
├── build_vector_db.py
├── pdf_utils.py
├── hybrid_retriever.py
├── router.py
├── query_intent.py
├── query_rewriter.py
├── web_search.py
├── context_fusion.py
├── inspect_vector_db.py
│
├── docs/
├── vector/
├── test/
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/PharmacologyGPT.git

cd PharmacologyGPT
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

### macOS/Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file.

```env
OPENAI_API_KEY=your_openai_api_key

TAVILY_API_KEY=your_tavily_api_key

VECTOR_ARCHIVE_URL=github_release_asset_url
```

---

## 📚 Build Vector Database

```bash
python build_vector_db.py
```

---

## ▶ Run Application

```bash
streamlit run app.py
```

---

## 💡 Example Queries

### Local Knowledge

- What is the mechanism of action of metformin?
- What are the adverse effects of warfarin?
- Explain insulin pharmacology.

### Web Search

- What is the latest FDA warning for semaglutide?
- Latest clinical trial for tirzepatide.
- Recent GLP-1 safety updates.

### Hybrid Retrieval

- Compare metformin pharmacology with recent safety evidence.
- Explain Ozempic using textbook knowledge and current FDA guidance.
- Compare warfarin interactions with current prescribing recommendations.

---

## 🧪 Test Suite

Run individual validation scripts:

```bash
python -m test.test_router

python -m test.test_query_intent

python -m test.test_query_rewriter

python -m test.test_web_search

python -m test.test_routed_context

python -m test.test_routed_answer
```

---

## 📈 Version History

### Version 1.0
- Basic RAG
- ChromaDB
- Streamlit

### Version 1.1
- Persistent Vector Database

### Version 1.2
- PDF Upload
- Dynamic Knowledge Base

### Version 1.3
- Hybrid Retrieval
- BM25 Search
- Query Router
- Query Intent Detection
- Query Rewriting
- Web Search
- Context Fusion

### Version 1.4
- Shared Vector Store
- Lazy Initialization
- Modular Architecture
- Deployment Configuration
- Production Refactoring

---

## 🚀 Roadmap

### Version 2.0

- FastAPI Backend
- Authentication
- Docker
- CI/CD
- PostgreSQL + pgvector
- RAG Evaluation (RAGAS / DeepEval)
- Monitoring & Logging
- API Deployment

---

## 📷 Demo

### Home Page

> _Add screenshot_

### Hybrid Retrieval Example

> _Add screenshot_

### Web Search Example

> _Add screenshot_

---

## 📦 Deployment

### Local

```bash
streamlit run app.py
```

### Streamlit Community Cloud

1. Push repository to GitHub.
2. Deploy from Streamlit Community Cloud.
3. Configure secrets:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
   - `VECTOR_ARCHIVE_URL`
4. Launch the application.

---

## 🤝 Contributing

Contributions, feature requests, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Marufa Sultana Sumi**

- LinkedIn: https://linkedin.com/in/marufasumi
- GitHub: https://github.com/marufasumi

---

⭐ If you found this project helpful, please consider giving it a star.
