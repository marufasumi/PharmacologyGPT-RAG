# 💊 PharmacologyGPT

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-red?style=for-the-badge&logo=streamlit)](https://pharmacologygpt.streamlit.app)

Enterprise-grade **Hybrid Retrieval-Augmented Generation (Hybrid RAG)** assistant that combines trusted pharmacology textbooks with real-time web search to answer drug information, FDA safety updates, and clinical pharmacology questions.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-black)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## ⭐ Key Highlights

- Enterprise-style Hybrid RAG architecture
- Intelligent query routing (Local • Web • Hybrid)
- BM25 + Vector Search with Reciprocal Rank Fusion (RRF)
- Modular LangChain pipeline
- Persistent ChromaDB knowledge base
- Cloud deployment on Streamlit
- Live FDA safety updates via Tavily Search

## 🏗️ System Architecture

<p align="center">
<img src="docs/images/architecture.png" width="900">
</p>

## ✨ Features

- Hybrid Retrieval (Vector + BM25)
- Intelligent Query Routing
- Query Intent Detection
- Query Rewriting
- Context Fusion
- Local PDF Knowledge Base
- Live Web Search
- User PDF Upload
- Persistent ChromaDB Storage
- Cloud Deployment

## 🛠️ Technology Stack

**Core**
- Python
- LangChain
- Streamlit

**Retrieval**
- ChromaDB
- BM25
- Reciprocal Rank Fusion (RRF)

**LLM**
- GPT-5 Nano
- text-embedding-3-small

**Search**
- Tavily Search

**Document Processing**
- PyPDF

## 📸 Application Preview

### Home Page

![Home](docs/images/home.png)

### Local Knowledge Retrieval

![Local](docs/images/local.png)

### Live Web Search

![Web](docs/images/web.png)


## 🚀 Quick Start

```bash
git clone https://github.com/marufasumi/PharmacologyGPT-RAG.git

cd PharmacologyGPT-RAG

pip install -r requirements.txt

streamlit run app.py

