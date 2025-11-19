#  Urban Regulations RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers **urban planning regulation** questions (zoning, building rules, PLU, etc.) using:

- Real regulation PDFs (zoning by-laws, PLU regulations)
- Vector search over chunks (Chroma + sentence-transformers)
- An OpenAI chat model for grounded answers with **citations**

---

## 1. Motivation & Use Case

Urban planning regulations are:

- Long, technical, and hard to search
- Spread across multiple PDFs / legal texts
- Used by citizens, planners, architects, developers

This project builds a prototype **Urban Regulations Assistant** that:

> Given a natural-language question (e.g.  
> “What are the rules for building heights in residential zones?”),  
> retrieves relevant regulation clauses and generates an answer  
> **with explicit references to the source documents (file + page).**

Target users:

- Urban planners and planning students  
- Architects and developers  
- Citizens who want to understand local zoning rules

---

## 2. Features

**Core RAG pipeline**

- Ingestion: PDFs → pages → cleaned chunks
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: **Chroma** (persisted locally)
- Retrieval: top-k relevant chunks for each question
- Generation: OpenAI chat model with a regulation-aware system prompt
- Answer format: natural language + list of **sources (file, page, section)**

**Serving & UI**

- **FastAPI** backend: `/ask` endpoint
- **Gradio** chat UI:
  - Chat-style interface
  - Shows answer + sources

**Urban/regulation specific**

- Designed for **zoning codes, PLU regulations, building regulations**
- Supports multiple documents / cities (e.g. EN + FR)

---

## 3. Project Structure

```text
urban-regulations-rag/
├─ src/
│  ├─ config.py         # Paths, model names, config
│  ├─ ingestion/
│  │  └─ build_index.py # Ingestion → chunking → Chroma index builder
│  ├─ rag/
│  │  ├─ embeddings.py  # SBERTEmbeddings wrapper
│  │  ├─ retriever.py   # Chroma retriever
│  │  └─ qa.py          # RAG QA pipeline (retrieval + LLM)
│  ├─ api/
│  │  └─ main.py        # FastAPI app with /ask endpoint
│  └─ ui/
│     └─ app.py         # Gradio chat UI calling the API
├─ requirements.txt
├─ .env.example
└─ README.md
