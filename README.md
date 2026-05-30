# LEO — Manulife Integration Support Assistant

LEO is a production-grade RAG chatbot that helps Manulife procurement employees
self-serve answers to Fieldglass-to-Workday integration errors — reducing support
ticket load for the procurement team.

## Problem It Solves

Manulife procurement staff faced recurring integration issues:
- Worker not found in Workday
- Incorrect manager tagged
- Name mismatches, date errors, terminated profiles
- Cost center and location errors

Instead of raising support tickets every time, employees ask LEO and get
instant answers with escalation guidance.

## Architecture

```
PDF Documents → PyPDFLoader → RecursiveCharacterTextSplitter (500 tokens, 50 overlap)
→ HuggingFace all-MiniLM-L6-v2 embeddings → ChromaDB vector store

User Question → Embed query → ChromaDB similarity search (top 5 chunks)
→ LangChain retrieval chain → Groq Llama-3 → Answer + Source citations
```

## Tech Stack

- **LLM:** Groq API (Llama-3.3-70b) — open source equivalent of Azure OpenAI GPT-4
- **Vector DB:** ChromaDB — open source equivalent of Azure AI Search
- **Embeddings:** HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Orchestration:** LangChain retrieval chain
- **Backend:** FastAPI with conversation memory (last 5 messages)
- **Frontend:** Streamlit
- **Evaluation:** Custom keyword relevancy + source accuracy framework

## Features

- Answers grounded strictly in knowledge base documents
- Conversation memory across follow-up questions
- Source citations showing which document each answer came from
- Graceful out-of-scope handling
- Custom evaluation pipeline (evaluate.py)

## Project Structure

```
leo/
├── docs/                        # Knowledge base PDFs
│   ├── error_runbook.pdf
│   ├── escalation_guide.pdf
│   ├── field_mapping.pdf
│   ├── procurement_policy.pdf
│   └── contingent_worker_policy.pdf
├── data/                        # ChromaDB vector store (auto-generated)
├── ingest.py                    # Ingestion pipeline
├── app.py                       # FastAPI backend
├── chat.py                      # Streamlit UI
├── evaluate.py                  # Evaluation framework
├── requirements.txt
└── .env                         # GROQ_API_KEY (not committed)
```

## How to Run

### 1. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 3. Ingest documents
```bash
python ingest.py
```

### 4. Start the backend
```bash
uvicorn app:app --reload
```

### 5. Start the UI
```bash
streamlit run chat.py
```

Open http://localhost:8501

### 6. Run evaluation
```bash
python evaluate.py
```

## Example Questions

- "My worker is showing in Fieldglass but not in Workday, what should I do?"
- "The manager tagged is incorrect, how do I fix it?"
- "Worker profile is showing as terminated, how do I resolve this?"
- "Who do I contact for a cost center not found error?"
- "What happens when I close an engagement in Fieldglass?"

## Enterprise Context

This project replicates the architecture of a RAG chatbot built at Manulife
Financial for procurement support. In the enterprise deployment:
- Azure OpenAI GPT-4 Turbo was used instead of Groq
- Azure AI Search was used instead of ChromaDB
- Azure Active Directory handled authentication
- Deployed on Azure Kubernetes Service

This personal project uses open-source equivalents to demonstrate the same
RAG pipeline architecture at zero cost.
