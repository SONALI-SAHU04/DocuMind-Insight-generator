# DocuMind Insight Generator

DocuMind is an AI-powered document question-answering system built using Retrieval-Augmented Generation (RAG). It enables users to upload documents and retrieve context-aware answers through semantic search while ensuring complete offline execution and data privacy.

## Features

* Document-based Question Answering
* Semantic Search using Vector Embeddings
* Offline Execution using Local LLM
* Multi-format Document Support (PDF, TXT, DOCX)
* RAG Evaluation Metrics
* Developer Console for Testing and Debugging

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* Ollama
* Llama 3
* Nomic Embeddings

## Installation

Clone the repository:

```bash
git clone https://github.com/SONALI-SAHU04/DocuMind-Insight-generator.git
cd DocuMind-Insight-generator
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install required models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

## Run the Main Application

Start Ollama:

```bash
ollama serve
```

Run Streamlit:

```bash
streamlit run app.py
```

## Run the Developer Console

Launch the developer console:

```bash
streamlit run dev_app.py
```

The developer console provides:

* Retrieval debugging
* Chunk inspection
* Source document tracking
* Latency monitoring
* RAG evaluation metrics
