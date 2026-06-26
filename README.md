Local RAG Pipeline with Qdrant and Llama 3
This repository contains a simple, local Retrieval-Augmented Generation pipeline. It uses LangChain to coordinate everything, Qdrant as a local vector database, and Llama 3 running via Ollama for local text generation.

The pipeline takes raw text from a document, converts it into semantic embeddings using a Hugging Face model, stores it in a local database index, and uses that context to answer questions locally without relying on external cloud APIs.

Project Structure
rag-pipeline/
├── data/
│   └── document.txt
├── src/
│   ├── ingest.py
│   └── rag.py
├── .gitignore
├── README.md
└── requirements.txt

Stack
Framework: LangChain

Vector Database: Qdrant (Local Storage Mode)

Embedding Model: all-MiniLM-L6-v2 (via Hugging Face)

LLM Engine: Ollama (Llama 3)

Language: Python 3.13+


Setup
Clone the repository and navigate into it:
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline

Install the required Python packages:
pip install -r requirements.txt

Download and install Ollama from the official website, then pull the Llama 3 model in your terminal:
ollama run llama3

How to Run
Add your raw source text into data/document.txt.

Run the ingestion script to process the text and build your local Qdrant database index:
python src/ingest.py

Run the main script to query the system and get responses based on your data:
python src/rag.py