# RAG Implementation v1.0

## Description
This is a basic Retrieval-Augmented Generation (RAG) pipeline for question answering over PDF documents using local embeddings and a language model. It loads PDFs, splits them into chunks, generates embeddings, stores them in a Chroma vector database, and uses a local LLM (Ollama) for answer generation.

## Tool Stack & Frameworks
- Python 3.11+
- LangChain
- ChromaDB
- Ollama (for local LLM and embeddings)
- PyPDF (PDF loading)
- pytest (testing)
- docling (optional, for document handling)
- boto3 (optional, for AWS integration)

## Models Used
- Embeddings: `nomic-embed-text` via Ollama
- LLM: Ollama (model can be customized)

## How it Works
1. **populate_database.py**: Loads PDFs from `data/`, splits documents, generates embeddings, and stores them in ChromaDB.
2. **query_data.py**: Accepts a user query, retrieves relevant chunks from ChromaDB, constructs a prompt, and queries the LLM for an answer.
3. **test_rag.py**: Contains sample tests for the RAG pipeline.

## Steps to Run
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Start Ollama and ensure the `nomic-embed-text` model is available.
3. Populate the database:
   ```sh
   python populate_database.py
   ```
   (Add `--reset` to clear and rebuild the database)
4. Query the system:
   ```sh
   python query_data.py "Your question here"
   ```
5. Run tests:
   ```sh
   pytest test_rag.py
   ```

