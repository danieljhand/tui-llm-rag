# TUI RAG Chat

A terminal-based Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents using Ollama and LangChain.

## Features

- **Interactive CLI**: Chat with your documents directly in the terminal using a beautiful TUI.
- **Automated Indexing**: Automatically detects new PDF files in `docs/to-import`, processes them, and moves them to `docs/indexed`.
- **Local & Private**: Runs entirely on your local machine using Ollama, ensuring your data never leaves your computer.
- **Source Tracking**: Every answer includes citations from the source documents (page number and content snippet).
- **Vector Storage**: Uses ChromaDB for persistent vector storage.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running.
- PDF documents to index.

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tui-llm-rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: If you don't have a requirements.txt, you'll need to install: langchain, langchain-community, langchain-ollama, langchain-chroma, chromadb, pypdf, rich, ollama)*

3. **Prepare Ollama models**:
   The application expects the following models to be pulled in Ollama:
   - `embeddinggemma` (or your preferred embedding model)
   - `gemma4:latest` (or your preferred chat model)

   ```bash
   ollama pull embeddinggemma
   ollama pull gemma4:latest
   ```

4. **Prepare your documents**:
   Place the PDF files you want to index into the `docs/to-import` directory.

## Usage

Run the application:

```bash
python app.py
```

- **Indexing**: Upon startup, the app will scan `docs/to-import`, process any new PDFs found, and move them to `docs/indexed`.
- **Chatting**: Once indexing is complete, you can enter questions in the prompt.
- **Exiting**: Type `exit` or `quit` to end the session.

## Directory Structure

- `docs/to-import`: Place new PDFs here for processing.
- `docs/indexed`: Where processed PDFs are moved after indexing.
- `docs/chroma`: Persistent storage for the vector database.

## License

This project is licensed under the Apache License 2.0.
