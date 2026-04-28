"""
RAG (Retrieval-Augmented Generation) Application with Ollama and ChromaDB.

This application provides an interactive chat interface that answers questions based on
PDF documents. It uses:
- Ollama for embeddings and chat completions
- ChromaDB for vector storage
- LangChain for orchestration
- Rich for terminal UI

The workflow:
1. Ingests PDFs from docs/to-import directory
2. Splits documents into chunks and creates embeddings
3. Stores vectors in ChromaDB
4. Provides interactive Q&A using retrieval-augmented generation
"""
import os
import shutil
import glob
import logging
import argparse
import json
from typing import List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown

from utils import trace_performance

# --- Global Initialization ---
# Rich console for formatted terminal output
console = Console()

# Configure logging to file only (not console) to avoid interfering with Rich output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log")] 
)
logger = logging.getLogger(__name__)


# --- Helpers & Decorators ---
@trace_performance("Vector Store Add (Batch Embeddings + Storage)")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def batch_add_docs_traced(vstore: Chroma, splits_data: List[Document]) -> None:
    """Adds documents to the vector store with retries for transient network issues.
    
    Args:
        vstore: ChromaDB vector store instance
        splits_data: List of document chunks to add to the vector store
        
    Raises:
        Exception: After 3 failed attempts with exponential backoff
        
    Note:
        Wrapped with @trace_performance to measure embedding generation time
    """
    vstore.add_documents(splits_data)

@trace_performance("Chat Model Response (Generation)")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def invoke_qa_chain(chain: RetrievalQA, query_text: str) -> dict:
    """Invokes the QA chain with retries for Ollama timeouts/failures.
    
    Args:
        chain: LangChain RetrievalQA chain instance
        query_text: User's question/query
        
    Returns:
        dict: Contains 'result' (answer) and 'source_documents' (retrieved chunks)
        
    Raises:
        Exception: After 3 failed attempts with exponential backoff
        
    Note:
        Wrapped with @trace_performance to measure LLM response time
    """
    return chain.invoke({"query": query_text})

def check_ollama_connection(base_url: str) -> bool:
    """Recommendation 3: Pre-flight Ollama connectivity check.
    
    Verifies that Ollama service is running and accessible before attempting
    to use it for embeddings or chat completions.
    
    Args:
        base_url: Ollama API base URL (e.g., http://localhost:11434)
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/api/version", timeout=5)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False


def ingest_documents(
    to_import_dir: str, 
    indexed_dir: str, 
    vectorstore: Chroma, 
    chunk_size: int, 
    chunk_overlap: int
) -> None:
    """Handles the batch reading, splitting, and vectorizing of new PDFs.
    
    Process:
    1. Scans to_import_dir for PDF files
    2. Loads each PDF and splits into chunks
    3. Batch-adds all chunks to vector store
    4. Moves successfully processed PDFs to indexed_dir
    
    Args:
        to_import_dir: Directory containing PDFs to process
        indexed_dir: Directory to move processed PDFs
        vectorstore: ChromaDB instance for storing embeddings
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between chunks (for context continuity)
        
    Note:
        Errors in individual PDFs are logged but don't stop batch processing
    """
    # Scan for all PDF files in the import directory
    pdf_files = glob.glob(os.path.join(to_import_dir, "*.[pP][dD][fF]"))
    
    # Early return if no work to do
    if not pdf_files:
        console.print("No new PDFs found in [dim]docs/to-import[/dim].")
        return

    console.print(f"Found {len(pdf_files)} new PDF(s) to process.")
    
    # Initialize text splitter with configured chunk parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Accumulate all document chunks before batch insertion (more efficient)
    all_splits: List[Document] = []
    processed_files: List[str] = []

    # Process each PDF individually
    for pdf_path in pdf_files:
        console.print(f"Processing: {pdf_path}")
        logger.info(f"Reading PDF: {pdf_path}")
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
            processed_files.append(pdf_path)
        except Exception as e:
            console.print(f"[bold red]Error processing {pdf_path}: {e}[/bold red]")
            logger.error(f"Error processing {pdf_path}: {e}", exc_info=True)

    # Batch insert all chunks at once (reduces network overhead)
    if all_splits:
        console.print(f"[dim]Batch-adding {len(all_splits)} chunks to vector store...[/dim]")
        batch_add_docs_traced(vectorstore, all_splits)

        # Move successfully processed files to indexed directory
        for pdf_path in processed_files:
            filename = os.path.basename(pdf_path)
            shutil.move(pdf_path, os.path.join(indexed_dir, filename))
            console.print(f"Successfully indexed and moved [green]{filename}[/green]")
            logger.info(f"Indexed and moved {filename}")


def execute_single_query(qa_chain: RetrievalQA, query: str) -> dict:
    """Executes a single query and returns results in JSON-serializable format.
    
    Args:
        qa_chain: Configured RetrievalQA chain
        query: User's question/query
        
    Returns:
        dict: Contains 'answer' (string) and 'sources' (list of dicts with metadata)
    """
    response = invoke_qa_chain(qa_chain, query)
    
    # Format sources as a list of dictionaries
    sources = []
    for doc in response["source_documents"]:
        source_path = doc.metadata.get('source', 'unknown')
        sources.append({
            "filename": os.path.basename(source_path),
            "page": doc.metadata.get('page', 'unknown'),
            "content_preview": doc.page_content[:200].replace('\n', ' ')
        })
    
    return {
        "answer": response["result"],
        "sources": sources
    }


def build_retrieval_chain(
    vectorstore: Chroma, 
    chat_model_name: str, 
    ollama_base_url: str, 
    top_k: int
) -> RetrievalQA:
    """Initializes the chat model and returns the retrieval chain.
    
    Creates a "stuff" chain that:
    1. Retrieves top_k most relevant document chunks
    2. Stuffs them into the LLM prompt as context
    3. Generates an answer based on the context
    
    Args:
        vectorstore: ChromaDB instance with embedded documents
        chat_model_name: Ollama model name for chat completions
        ollama_base_url: Ollama API base URL
        top_k: Number of document chunks to retrieve for context
        
    Returns:
        RetrievalQA: Configured LangChain retrieval chain
    """
    console.print(f"Initializing Chat model '{chat_model_name}' at {ollama_base_url}")
    chat_model = ChatOllama(
        base_url=ollama_base_url,
        model=chat_model_name,
        temperature=0
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


def run_chat_loop(qa_chain: RetrievalQA, max_query_length: int) -> None:
    """Handles the interactive Q&A loop.
    
    Continuously prompts user for questions and displays:
    - Generated answer from the LLM
    - Source documents used for context (with filename, page, preview)
    
    Args:
        qa_chain: Configured RetrievalQA chain
        max_query_length: Maximum allowed characters in a query
        
    Exit conditions:
    - User types 'exit' or 'quit'
    - User presses Ctrl+C
    """
    console.print("\n[bold magenta]--- Interactive RAG Chat ---[/bold magenta]")
    console.print("Enter your queries below. Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop.")

    while True:
        # Catch interrupt specifically on prompt input to exit cleanly
        try:
            query = Prompt.ask("\n[orange1]Question[/orange1]")
        except KeyboardInterrupt:
            break

        # Check for exit commands
        if query.lower() in ["exit", "quit"]:
            console.print("[bold red]Exiting chat...[/bold red]")
            break

        # Skip empty queries
        if not query.strip():
            continue

        # Validate query length to prevent excessive token usage
        if len(query) > max_query_length:
            console.print(f"[bold red]Query too long ({len(query)} characters). Max allowed is {max_query_length}.[/bold red]")
            continue

        console.print("[dim]Generating answer...[/dim]")
        # Execute the retrieval + generation pipeline
        try:
            response = invoke_qa_chain(qa_chain, query)
            
            # Display the generated answer with Markdown rendering
            console.print("\n[bold green]--- Answer ---[/bold green]")
            console.print(Markdown(response["result"]))

            # Display source documents for transparency and verification
            console.print("\n[bold yellow]--- Sources ---[/bold yellow]")
            for doc in response["source_documents"]:
                # Extract metadata for citation
                source = doc.metadata.get('source', 'unknown source')
                filename = os.path.basename(source)
                page = doc.metadata.get('page', 'unknown page')
                # Show preview of chunk content (first 100 chars)
                content = doc.page_content[:100].replace('\n', ' ')
                console.print(f"[dim]- {filename} (Page {page}): {content}...[/dim]")
                
        except Exception as e:
            console.print(f"[bold red]Error during query execution: {e}[/bold red]")
            logger.error(f"Error executing query: {e}", exc_info=True)


# --- Main Orchestrator ---

def main() -> None: # Recommendation 17: Type annotations
    """Main orchestrator for the RAG application.
    
    Supports two modes:
    1. Interactive mode (default): Continuous Q&A loop
    2. Single query mode (--query): Execute one query and output JSON to stdout
    
    Command-line Arguments:
        --query: Single query to execute (outputs JSON and exits)
    
    Workflow:
    1. Load configuration from environment variables
    2. Verify Ollama connectivity
    3. Initialize ChromaDB with embeddings
    4. Ingest any new PDFs from to-import directory
    5. Build retrieval chain
    6. Start interactive chat loop (or execute single query)
    7. Clean up resources on exit
    
    Environment Variables:
        OLLAMA_BASE_URL: Ollama API endpoint (default: http://localhost:11434)
        EMBEDDING_MODEL_NAME: Model for embeddings (default: embeddinggemma)
        CHAT_MODEL_NAME: Model for chat (default: gemma4:latest)
        CHUNK_SIZE: Document chunk size (default: 1000)
        CHUNK_OVERLAP: Chunk overlap (default: 100)
        RETRIEVAL_TOP_K: Number of chunks to retrieve (default: 5)
        MAX_QUERY_LENGTH: Max query characters (default: 2000)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG Application with Ollama and ChromaDB")
    parser.add_argument("--query", type=str, help="Execute a single query and output JSON results")
    args = parser.parse_args()
    # Configuration via Env Vars (Recommendations 12, 13, 14 implemented)
    # Directory structure for document management
    to_import_dir = "docs/to-import"
    indexed_dir = "docs/indexed"
    persist_directory = "./docs/chroma"
    
    # Ollama connection settings
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # Model selection
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")
    chat_model_name = os.getenv("CHAT_MODEL_NAME", "gemma4:latest")
    
    # Document processing parameters
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
    # Retrieval and query parameters
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", 5))
    max_query_length = int(os.getenv("MAX_QUERY_LENGTH", 2000))

    # Ensure directory structure exists
    os.makedirs(to_import_dir, exist_ok=True)
    os.makedirs(indexed_dir, exist_ok=True)

    # Initialize here so it exists in `locals()` for the `finally` block
    vectorstore = None

    try:
        console.print(f"Initializing Ollama bindings at {ollama_base_url}")
        
        # Pre-flight check: verify Ollama is accessible
        if not check_ollama_connection(ollama_base_url):
            console.print(f"[bold red]Error: Cannot connect to Ollama at {ollama_base_url}. Is it running?[/bold red]")
            return

        # Initialize embedding model for vector generation
        embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=embedding_model_name
        )

        # Initialize or load existing vector store
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        # Process any new documents in the import directory
        ingest_documents(to_import_dir, indexed_dir, vectorstore, chunk_size, chunk_overlap)
        # Build the RAG chain (retriever + LLM)
        qa_chain = build_retrieval_chain(vectorstore, chat_model_name, ollama_base_url, retrieval_top_k)
        
        # Check if running in single-query mode or interactive mode
        if args.query:
            # Single query mode: execute and output JSON
            try:
                result = execute_single_query(qa_chain, args.query)
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except Exception as e:
                error_result = {
                    "error": str(e),
                    "answer": None,
                    "sources": []
                }
                print(json.dumps(error_result, indent=2, ensure_ascii=False))
                logger.error(f"Error executing single query: {e}", exc_info=True)
        else:
            # Interactive mode: start chat loop
            run_chat_loop(qa_chain, max_query_length)

    except KeyboardInterrupt: # Recommendation 1: Dedicated interrupt block
        console.print("\n[bold yellow]Shutdown requested by user (Ctrl+C).[/bold yellow]")
        logger.info("Application closed via KeyboardInterrupt.")
    except Exception as e:
        console.print(f"[bold red]Critical Error: {e}[/bold red]")
        logger.critical(f"Unhandled critical error: {e}", exc_info=True)
    # Ensure clean shutdown of database connection
    finally:
        # Safely access the ChromaDB client if vectorstore was initialized
        client = getattr(vectorstore, "_client", None) if vectorstore else None
        if client is not None:
            client.close()
            # Only print to console in interactive mode
            if not (hasattr(locals().get('args'), 'query') and locals().get('args').query):
                console.print("[dim]Chroma database connection cleanly closed.[/dim]")


if __name__ == "__main__":
    main()
