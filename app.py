import os
import shutil
import glob
import logging
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

from utils import trace_performance

# --- Global Initialization ---
console = Console()

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
    """Adds documents to the vector store with retries for transient network issues."""
    vstore.add_documents(splits_data)

@trace_performance("Chat Model Response (Generation)")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def invoke_qa_chain(chain: RetrievalQA, query_text: str) -> dict:
    """Invokes the QA chain with retries for Ollama timeouts/failures."""
    return chain.invoke({"query": query_text})

def check_ollama_connection(base_url: str) -> bool:
    """Recommendation 3: Pre-flight Ollama connectivity check."""
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
    """Handles the batch reading, splitting, and vectorizing of new PDFs."""
    pdf_files = glob.glob(os.path.join(to_import_dir, "*.pdf"))
    
    if not pdf_files:
        console.print("No new PDFs found in [dim]docs/to-import[/dim].")
        return

    console.print(f"Found {len(pdf_files)} new PDF(s) to process.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_splits: List[Document] = []
    processed_files: List[str] = []

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

    if all_splits:
        console.print(f"[dim]Batch-adding {len(all_splits)} chunks to vector store...[/dim]")
        batch_add_docs_traced(vectorstore, all_splits)

        for pdf_path in processed_files:
            filename = os.path.basename(pdf_path)
            shutil.move(pdf_path, os.path.join(indexed_dir, filename))
            console.print(f"Successfully indexed and moved [green]{filename}[/green]")
            logger.info(f"Indexed and moved {filename}")


def build_retrieval_chain(
    vectorstore: Chroma, 
    chat_model_name: str, 
    ollama_base_url: str, 
    top_k: int
) -> RetrievalQA:
    """Initializes the chat model and returns the retrieval chain."""
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
    """Handles the interactive Q&A loop."""
    console.print("\n[bold magenta]--- Interactive RAG Chat ---[/bold magenta]")
    console.print("Enter your queries below. Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop.")

    while True:
        # Catch interrupt specifically on prompt input to exit cleanly
        try:
            query = Prompt.ask("\n[orange1]Question[/orange1]")
        except KeyboardInterrupt:
            break

        if query.lower() in ["exit", "quit"]:
            console.print("[bold red]Exiting chat...[/bold red]")
            break

        if not query.strip():
            continue

        if len(query) > max_query_length:
            console.print(f"[bold red]Query too long ({len(query)} characters). Max allowed is {max_query_length}.[/bold red]")
            continue

        console.print("[dim]Generating answer...[/dim]")
        try:
            response = invoke_qa_chain(qa_chain, query)
            
            console.print("\n[bold green]--- Answer ---[/bold green]")
            console.print(response["result"])

            console.print("\n[bold yellow]--- Sources ---[/bold yellow]")
            for doc in response["source_documents"]:
                source = doc.metadata.get('source', 'unknown source')
                filename = os.path.basename(source)
                page = doc.metadata.get('page', 'unknown page')
                content = doc.page_content[:100].replace('\n', ' ')
                console.print(f"[dim]- {filename} (Page {page}): {content}...[/dim]")
                
        except Exception as e:
            console.print(f"[bold red]Error during query execution: {e}[/bold red]")
            logger.error(f"Error executing query: {e}", exc_info=True)


# --- Main Orchestrator ---

def main() -> None: # Recommendation 17: Type annotations
    # Configuration via Env Vars (Recommendations 12, 13, 14 implemented)
    to_import_dir = "docs/to-import"
    indexed_dir = "docs/indexed"
    persist_directory = "./docs/chroma"
    
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")
    chat_model_name = os.getenv("CHAT_MODEL_NAME", "gemma4:latest")
    
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", 5))
    max_query_length = int(os.getenv("MAX_QUERY_LENGTH", 2000))

    os.makedirs(to_import_dir, exist_ok=True)
    os.makedirs(indexed_dir, exist_ok=True)

    # Initialize here so it exists in `locals()` for the `finally` block
    vectorstore = None

    try:
        console.print(f"Initializing Ollama bindings at {ollama_base_url}")
        
        if not check_ollama_connection(ollama_base_url):
            console.print(f"[bold red]Error: Cannot connect to Ollama at {ollama_base_url}. Is it running?[/bold red]")
            return

        embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=embedding_model_name
        )

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        ingest_documents(to_import_dir, indexed_dir, vectorstore, chunk_size, chunk_overlap)
        qa_chain = build_retrieval_chain(vectorstore, chat_model_name, ollama_base_url, retrieval_top_k)
        
        run_chat_loop(qa_chain, max_query_length)

    except KeyboardInterrupt: # Recommendation 1: Dedicated interrupt block
        console.print("\n[bold yellow]Shutdown requested by user (Ctrl+C).[/bold yellow]")
        logger.info("Application closed via KeyboardInterrupt.")
    except Exception as e:
        console.print(f"[bold red]Critical Error: {e}[/bold red]")
        logger.critical(f"Unhandled critical error: {e}", exc_info=True)
    finally:
        client = getattr(vectorstore, "_client", None) if vectorstore else None
        if client is not None:
            client.close()
            console.print("[dim]Chroma database connection cleanly closed.[/dim]")


if __name__ == "__main__":
    main()
