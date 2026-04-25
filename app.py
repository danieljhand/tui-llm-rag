import os
import shutil
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.prompt import Prompt
from utils import trace_performance

def main():
    console = Console()
    # Configuration
    to_import_dir = "docs/to-import"
    indexed_dir = "docs/indexed"
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "embeddinggemma")
    chat_model_name = os.getenv("CHAT_MODEL_NAME", "gemma4:latest")
    persist_directory = "./docs/chroma"

    # Ensure directories exist
    os.makedirs(to_import_dir, exist_ok=True)
    os.makedirs(indexed_dir, exist_ok=True)

    console.print(f"Initializing Ollama embeddings with model '{embedding_model_name}' at {ollama_base_url}")
    try:
        embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=embedding_model_name
        )

        # Initialize vector store once (shared by ingestion and chat)
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        # Process new documents
        pdf_files = glob.glob(os.path.join(to_import_dir, "*.pdf"))
        if pdf_files:
            console.print(f"Found {len(pdf_files)} new PDF(s) to process.")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )

            for pdf_path in pdf_files:
                console.print(f"Processing: {pdf_path}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    #loader = UnstructuredPDFLoader(pdf_path)
                    docs = loader.load()
                    splits = text_splitter.split_documents(docs)

                    # We use a wrapper approach because we cannot decorate
                    # an object method call directly in-place like this
                    # without a helper function or wrapping the object.
                    # To keep it simple and avoid syntax errors,
                    # we'll use a small helper function.

                    def add_docs_traced(vstore, splits_data):
                        @trace_performance("Vector Store Add (Embeddings + Storage)")
                        def wrapper():
                            vstore.add_documents(splits_data)
                        return wrapper()

                    add_docs_traced(vectorstore, splits)

                    # Move to indexed directory
                    filename = os.path.basename(pdf_path)
                    shutil.move(pdf_path, os.path.join(indexed_dir, filename))
                    console.print(f"Successfully indexed and moved {filename}")
                except Exception as e:
                    console.print(f"Error processing {pdf_path}: {e}")
        else:
            console.print("No new PDFs found in docs/to-import.")

        # Chat and RAG Setup
        console.print(f"Initializing Chat model '{chat_model_name}' at {ollama_base_url}")
        chat_model = ChatOllama(
            base_url=ollama_base_url,
            model=chat_model_name,
            temperature=0
        )

        # Create the RAG chain using RetrievalQA
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        console.print("\n[bold magenta]--- Interactive RAG Chat ---[/bold magenta]")
        console.print("Enter your queries below. Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop.")
        
        while True:
            query = Prompt.ask("\n[orange1]Question[/orange1]")
            if query.lower() in ["exit", "quit"]:
                console.print("[bold red]Exiting chat...[/bold red]")
                break
            
            if not query.strip():
                continue
                
            console.print("[dim]Generating answer...[/dim]")
            try:
                # We use a helper to wrap the call since we can't decorate a line of code
                def run_qa(chain, query_text):
                    @trace_performance("Chat Model Response (Generation)")
                    def wrapper():
                        return chain.invoke({"query": query_text})
                    return wrapper()

                response = run_qa(qa_chain, query)
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
                console.print(f"[bold red]Error during query: {e}[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Critical Error: {e}[/bold red]")
    finally:
        # Close the Chroma client to release the database lock.
        # Use getattr to stay safe if vectorstore was only partially constructed.
        client = getattr(locals().get("vectorstore"), "_client", None)
        if client is not None:
            client.close()
            console.print("[dim]Chroma database connection closed.[/dim]")

if __name__ == "__main__":
    main()
