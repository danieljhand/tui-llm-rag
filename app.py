import os
import shutil
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

def main():
    # Configuration
    to_import_dir = "docs/to-import"
    indexed_dir = "docs/indexed"
    ollama_base_url = "http://localhost:11434"
    model_name = "embeddinggemma"
    chat_model_name = "gemma4:latest"
    persist_directory = "./docs/chroma"

    # Ensure directories exist
    os.makedirs(to_import_dir, exist_ok=True)
    os.makedirs(indexed_dir, exist_ok=True)

    print(f"Initializing Ollama embeddings with model '{model_name}' at {ollama_base_url}")
    try:
        embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=model_name
        )

        # Process new documents
        pdf_files = glob.glob(os.path.join(to_import_dir, "*.pdf"))
        if pdf_files:
            print(f"Found {len(pdf_files)} new PDF(s) to process.")
            
            # Initialize or load vector store
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )

            for pdf_path in pdf_files:
                print(f"Processing: {pdf_path}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    splits = text_splitter.split_documents(docs)
                    vectorstore.add_documents(splits)
                    
                    # Move to indexed directory
                    filename = os.path.basename(pdf_path)
                    shutil.move(pdf_path, os.path.join(indexed_dir, filename))
                    print(f"Successfully indexed and moved {filename}")
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")
        else:
            print("No new PDFs found in docs/to-import.")
            # We still need the vectorstore for the chat loop
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )

        # Chat and RAG Setup
        print(f"Initializing Chat model '{chat_model_name}' at {ollama_base_url}")
        chat_model = ChatOllama(
            base_url=ollama_base_url,
            model=chat_model_name,
            temperature=0
        )

        # Create the RAG chain using RetrievalQA
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )

        print("\n--- Interactive RAG Chat ---")
        print("Enter your queries below. Type 'exit' or 'quit' to stop.")
        
        while True:
            query = input("\nQuery > ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break
            
            if not query.strip():
                continue
                
            print("Generating answer...")
            try:
                response = qa_chain.invoke({"query": query})
                print("\n--- Answer ---")
                print(response["result"])
                
                print("\n--- Sources ---")
                for doc in response["source_documents"]:
                    page = doc.metadata.get('page', 'unknown page')
                    content = doc.page_content[:100].replace('\n', ' ')
                    print(f"- Page {page}: {content}...")
            except Exception as e:
                print(f"Error during query: {e}")

    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()
