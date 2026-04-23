import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

def main():
    # Configuration
    pdf_path = "example.pdf"  # Replace with your PDF path
    ollama_base_url = "http://localhost.localdomain:11434"  # Replace with your Ollama server URL
    model_name = "embeddinggemma"
    chat_model_name = "gemma4:latest"
    persist_directory = "./docs/chroma"

    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found.")
        return

    print(f"Loading PDF: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages.")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print(f"Initializing Ollama embeddings with model '{model_name}' at {ollama_base_url}")
    try:
        embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=model_name
        )

        print(f"Creating Chroma vector store at {persist_directory}...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print("Vector store created and documents persisted successfully.")

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
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        # Simple Test of Semantic Search and RAG
        if splits:
            print("\n--- Testing Semantic Search and RAG ---")
            query = input("Enter a search query: ")
            if query.strip():
                print(f"Query: {query}")
                
                print("Generating answer using RAG...")
                response = qa_chain.invoke({"query": query})
                
                print("\n--- Answer ---")
                print(response["result"])
                
                print("\n--- Sources ---")
                for doc in response["source_documents"]:
                    print(f"- {doc.metadata.get('page', 'unknown page')}: {doc.page_content[:100]}...")
            else:
                print("Empty query provided. Skipping search.")
        else:
            print("No chunks available to test search.")

    except Exception as e:
        print(f"Error during embedding, vector store, or RAG process: {e}")

if __name__ == "__main__":
    main()
