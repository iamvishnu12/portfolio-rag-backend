import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


def load_documents(docs_path="docs"):
    """Load PDF documents from a directory."""
    print(f"Loading documents from '{docs_path}'")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError("No PDF files found in the directory")

    print(f"Loaded {len(documents)} documents")

    for i, doc in enumerate(documents[:3]):  # preview first 3
        print(f"\nDocument {i+1}")
        print(f"Metadata: {doc.metadata}")
        print(f"Preview: {doc.page_content[:200]}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into chunks suitable for embeddings."""
    print("\nSplitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0].page_content[:200])

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create embeddings and store them in Chroma vector DB."""
    print("\nCreating embeddings...")

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    os.makedirs(persist_directory, exist_ok=True)

    print("Creating Chroma vector store...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("\nVector database created successfully")
    print(f"Stored {len(chunks)} chunks in '{persist_directory}'")

    return vector_store


def main():
    print("\nStarting RAG ingestion pipeline...\n")

    load_dotenv()

    documents = load_documents("docs")
    chunks = split_documents(documents)
    create_vector_store(chunks)

    print("\nRAG ingestion completed successfully!")


if __name__ == "__main__":
    main()