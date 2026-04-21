"""
Data ingestion: scrape 飞享IM website and build the vector store.
Run once before starting the Q&A service.
"""

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    FSHARECHAT_URLS,
    FSHARECHAT_STATIC_KNOWLEDGE,
)


def scrape_page(url: str) -> str:
    """Fetch and extract clean text from a webpage."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"  Warning: failed to scrape {url}: {e}")
        return ""


def load_documents() -> list[Document]:
    """Load documents from website scraping + static knowledge."""
    docs: list[Document] = []

    # 1. Scrape live pages
    print("Scraping 飞享IM website...")
    for url in FSHARECHAT_URLS:
        print(f"  Fetching {url}")
        text = scrape_page(url)
        if text:
            docs.append(Document(page_content=text, metadata={"source": url}))

    # 2. Always include curated static knowledge (covers gaps from scraping)
    docs.append(Document(
        page_content=FSHARECHAT_STATIC_KNOWLEDGE,
        metadata={"source": "static_knowledge"},
    ))

    print(f"Loaded {len(docs)} documents total")
    return docs


def build_vectorstore() -> Chroma:
    """Chunk documents and store embeddings in ChromaDB."""
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Building ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Vector store saved to {CHROMA_PERSIST_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


if __name__ == "__main__":
    build_vectorstore()
    print("Ingestion complete. Ready to answer questions about 飞享IM.")
