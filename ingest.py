"""
Data ingestion: scrape 飞享IM website and build the BM25 retriever.
Run once before starting the Q&A service.
"""

import json
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import (
    DOCS_PERSIST_PATH,
    FSHARECHAT_URLS,
    FSHARECHAT_STATIC_KNOWLEDGE,
)


def scrape_page(url: str) -> str:
    """Fetch and extract clean text from a webpage."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"  Warning: failed to scrape {url}: {e}")
        return ""


def load_documents() -> list[Document]:
    """Load documents from website scraping + static knowledge."""
    docs: list[Document] = []

    print("Scraping 飞享IM website...")
    for url in FSHARECHAT_URLS:
        print(f"  Fetching {url}")
        text = scrape_page(url)
        if text:
            docs.append(Document(page_content=text, metadata={"source": url}))

    docs.append(Document(
        page_content=FSHARECHAT_STATIC_KNOWLEDGE,
        metadata={"source": "static_knowledge"},
    ))

    print(f"Loaded {len(docs)} documents total")
    return docs


def _chunks_to_json(chunks: list[Document]) -> list[dict]:
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in chunks]


def _json_to_chunks(data: list[dict]) -> list[Document]:
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]


def build_retriever() -> BM25Retriever:
    """Chunk documents, persist to JSON, and return a BM25Retriever."""
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    with open(DOCS_PERSIST_PATH, "w", encoding="utf-8") as f:
        json.dump(_chunks_to_json(chunks), f, ensure_ascii=False, indent=2)
    print(f"Documents saved to {DOCS_PERSIST_PATH}")

    retriever = BM25Retriever.from_documents(chunks, k=5)
    return retriever


def load_retriever() -> BM25Retriever:
    """Load persisted chunks from JSON and return a BM25Retriever."""
    with open(DOCS_PERSIST_PATH, "r", encoding="utf-8") as f:
        chunks = _json_to_chunks(json.load(f))
    return BM25Retriever.from_documents(chunks, k=5)


if __name__ == "__main__":
    build_retriever()
    print("Ingestion complete. Ready to answer questions about 飞享IM.")
