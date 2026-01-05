# app/core/rag.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

from app.core.logging_config import setup_logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = setup_logging(__name__)

DEFAULT_PDF_PATHS = [
    Path("data/raw/Bhatla.pdf"),
    Path("data/raw/EBA_ECB 2024 Report on Payment Fraud.pdf"),
]

DEFAULT_TEXT_PATHS = [
    Path("data/raw/EBA_ECB chart_summary.md"),
]

DB_FAISS_PATH = Path("data/processed/db_faiss_multi")

# English-focused, small & fast
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cpu"


@dataclass
class RagArtifact:
    db: FAISS
    model_name: str = MODEL_NAME


def _get_embeddings() -> HuggingFaceEmbeddings:
    logger.debug("Init embeddings | model=%s | device=%s", MODEL_NAME, DEVICE)
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": DEVICE},
    )


def _load_pdfs_as_documents(pdf_paths: List[Path]) -> List[Document]:
    all_docs: List[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = pdf_path.name
        all_docs.extend(docs)
    return all_docs

def _load_texts_as_documents(text_paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    for p in text_paths:
        if not p.exists():
            logger.warning("Text file not found (skip): %s", p)
            continue
        content = p.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=content, metadata={"source": p.name, "page": 0}))
    return docs

def _split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def bootstrap_rag(
    pdf_paths: List[Path] = DEFAULT_PDF_PATHS,
    text_paths: List[Path] = DEFAULT_TEXT_PATHS,
    faiss_dir: Path = DB_FAISS_PATH,
    force_rebuild: bool = False,
) -> RagArtifact:
    """
    Build or load FAISS index for one or more PDFs (merged into a single index).
    """
    for p in pdf_paths:
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")

    t0 = time.perf_counter()
    logger.info(
        "Bootstrap RAG start | pdfs=%s | faiss_dir=%s | force_rebuild=%s",
        [p.name for p in pdf_paths],
        faiss_dir,
        force_rebuild,
    )

    embeddings = _get_embeddings()
    faiss_dir.mkdir(parents=True, exist_ok=True)

    index_faiss = faiss_dir / "index.faiss"
    index_pkl = faiss_dir / "index.pkl"
    index_ok = index_faiss.exists() and index_pkl.exists()

    if index_ok and not force_rebuild:
        logger.info("FAISS index found: loading local index | faiss_dir=%s", faiss_dir)
        db = FAISS.load_local(
            str(faiss_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Bootstrap RAG done (loaded) | elapsed=%.2fs", time.perf_counter() - t0)
        return RagArtifact(db=db)

    if force_rebuild and index_ok:
        logger.warning("Force rebuild enabled: rebuilding existing index | faiss_dir=%s", faiss_dir)
    else:
        logger.info("No FAISS index found: building new index | faiss_dir=%s", faiss_dir)

    # -------- Build --------
    t_load = time.perf_counter()
    docs_pdf = _load_pdfs_as_documents(pdf_paths)
    docs_txt = _load_texts_as_documents(text_paths)
    docs = docs_pdf + docs_txt

    logger.info(
        "Docs loaded | pdf_pages=%d | text_docs=%d | total_docs=%d | elapsed=%.2fs",
        len(docs_pdf),
        len(docs_txt),
        len(docs),
        time.perf_counter() - t_load,
    )

    t_split = time.perf_counter()
    chunks = _split_documents(docs)
    logger.info("Documents split | chunks=%d | elapsed=%.2fs", len(chunks), time.perf_counter() - t_split)

    t_build = time.perf_counter()
    db = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS built | elapsed=%.2fs", time.perf_counter() - t_build)

    t_save = time.perf_counter()
    db.save_local(str(faiss_dir))
    logger.info("FAISS saved | path=%s | elapsed=%.2fs", faiss_dir, time.perf_counter() - t_save)

    logger.info("Bootstrap RAG done (rebuilt) | elapsed=%.2fs", time.perf_counter() - t0)
    return RagArtifact(db=db)



def retrieve_context(rag: RagArtifact, query: str, k: int = 5) -> List[Tuple[str, dict]]:
    t0 = time.perf_counter()
    docs = rag.db.similarity_search(query, k=k)
    logger.info(
        "Retrieve | k=%d | hits=%d | elapsed=%.2fms | query=%r",
        k,
        len(docs),
        (time.perf_counter() - t0) * 1000,
        query,
    )
    return [(d.page_content, d.metadata) for d in docs]


def format_citations(metas: List[dict]) -> List[str]:
    """
    Produce human-friendly citations like: ["Bhatla.pdf p.3", "Bhatla.pdf p.7"]
    """
    cites: List[str] = []
    for m in metas:
        src = m.get("source") or "document"
        page0 = m.get("page")
        if isinstance(page0, int):
            cites.append(f"{src} p.{page0 + 1}")
        else:
            cites.append(src)

    # de-duplicate preserving order
    seen = set()
    out: List[str] = []
    for c in cites:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


if __name__ == "__main__":
    bootstrap_rag()
    print(f"RAG index ready: {DB_FAISS_PATH}")
