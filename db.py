"""ChromaDB: persistent local vector store for RAG chunks."""

from __future__ import annotations

import shutil
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "rag_documents"

_embedding_fn = None
_client = None


def get_chroma_path() -> str:
    return str(CHROMA_DIR)


def _embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )
    return _embedding_fn


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _open_collection(client):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def get_collection():
    """
    Open (or create) the Chroma collection. Diskteki HNSW indeksi bozuksa
    (ör. yarım kalmış bir yazma, chromadb sürüm uyuşmazlığı, ya da
    OneDrive/Dropbox gibi bir senkronizasyon aracının dosyaları ortasında
    kilitlemesi/değiştirmesi — "Error in compaction: ... Error loading hnsw
    index" hatası tam olarak bunun belirtisidir) ChromaDB bir istisna fırlatır
    ve bu durum eskiden app.py'deki başlatma kodunda yutulup "RAG ingest
    atlandı" olarak loglanıyordu; bu da RAG'ı kalıcı olarak kırık bırakıyordu
    (her başlangıçta aynı hata tekrar ediyordu). Burada depoyu bir kez
    otomatik olarak sıfırlayıp (purge_storage) tekrar deniyoruz, böylece
    uygulama kendi kendini onarıyor. Sıfırlama, diskteki eski indeksi siler;
    PDF'ler bir sonraki ingest_all_pdfs() çağrısında sıfırdan yeniden
    indekslenir.
    """
    client = get_client()
    try:
        return _open_collection(client)
    except Exception as e:
        print(f"[db] Koleksiyon okunamadı, Chroma deposu sıfırlanıyor: {e}")
        purge_storage()
        client = get_client()
        return _open_collection(client)


def purge_storage() -> None:
    """Remove on-disk Chroma data (fixes corrupt HNSW / compaction errors)."""
    global _client
    _client = None
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def reset_collection() -> None:
    """Drop collection; if the store is corrupt, purge the whole directory."""
    try:
        client = get_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        get_collection()
    except Exception:
        purge_storage()
        get_collection()