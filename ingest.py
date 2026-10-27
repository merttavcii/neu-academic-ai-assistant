from __future__ import annotations

"""Load PDFs from /data, extract text with PyPDF2, chunk, upsert into ChromaDB."""

import hashlib
from pathlib import Path

from PyPDF2 import PdfReader

import db as db_mod
import rag

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def _ingest_log(msg: str, *, verbose: bool) -> None:
    if verbose:
        print(f"[ingest] {msg}", flush=True)


def _pdf_paths() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in DATA_DIR.glob("*.pdf") if p.is_file())


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        parts.append(t)
    return "\n\n".join(parts).strip()


def upsert_text_as_source(text: str, source: str, *, verbose: bool = False) -> dict:
    """
    Ham metni (ör. PDF'ten çıkarılmış) parçalayıp Chroma'ya yazar; aynı `source`
    etiketine sahip eski parçaları önce siler. LLM uydurması değil, verilen metin indekslenir.
    """
    text = (text or "").strip()
    if not text:
        return {"chunks_added": 0, "message": "Boş metin, indekslenmedi."}

    collection = db_mod.get_collection()
    try:
        collection.delete(where={"source": source})
    except Exception:
        pass

    chunks = [c for c in rag.chunk_text(text) if c.strip()]
    if not chunks:
        return {"chunks_added": 0, "message": "Parça üretilemedi."}

    key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    ids: list[str] = []
    metadatas: list[dict] = []
    for i, _ch in enumerate(chunks):
        ids.append(f"t_{key}_{i:05d}")
        metadatas.append({"source": source, "chunk_index": i})

    _ingest_log(f"upsert_text_as_source source={source!r} chunks={len(chunks)}", verbose=verbose)
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return {"chunks_added": len(chunks), "message": "Tamam", "source": source}


def ingest_all_pdfs(*, reset: bool = False, verbose: bool = False) -> dict:
    """
    Read all PDFs under ./data, chunk, store in Chroma.
    Returns stats dict.
    """
    _ingest_log(f"start (reset={reset})", verbose=verbose)
    _ingest_log(f"data_dir={DATA_DIR.resolve()}", verbose=verbose)

    paths = _pdf_paths()
    _ingest_log(f"pdf files found: {len(paths)}", verbose=verbose)
    if verbose and paths:
        _ingest_log("  " + ", ".join(p.name for p in paths), verbose=verbose)

    if reset:
        _ingest_log("purging Chroma storage (full reset)", verbose=verbose)
        db_mod.purge_storage()
    collection = db_mod.get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for pdf_path in paths:
        if not reset:
            try:
                collection.delete(where={"source": pdf_path.name})
            except Exception:
                pass

        _ingest_log(f"reading: {pdf_path.name}", verbose=verbose)
        text = extract_pdf_text(pdf_path)
        if not text:
            _ingest_log(f"  skip (no extractable text): {pdf_path.name}", verbose=verbose)
            continue
        chunks = [c for c in rag.chunk_text(text) if c.strip()]
        _ingest_log(
            f"  extracted {len(text)} chars -> {len(chunks)} chunks",
            verbose=verbose,
        )
        for i, chunk in enumerate(chunks):
            cid = f"{pdf_path.stem}_{i:05d}"
            ids.append(cid)
            documents.append(chunk)
            metadatas.append({"source": pdf_path.name, "chunk_index": i})

    if not ids:
        _ingest_log("no chunks to upsert (empty index unchanged unless reset cleared it)", verbose=verbose)
        result = {
            "files": len(paths),
            "chunks_added": 0,
            "message": "Veri klasöründe PDF yok veya metin çıkarılamadı.",
        }
        _ingest_log(f"done {result}", verbose=verbose)
        return result

    _ingest_log(f"upserting {len(ids)} chunks into ChromaDB", verbose=verbose)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    _ingest_log("upsert complete", verbose=verbose)

    known_names = {p.name for p in paths}
    if not reset and known_names:
        try:
            data = collection.get(include=["metadatas"])
            metas = data.get("metadatas") or []
            stale_sources = set()
            for m in metas:
                src = (m or {}).get("source")
                if not src or src.startswith("upload:"):
                    continue
                if src not in known_names:
                    stale_sources.add(src)
            for src in stale_sources:
                collection.delete(where={"source": src})
        except Exception:
            pass

    result = {
        "files": len(paths),
        "chunks_added": len(ids),
        "message": "Tamam",
    }
    _ingest_log(f"done {result}", verbose=verbose)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index PDFs from ./data into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the collection and rebuild from ./data PDFs.",
    )
    args = parser.parse_args()

    print("[ingest] CLI run - ingesting from ./data ...", flush=True)
    out = ingest_all_pdfs(reset=args.reset, verbose=True)
    print("[ingest] files:", out.get("files"), "| chunks_added:", out.get("chunks_added"), "| message:", out.get("message"), flush=True)
