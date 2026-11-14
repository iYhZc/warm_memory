"""Optional embedding retrieval; vectors are disposable derived state."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from .domain import MemoryRecord

_EMBED_BATCH = 16
"""Stay below providers with a 20-input batch limit."""


def _vector(value: object, dimension: int = 0) -> list[float]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 32768:
        raise ValueError("invalid embedding vector")
    if dimension and len(value) != dimension:
        raise ValueError("embedding dimension changed")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("non-finite embedding vector")
    result = [float(x) for x in value]
    norm = math.hypot(*result)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("invalid embedding norm")
    return [x / norm for x in result]


def _fingerprint(provider: object) -> str:
    config = getattr(provider, "provider_config", {})
    fields = ("id", "type", "embedding_model", "model", "embedding_dimensions",
              "embedding_dimensions_mode", "embedding_api_base", "api_base", "base_url",
              "model_revision", "embedding_model_revision")
    selected = {k:config[k] for k in fields if isinstance(config, dict) and k in config}
    selected["runtime_model"] = str(getattr(provider, "model", getattr(provider, "_model", "")))
    dim = getattr(provider, "get_dim", None)
    selected["declared_dimension"] = dim() if callable(dim) else 0
    return hashlib.sha256(json.dumps(selected,sort_keys=True,default=str).encode()).hexdigest()


class VectorRecall:
    """Cache only vectors for explicitly supplied, already scoped memories."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self._queries: dict[tuple[str,str], list[float]] = {}

    def _connect(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        db = sqlite3.connect(self.path,timeout=10)
        db.execute("""CREATE TABLE IF NOT EXISTS vectors (
            namespace TEXT NOT NULL, memory_id INTEGER NOT NULL,
            content_hash TEXT NOT NULL, vector TEXT NOT NULL,
            PRIMARY KEY(namespace, memory_id))""")
        db.commit()
        return db

    def _read(self, namespace: str, records: Sequence[MemoryRecord]) -> dict[int,list[float]]:
        with closing(self._connect()) as db:
            result = {}
            for record in records:
                row = db.execute("SELECT vector FROM vectors WHERE namespace=? AND memory_id=? AND content_hash=?",
                                 (namespace,record.id,hashlib.sha256(record.content.encode()).hexdigest())).fetchone()
                if row:
                    try: result[record.id] = _vector(json.loads(row[0]))
                    except (ValueError, TypeError): pass
            return result

    def _write(self, namespace: str, records: Sequence[MemoryRecord], vectors: Sequence[list[float]]):
        with closing(self._connect()) as db:
            with db:
                db.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                    [(namespace,r.id,hashlib.sha256(r.content.encode()).hexdigest(),json.dumps(v))
                     for r,v in zip(records,vectors)])

    @staticmethod
    def _rank(records, query_vector, cached, min_similarity, top_k):
        hits = []
        for record in records:
            vector = cached.get(record.id)
            if vector is None or len(vector) != len(query_vector):
                continue
            score = max(-1.0, min(1.0, math.fsum(a*b for a,b in zip(query_vector, vector))))
            if score >= min_similarity:
                hits.append((record.id, score))
        return sorted(hits, key=lambda row: (-row[1], row[0]))[:top_k]

    async def search_cached(self, records: Sequence[MemoryRecord], query: str, provider: object,
                            *, min_similarity: float, top_k: int) -> list[tuple[int, float]]:
        """Score completed chunks after a timeout; never starts a remote call."""
        if not records: return []
        async with self._lock:
            base = _fingerprint(provider)
            query_vector = self._queries.get((base, query))
            if query_vector is None: return []
            cached = await asyncio.to_thread(self._read, base + ":" + str(len(query_vector)), records)
            return self._rank(records, query_vector, cached, min_similarity, top_k)

    async def search(self, records: Sequence[MemoryRecord], query: str, provider: object,
                     *, min_similarity: float, top_k: int) -> list[tuple[int,float]]:
        if not records: return []
        async with self._lock:
            base = _fingerprint(provider)
            query_key = (base,query)
            query_vector = self._queries.get(query_key)
            if query_vector is None:
                query_vector = _vector(await provider.get_embedding(query))
                declared = getattr(provider,"get_dim",lambda:0)()
                if declared and declared != len(query_vector):
                    raise ValueError("configured embedding dimension mismatch")
                while len(self._queries) >= 128: self._queries.pop(next(iter(self._queries)))
                self._queries[query_key] = query_vector
            dimension = len(query_vector)
            namespace = base + ":" + str(dimension)
            cached = await asyncio.to_thread(self._read,namespace,records)
            missing = [r for r in records if r.id not in cached or len(cached[r.id]) != dimension]
            for start in range(0, len(missing), _EMBED_BATCH):
                chunk = missing[start:start + _EMBED_BATCH]
                try:
                    raw = await provider.get_embeddings([r.content for r in chunk])
                    if not isinstance(raw,(list,tuple)) or len(raw) != len(chunk):
                        raise ValueError("embedding batch size mismatch")
                    vectors = [_vector(v,dimension) for v in raw]
                except Exception:
                    # Keep good chunks; cancellation still propagates. A later
                    # query only retries vectors that are actually missing.
                    continue
                await asyncio.to_thread(self._write,namespace,chunk,vectors)
                cached.update({r.id:v for r,v in zip(chunk,vectors)})
            if not any(len(cached.get(r.id, ())) == dimension for r in records):
                # The caller emits a rate-limited warning and falls back to B/local.
                raise ValueError("no usable memory embeddings")
            return self._rank(records, query_vector, cached, min_similarity, top_k)
