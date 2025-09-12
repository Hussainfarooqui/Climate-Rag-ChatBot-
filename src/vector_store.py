import sqlite3
import json
import numpy as np

class SqliteVectorStore:
    def __init__(self, db_path="vectors.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()

    def _init_table(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            text TEXT,
            embedding TEXT,
            metadata TEXT
        )
        """)
        self.conn.commit()

    def add(self, id, text, embedding, metadata):
        c = self.conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO chunks (id,text,embedding,metadata) VALUES (?,?,?,?)",
            (id, text, json.dumps(embedding.tolist()), json.dumps(metadata))
        )
        self.conn.commit()

    def all(self):
        c = self.conn.cursor()
        c.execute("SELECT id,text,embedding,metadata FROM chunks")
        items = []
        for r in c.fetchall():
            items.append({
                "id": r[0],
                "text": r[1],
                "embedding": np.array(json.loads(r[2]), dtype=float),
                "metadata": json.loads(r[3])
            })
        return items

    def search(self, query_embedding, top_k=5):
        entries = self.all()
        if not entries:
            return []
        embs = np.stack([e["embedding"] for e in entries])
        qn = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
        embs_n = np.vstack([e/ (np.linalg.norm(e)+1e-12) for e in embs])
        sims = embs_n.dot(qn)
        idxs = sims.argsort()[::-1][:top_k]
        return [{"score": float(sims[i]), **entries[i]} for i in idxs]
