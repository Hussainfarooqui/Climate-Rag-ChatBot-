import uuid
from .loader import load_file
from .chunker import chunk_text
from .embeddings import embed_texts
from .vector_store import SqliteVectorStore
from groq import Groq
import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

class RAGPipeline:
    def __init__(self, db_path="vectors.db", embed_model=None, chat_model="openai/gpt-oss-20b"):
        self.store = SqliteVectorStore(db_path)
        self.embed_model = embed_model
        self.chat_model = chat_model

    def ingest_files(self, uploaded_files, chunk_size=400, overlap=50):
        all_chunks = []
        for f in uploaded_files:
            text = load_file(f)
            chunks = chunk_text(text, chunk_size, overlap)
            for i, ch in enumerate(chunks):
                all_chunks.append((str(uuid.uuid4()), ch, {"source": f.name, "chunk_index": i}))
        texts = [c[1] for c in all_chunks]
        embeddings = embed_texts(texts, model=self.embed_model)
        for (uid, text, meta), emb in zip(all_chunks, embeddings):
            self.store.add(uid, text, emb, meta)
        return len(all_chunks)

    def query(self, question, top_k=5):
        q_emb = embed_texts([question], model=self.embed_model)[0]
        hits = self.store.search(q_emb, top_k)
        context_texts = [f"[{h['metadata']['source']} chunk:{h['metadata']['chunk_index']}] {h['text']}" for h in hits]
        messages = [
            {"role": "system", "content": "Use the following context to answer:\n\n" + "\n\n".join(context_texts)},
            {"role": "user", "content": question}
        ]
        try:
            completion = client.chat.completions.create(messages=messages, model=self.chat_model)
            answer = completion.choices[0].message.content
        except Exception as e:
            answer = f"[Groq chat error] {e}"
        return {"answer": answer, "sources": hits}
