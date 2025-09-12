import os
import numpy as np
from groq import Groq
import time

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"

def embed_texts(texts, model=DEFAULT_EMBED_MODEL, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=model)
            for item in resp.data:
                embeddings.append(np.array(item.embedding, dtype=float))
        except Exception as e:
            print(f"[embedding error] {e}")
            embeddings.extend([np.zeros(768)] * len(batch))
            time.sleep(0.5)
    return embeddings
