# pinecone_upload.py
import json
import time
from typing import List
from tqdm import tqdm
import requests
from pinecone import Pinecone, ServerlessSpec
import config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 1536 for text-embedding-3-small

# -----------------------------
# Initialize clients
# -----------------------------
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Ollama local HTTP settings (running on localhost:11434 by default)
OLLAMA_HOST = getattr(config, "OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = getattr(config, "OLLAMA_MODEL", "mxbai-embed-large")


def _ollama_batch_embed(texts: List[str]) -> List[List[float]]:
    """Call local Ollama HTTP API to embed a batch of texts.

    Requires ollama server running and model pulled:
      ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0
      ollama serve
    """
    # Try modern Ollama embeddings endpoint first
    endpoints = ["/v1/embeddings", "/embed"]
    last_error = None
    for ep in endpoints:
        url = f"{OLLAMA_HOST}{ep}"
        payload = {"model": OLLAMA_MODEL, "input": texts}
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
        except requests.HTTPError as e:
            # try next endpoint if 404 or other HTTP error
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

        data = resp.json()
        # Common possible shapes:
        # - {"data": [{"embedding": [...]}, ...]}
        # - {"embeddings": [[...], [...]]}
        # - [[...], [...]]
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                embs = []
                for item in data["data"]:
                    if isinstance(item, dict) and ("embedding" in item or "embeddings" in item):
                        emb = item.get("embedding") or item.get("embeddings")
                        embs.append(emb)
                    elif isinstance(item, list):
                        embs.append(item)
                if embs:
                    return embs

            if "embeddings" in data and isinstance(data["embeddings"], list):
                return data["embeddings"]

        if isinstance(data, list) and all(isinstance(d, list) for d in data):
            return data

        # If we got here, response was unexpected for this endpoint; try next
        last_error = RuntimeError(f"Unexpected Ollama response format at {url}: {data}")

    # If none of the endpoints worked, raise the last collected error with guidance
    raise RuntimeError(
        f"Failed to get embeddings from Ollama. Tried endpoints {endpoints}. Last error: {last_error}"
    )

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using local Ollama HTTP API.

    Requires Ollama server running and the model pulled:
      ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0
      ollama serve
    """
    try:
        return _ollama_batch_embed(texts)
    except Exception as e:
        raise RuntimeError(f"Ollama embedding failed: {e}")

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)
        # sleep between batches to avoid rate limits
        time.sleep(1)

    print("All items uploaded successfully.")

# -----------------------------
if __name__ == "__main__":
    main()