from fastapi import FastAPI, Query
from pydantic import BaseModel
from openai import OpenAI
import faiss
import numpy as np
from dotenv import load_dotenv
import os


load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"
USE_QUERY_REWRITER = False  # set False if you want direct embedding

index = faiss.read_index("faiss_index.idx")

with open("id_to_record.pkl", "rb") as f:
    id_to_record = pickle.load(f)

with open("idx_to_id.pkl", "rb") as f:
    idx_to_id = pickle.load(f)


print(f"Loaded FAISS index with {index.ntotal} vectors")
print(f"Loaded {len(id_to_record)} records mapping")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Zenodo Semantic Search API")

class SearchResult(BaseModel):
    record: dict
    distance: float

def create_embedding(text: str):
    """Generate embedding for query."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

def rewrite_query(query: str):
    """Optional: rewrite user query into keywords."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Rewrite this search query into short keywords: {query}"}],
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

@app.get("/search", response_model=list[SearchResult])
def search(query: str = Query(..., description="Your search phrase"),
           k: int = Query(5, description="Number of results to return")):

    if USE_QUERY_REWRITER:
        query_clean = rewrite_query(query)
    else:
        query_clean = query

    query_emb = create_embedding(query_clean)

    distances, indices = index.search(query_emb, k)

    k = 5
    distances, indices = index.search(query_emb, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if int(idx) in idx_to_id:
            rec = id_to_record[idx_to_id[int(idx)]]

            # normalize: wrap strings into dict
            if isinstance(rec, str):
                rec = {"text": rec}

            results.append(SearchResult(
                    record=rec,
                    distance=float(distances[0][rank])
                ))
    
    return results
