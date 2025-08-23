import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy import Column, Integer, Text, select, text
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"
USE_QUERY_REWRITER = False  # set False if you want direct embedding
TABLE_NAME = os.environ["TABLE_NAME"]
DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


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

class KeywordSearchResult(BaseModel):
    id: int
    title: str
    description: str
    rank: float | None = None

class Base(AsyncAttrs, DeclarativeBase):
    pass

class Record(Base):
    __tablename__ = TABLE_NAME

    creator = Column(Text)
    date = Column(Text)
    description = Column(Text)
    identifier = Column(Integer, primary_key=True, index=True)
    publisher = Column(Text)
    relation = Column(Text)
    rights = Column(Text) 
    subject = Column(Text)
    type = Column(Text)
    title = Column(Text)
    language = Column(Text)
    source = Column(Text)
    contributor = Column(Text)

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

@app.get("/match", response_model=list[SearchResult])
def match(query: str = Query(..., description="Your search phrase")):
    pass
    
@app.get("/keyword-search", response_model=list[KeywordSearchResult])
def keyword_search(query: str = Query(..., description="Keyword search phrase"),
                   limit: int = 10):
    sql = text("""
        SELECT *
        FROM records
        WHERE title ILIKE '%' || :query || '%'
        OR description ILIKE '%' || :query || '%'
        LIMIT :limit
        """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"query": query, "limit": limit}).fetchall()

    results = [KeywordSearchResult(
        id=row.id,
        title=row.title,
        description=row.description,
        rank=row.rank
    ) for row in rows]

    return results


@app.get("/keyword-search", response_model=list[KeywordSearchResult])
async def keyword_search(query: str, limit: int = 10):
    async with AsyncSessionLocal() as session:
        stmt = select(Record).where(
            Record.title.ilike(f"%{query}%") |
            Record.description.ilike(f"%{query}%")
        ).limit(limit)

        result = await session.execute(stmt)
        rows = result.scalars().all()

        return [
            KeywordSearchResult(
                id=r.identifier,
                title=r.title,
                description=r.description,
            )
            for r in rows
        ]