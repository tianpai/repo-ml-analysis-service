#  USAGE
#  curl "http://localhost:8000/analyze-keywords?topN=10"
#  curl "http://localhost:8000/analyze-keywords?topN=10&includeRelated=true"
#
from __future__ import annotations

import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# ── stop-list import ──────────────────────────────────────────────────────────
from utils.languages import languages as _LANGS

LANGUAGE_FILTER: set[str] = {s.lower() for s in _LANGS}

# ── MongoDB setup ─────────────────────────────────────────────────────────────
load_dotenv()  # reads .env if present
MONGO = os.getenv("MONGO")  # same var on Railway
DB_NAME, COLL_NAME = "test", "repos"

client = MongoClient(MONGO, serverSelectionTimeoutMS=5000)
coll = client[DB_NAME][COLL_NAME]

# ── ML model (loaded once) ────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

# Get port from environment variable (Railway sets this)
PORT = int(os.getenv("PORT", 8000))

# ── aggregation pipeline (from Compass) ───────────────────────────────────────
PIPELINE = [
    {"$group": {"_id": None, "latestTrendingDate": {"$max": "$trendingDate"}}},
    {
        "$lookup": {
            "from": COLL_NAME,
            "let": {"latestDate": "$latestTrendingDate"},
            "pipeline": [
                {"$match": {"$expr": {"$eq": ["$trendingDate", "$$latestDate"]}}},
                {"$project": {"topics": 1, "_id": 0}},
            ],
            "as": "reposWithLatestDate",
        }
    },
    {"$unwind": "$reposWithLatestDate"},
    {"$unwind": "$reposWithLatestDate.topics"},
    {"$group": {"_id": None, "topics": {"$push": "$reposWithLatestDate.topics"}}},
    {"$project": {"_id": 0, "topics": 1}},
]


def fetch_latest_topics() -> List[str]:
    """Run aggregation and filter out language tokens."""
    doc = coll.aggregate(PIPELINE, maxTimeMS=60_000, allowDiskUse=True).next()
    raw: List[str] = doc.get("topics", [])
    return [t for t in raw if t.lower() not in LANGUAGE_FILTER]


# ── ML clustering util ────────────────────────────────────────────────────────
def top_keywords(
    keywords: List[str],
    top_n: int,
    return_related: bool = False,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Cluster keywords semantically and return representatives (+ related)."""
    # frequency map
    freq: Dict[str, int] = {}
    for kw in keywords:
        k = kw.strip().lower()
        if k:
            freq[k] = freq.get(k, 0) + 1
    uniques = list(freq)
    if not uniques:
        return [], {}

    # embeddings + similarity
    emb = model.encode(uniques)
    sim = cosine_similarity(emb)

    clustering = AgglomerativeClustering(
        distance_threshold=0.25,
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
    ).fit(1 - sim)

    # label → list of keywords
    clusters: Dict[int, List[str]] = {}
    for kw, lbl in zip(uniques, map(int, clustering.labels_)):
        clusters.setdefault(lbl, []).append(kw)

    rep_to_related: Dict[str, List[str]] = {}
    scores: List[Tuple[str, int]] = []

    for kws in clusters.values():
        rep = max(kws, key=lambda k: freq[k])  # most-frequent in cluster
        total = sum(freq[k] for k in kws)
        scores.append((rep, total))
        rep_to_related[rep] = [w for w in kws if w != rep]

    scores.sort(key=lambda x: x[1], reverse=True)
    top_reps = [kw for kw, _ in scores[:top_n]]

    if return_related:
        return top_reps, {k: rep_to_related[k] for k in top_reps}

    return top_reps, {}


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/analyze-keywords")
async def analyze_keywords(
    topN: int = Query(5, ge=1, le=30),
    includeRelated: bool = Query(False, description="Return cluster-mates as well"),
):
    """
    GET /analyze-keywords?topN=10&includeRelated=true
    Pulls latest topics, removes language tokens, clusters semantically,
    and returns the top N representative keywords.
    """
    try:
        topics = fetch_latest_topics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    top, related = top_keywords(topics, topN, includeRelated)
    resp: Dict[str, object] = {"topKeywords": top}
    if includeRelated:
        resp["related"] = related
    return resp


@app.get("/")
async def root():
    return {"message": "repo-ml-analysis-service up (MongoDB + ML keyword clustering)"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
