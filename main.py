import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# ── Setup ─────────────────────────────────────────────────────────────-------
load_dotenv()

# ── ML model (loaded once) ────────────────────────────────────────────────────
# model = SentenceTransformer("all-MiniLM-L12-v2")
model = SentenceTransformer("all-MiniLM-L12-v2")  # smaller model

app = FastAPI()
PORT = int(os.getenv("PORT", 8000))


# ── ML clustering util ────────────────────────────────────────────────────────
def top_keywords(
    keywords: List[str],
    top_n: int,
    return_related: bool = False,
    distance_threshold: float = 0.25,
    return_cluster_sizes: bool = False,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, int]]:
    """Cluster keywords semantically and return representatives (+ related)."""
    freq: Dict[str, int] = {}
    for kw in keywords:
        k = kw.strip().lower()
        if k:
            freq[k] = freq.get(k, 0) + 1
    uniques = list(freq)
    if not uniques:
        return [], {}, {}

    # embeddings + similarity
    emb = model.encode(uniques)
    sim = cosine_similarity(emb)

    clustering = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        # HINT: ignore all error or warnings about this
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
    ).fit(1 - sim)

    # label → list of keywords
    clusters: Dict[int, List[str]] = {}
    for kw, lbl in zip(uniques, map(int, clustering.labels_)):
        clusters.setdefault(lbl, []).append(kw)

    rep_to_related: Dict[str, List[str]] = {}
    rep_to_cluster_size: Dict[str, int] = {}
    scores: List[Tuple[str, int]] = []

    for kws in clusters.values():
        rep = max(kws, key=lambda k: freq[k])  # most-frequent in cluster
        total = sum(freq[k] for k in kws)
        scores.append((rep, total))
        rep_to_related[rep] = [w for w in kws if w != rep]
        rep_to_cluster_size[rep] = len(kws)

    scores.sort(key=lambda x: x[1], reverse=True)
    top_reps = [kw for kw, _ in scores[:top_n]]

    related_dict = {k: rep_to_related[k] for k in top_reps} if return_related else {}
    cluster_sizes_dict = (
        {k: rep_to_cluster_size[k] for k in top_reps} if return_cluster_sizes else {}
    )

    return top_reps, related_dict, cluster_sizes_dict


# ── Pydantic models ──────────────────────────────────────────────────────────
class TopicsRequest(BaseModel):
    topics: List[str]
    topN: int = 5
    includeRelated: bool = False
    distance_threshold: float = 0.25
    includeClusterSizes: bool = False


# ── routes ────────────────────────────────────────────────────────────────────
@app.post("/analyze-keywords")
async def analyze_keywords_post(request: TopicsRequest):
    """
    POST /analyze-keywords
    Accepts a list of topics and returns the analysis result.
    """
    if not request.topics:
        raise HTTPException(status_code=400, detail="Topics list cannot be empty")

    if not (1 <= request.topN <= 30):
        raise HTTPException(status_code=400, detail="topN must be between 1 and 30")

    if not (0.01 <= request.distance_threshold <= 1.0):
        raise HTTPException(
            status_code=400, detail="distance_threshold must be between 0.01 and 1.0"
        )

    top, related, cluster_sizes = top_keywords(
        request.topics,
        request.topN,
        request.includeRelated,
        request.distance_threshold,
        request.includeClusterSizes,
    )
    resp: Dict[str, object] = {"topKeywords": top}
    if request.includeRelated:
        resp["related"] = related
    if request.includeClusterSizes:
        resp["clusterSizes"] = cluster_sizes
    return resp


@app.get("/")
async def root():
    return {"message": "repo-ml-analysis-service up (ML keyword clustering)"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
