"""Backend de recherche (TD2 + TD3).

1. embedding de la question (Ollama, modèle "embedding")
2. recherche knn dans "data_clusters" -> clusters les plus proches (champ "data_index")
3. recherche hybride (BM25 + knn) dans les index "data_X" de ces clusters
4. rerank des résultats par le modèle "chat" et élimination des scores < MIN_RERANK_SCORE
"""

import os
import time

import httpx
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from rerank import RERANK_MODEL, rerank

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.4"))
# poids du score knn dans la recherche hybride linéaire (le score BM25 n'est pas borné, le score knn est entre 0 et 1)
KNN_BOOST = float(os.getenv("KNN_BOOST", "10"))
CLUSTERS_INDEX = "data_clusters"

# qwen3-embedding est entraîné avec une instruction devant les requêtes (pas devant les documents)
QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve public procurement notices that match it\nQuery: "

app = FastAPI(title="IA Search", description="Recherche sémantique par clusters, hybride et rerank par LLM")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

es = Elasticsearch(ELASTICSEARCH_HOST, request_timeout=60)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, examples=["rénovation énergétique de logements"])
    clusters: int = Field(3, ge=1, le=20, description="nombre de clusters dans lesquels chercher")
    size: int = Field(10, ge=1, le=50, description="nombre de résultats retournés")
    rerank: bool = Field(True, description="trier les résultats avec le modèle rerank")
    min_score: float = Field(MIN_RERANK_SCORE, ge=0, le=1, description="score de rerank minimum")


def embed(text: str) -> list[float]:
    response = httpx.post(
        f"{OLLAMA_HOST}/api/embed", json={"model": EMBEDDING_MODEL, "input": text}, timeout=120
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def search_clusters(query_vector: list[float], k: int) -> list[dict]:
    """Recherche vectorielle (knn) des clusters les plus proches de la question."""
    response = es.search(
        index=CLUSTERS_INDEX,
        knn={"field": "embedding", "query_vector": query_vector, "k": k, "num_candidates": 100},
        source_excludes=["embedding"],
        size=k,
    )
    return [{**hit["_source"], "score": hit["_score"]} for hit in response["hits"]["hits"]]


def search_documents(query: str, query_vector: list[float], indexes: list[str], size: int) -> list[dict]:
    """Recherche hybride dans les index des clusters : BM25 sur le texte + knn sur l'embedding,
    combinés linéairement (somme des scores pondérés)."""
    response = es.search(
        index=",".join(indexes),
        query={
            "bool": {
                "should": [
                    {"match": {"line": {"query": query}}},
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_vector,
                            "k": size,
                            "num_candidates": 100,
                            "boost": KNN_BOOST,
                        }
                    },
                ]
            }
        },
        source_excludes=["embedding"],
        size=size,
    )
    return [
        {"line": hit["_source"]["line"], "data_index": hit["_index"], "es_score": hit["_score"]}
        for hit in response["hits"]["hits"]
    ]


@app.get("/")
def root():
    return {"service": "iasearch backend", "docs": "/docs"}


@app.get("/health")
def health():
    status = {"elasticsearch": False, "ollama": False, "indexed": False}
    try:
        status["elasticsearch"] = es.ping()
        status["indexed"] = bool(es.indices.exists(index=CLUSTERS_INDEX))
    except Exception:
        pass
    try:
        status["ollama"] = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5).status_code == 200
    except Exception:
        pass
    return status


@app.get("/clusters")
def clusters():
    """Liste des clusters avec leur taille et les 3 premières lignes comme libellé."""
    response = es.search(index=CLUSTERS_INDEX, source_excludes=["embedding"], size=1000, sort=["data_index"])
    return [hit["_source"] for hit in response["hits"]["hits"]]


@app.post("/search")
async def search(request: SearchRequest):
    timings = {}
    try:
        started = time.perf_counter()
        query_vector = await run_in_threadpool(embed, QUERY_INSTRUCTION + request.query)
        timings["embedding"] = time.perf_counter() - started

        started = time.perf_counter()
        found_clusters = await run_in_threadpool(search_clusters, query_vector, request.clusters)
        indexes = [cluster["data_index"] for cluster in found_clusters]
        # on récupère plus de candidats que nécessaire pour laisser le rerank les départager
        candidates_size = request.size * 2 if request.rerank else request.size
        documents = await run_in_threadpool(search_documents, request.query, query_vector, indexes, candidates_size)
        timings["elasticsearch"] = time.perf_counter() - started
    except httpx.HTTPError as error:
        raise HTTPException(503, f"Ollama indisponible : {error}") from error
    except Exception as error:
        raise HTTPException(503, f"ElasticSearch indisponible ou données non indexées : {error}") from error

    rejected = []
    if request.rerank and documents:
        started = time.perf_counter()
        try:
            documents = await rerank(request.query, documents)
        except httpx.HTTPError as error:
            raise HTTPException(503, f"Modèle rerank indisponible : {error}") from error
        timings["rerank"] = time.perf_counter() - started
        rejected = [document for document in documents if document["rerank_score"] < request.min_score]
        documents = [document for document in documents if document["rerank_score"] >= request.min_score]

    return {
        "query": request.query,
        "clusters": found_clusters,
        "results": documents[: request.size],
        "rejected": rejected,
        "rerank_model": RERANK_MODEL if request.rerank else None,
        "timings": {name: round(seconds, 3) for name, seconds in timings.items()},
    }
