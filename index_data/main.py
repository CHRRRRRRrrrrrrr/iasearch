"""Indexation des données (TD1-3).

lit le dataset -> embedding de chaque ligne (Ollama) -> clustering KMeans
-> 1 index ElasticSearch par cluster (data_1, data_2, ...) + un index "data_clusters"
contenant l'embedding (centroïde) de chaque cluster et le nom de l'index correspondant.
"""

import html
import os
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from elasticsearch import Elasticsearch, helpers
from sklearn.cluster import KMeans

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() in ("1", "true", "yes")
DATA_FILE = Path(os.getenv("DATA_FILE", Path(__file__).parent / "data" / "objet_consultations_1000.txt"))

PCLOUD_API_URL = (
    "https://apilux3.pcloud.com/getpublinkdownload?fileid=93128819642"
    "&code=XZnqkRJZ98VUd3rEuMXAFIsEpAtnNHoKYK8k&linkpassword=&forcedownload=1"
)
CLUSTERS_INDEX = "data_clusters"


def clean_line(line: str) -> str:
    """Supprime les balises HTML (<br />...) et les espaces superflus."""
    line = html.unescape(re.sub(r"<[^>]+>", " ", line))
    return re.sub(r"\s+", " ", line).strip()


def get_data_from_cloud() -> list[str]:
    """Récupère le lien de téléchargement dans la réponse JSON de pCloud ("path" et premier "host"),
    télécharge le fichier, et le garde en cache local pour les exécutions suivantes."""
    if DATA_FILE.exists():
        text = DATA_FILE.read_text(encoding="utf-8")
    else:
        meta = httpx.get(PCLOUD_API_URL, timeout=30).json()
        if meta.get("result") != 0:
            raise RuntimeError(f"Réponse pCloud inattendue : {meta}")
        url = f"https://{meta['hosts'][0]}{meta['path']}"
        print(f"Téléchargement du dataset : {url}")
        text = httpx.get(url, timeout=120, follow_redirects=True).text
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_text(text, encoding="utf-8")

    lines = [clean_line(line) for line in text.splitlines()]
    # dédoublonnage en gardant l'ordre
    return list(dict.fromkeys(line for line in lines if line))


def ensure_model(model: str) -> None:
    """Télécharge le modèle dans le docker Ollama s'il n'y est pas déjà (équivalent de `ollama pull`)."""
    tags = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=30).json()
    if any(m["name"] == model for m in tags.get("models", [])):
        return
    print(f"Téléchargement du modèle {model} dans Ollama...")
    httpx.post(f"{OLLAMA_HOST}/api/pull", json={"model": model, "stream": False}, timeout=None).raise_for_status()


def get_data_embeddings(lines: list[str]) -> np.ndarray:
    """Calcule l'embedding de chaque ligne en les envoyant par lots au docker Ollama."""
    embeddings = []
    with httpx.Client(base_url=OLLAMA_HOST, timeout=300) as client:
        for start in range(0, len(lines), BATCH_SIZE):
            batch = lines[start : start + BATCH_SIZE]
            response = client.post("/api/embed", json={"model": EMBEDDING_MODEL, "input": batch})
            response.raise_for_status()
            embeddings.extend(response.json()["embeddings"])
            print(f"  embeddings : {min(start + BATCH_SIZE, len(lines))}/{len(lines)}", end="\r")
    print()
    return np.asarray(embeddings, dtype=np.float32)


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def cluster_data(line_embeddings: np.ndarray, lines: list[str]) -> tuple[dict[int, list[int]], np.ndarray]:
    """KMeans sur les embeddings normalisés : retourne {cluster: [indices des lignes]} et les centroïdes."""
    n_clusters = min(N_CLUSTERS, len(lines))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalize(line_embeddings))

    cluster_to_lines: dict[int, list[int]] = {}
    for line_idx, label in enumerate(labels):
        cluster_to_lines.setdefault(int(label), []).append(line_idx)
    return cluster_to_lines, normalize(kmeans.cluster_centers_)


def data_index_mapping(dims: int) -> dict:
    return {
        "properties": {
            "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            # analyseur français pour le BM25 (stemming, stop words)
            "line": {"type": "text", "analyzer": "french"},
        }
    }


def clusters_index_mapping(dims: int) -> dict:
    return {
        "properties": {
            "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            "data_index": {"type": "keyword"},
            "size": {"type": "integer"},
            "labels": {"type": "text", "analyzer": "french"},
        }
    }


def wait_for_elasticsearch(es: Elasticsearch, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if es.ping():
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"ElasticSearch ne répond pas sur {ELASTICSEARCH_HOST}")


def index_data(es: Elasticsearch, lines: list[str], line_embeddings: np.ndarray,
               cluster_to_lines: dict[int, list[int]], centroids: np.ndarray) -> None:
    dims = line_embeddings.shape[1]

    # suppression des anciens index
    for name in es.indices.get(index="data_*", ignore_unavailable=True, allow_no_indices=True):
        es.indices.delete(index=name)

    clusters_docs = []
    for position, (cluster, line_ids) in enumerate(sorted(cluster_to_lines.items()), start=1):
        data_index = f"data_{position}"
        es.indices.create(index=data_index, mappings=data_index_mapping(dims))
        helpers.bulk(
            es,
            (
                {"_index": data_index, "_source": {"embedding": line_embeddings[i].tolist(), "line": lines[i]}}
                for i in line_ids
            ),
        )
        clusters_docs.append(
            {
                "_index": CLUSTERS_INDEX,
                "_id": data_index,
                "_source": {
                    "embedding": centroids[cluster].tolist(),
                    "data_index": data_index,
                    "size": len(line_ids),
                    # les 3 premières lignes servent de libellé au cluster
                    "labels": [lines[i] for i in line_ids[:3]],
                },
            }
        )
        print(f"  {data_index} : {len(line_ids)} lignes")

    es.indices.create(index=CLUSTERS_INDEX, mappings=clusters_index_mapping(dims))
    helpers.bulk(es, clusters_docs, refresh="wait_for")
    es.indices.refresh(index="data_*")


def main() -> None:
    es = Elasticsearch(ELASTICSEARCH_HOST, request_timeout=120)
    wait_for_elasticsearch(es)

    if not FORCE_REINDEX and es.indices.exists(index=CLUSTERS_INDEX):
        count = es.count(index=CLUSTERS_INDEX)["count"]
        if count:
            print(f"Index déjà présents ({count} clusters), rien à faire. FORCE_REINDEX=true pour réindexer.")
            return

    lines = get_data_from_cloud()
    print(f"{len(lines)} lignes chargées")

    ensure_model(EMBEDDING_MODEL)
    started = time.time()
    line_embeddings = get_data_embeddings(lines)
    print(f"Embeddings calculés en {time.time() - started:.0f}s (dimension {line_embeddings.shape[1]})")

    cluster_to_lines, centroids = cluster_data(line_embeddings, lines)
    print(f"{len(cluster_to_lines)} clusters")

    index_data(es, lines, line_embeddings, cluster_to_lines, centroids)
    print("Indexation terminée")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Erreur : {error}", file=sys.stderr)
        sys.exit(1)
