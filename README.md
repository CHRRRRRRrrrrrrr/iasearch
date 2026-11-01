# IA Search — projet d'évaluation « Architectures IA »

Moteur de recherche sur 1000 objets de consultations de marchés publics, inspiré de la chaîne
d'indexation de ChatGPT (`rerank400_cluster100_bm10_ann4096`) :

| Brique | Techno | Rôle |
|---|---|---|
| Script d'indexation | Python + uv | lit le dataset, calcule les embeddings, clusterise (KMeans), indexe |
| Base de données | ElasticSearch 9.3.2 | index vectoriel des clusters + 1 index par cluster (BM25 + knn) |
| Modèle « embedding » | Ollama `qwen3-embedding:0.6b` | vectorise les lignes et la question |
| Modèle « rerank » | Ollama `qwen3.5:0.8b` | score de pertinence question / document |
| Backend | FastAPI | orchestre la recherche |
| Front | React + TypeScript (Vite), servi par nginx | champ de recherche et résultats |

Tout est dockerisé dans un seul `docker-compose.yml`.

## Architecture

```
                     ┌──────────────── indexation (index_data) ────────────────┐
dataset (pCloud) ──► │ embedding de chaque ligne ──► KMeans (20 clusters)       │
                     │   ├─► index data_1 … data_20  { embedding, line }        │
                     │   └─► index data_clusters    { embedding, data_index }   │
                     └──────────────────────────────────────────────────────────┘

Front React ──POST /search──► Backend FastAPI
                                 1. embedding de la question (Ollama)
                                 2. knn sur data_clusters ──► clusters les plus proches
                                 3. recherche hybride dans ces data_X : BM25 (match) + knn
                                 4. rerank par LLM : p_yes / (p_yes + p_no) sur les logprobs
                                 5. élimination des documents avec un score < 0.4
```

### Détails

- **Indexation** (`index_data/main.py`) : `get_data_from_cloud()` appelle l'API pCloud et récupère le lien
  dans `path` + premier `host` (le fichier est aussi fourni dans `index_data/data/` au cas où le lien expire).
  Les balises HTML (`<br />`) sont nettoyées et les doublons supprimés (966 lignes uniques).
  `get_data_embeddings()` envoie les lignes par lots à Ollama, `cluster_data()` fait un KMeans sur
  les embeddings normalisés. L'embedding d'un cluster est son centroïde. Chaque cluster stocke aussi
  ses 3 premières lignes comme libellé.
- **Recherche de clusters** : requête `knn` sur `data_clusters` (similarité cosinus).
- **Recherche hybride** : `bool.should` avec un `match` BM25 (analyseur français) et une requête `knn`,
  combinés linéairement (le score knn est pondéré par `KNN_BOOST`, car le score BM25 n'est pas borné).
- **Rerank** (`fastapi-backend/rerank.py`) : chaque document est envoyé au modèle dans une requête séparée,
  en parallèle, avec une sortie contrainte au schéma JSON `{"answer": "yes" | "no"}`. On lit les
  `logprobs` du token de la réponse, on additionne les probabilités des variantes de « yes » et de « no »,
  et le score vaut `p_yes / (p_yes + p_no)`. Les résultats sont triés par score décroissant.
- **Seuil 0.4** : un score de rerank inférieur à 0.4 indique que le document n'a pas de rapport avec la
  question. Ces documents sont écartés : une question hors sujet (« recette de la tarte aux pommes »)
  ne renvoie aucun résultat.

## Lancer le projet

Prérequis : Docker Desktop.

```bash
docker compose up -d --build
```

Au premier démarrage :
1. `ollama-init` télécharge les deux modèles dans le docker Ollama,
2. `indexer` calcule les embeddings et remplit ElasticSearch (environ 4 min sur CPU), puis s'arrête.
   Aux démarrages suivants il ne réindexe pas, sauf avec `FORCE_REINDEX=true docker compose up indexer`.

Suivre l'indexation : `docker logs -f indexer`

| Service | URL |
|---|---|
| Front | http://localhost:3000 |
| Backend (Swagger) | http://localhost:8000/docs |
| ElasticSearch | http://localhost:9200 |
| Ollama | http://localhost:11434 |

Avec une carte NVIDIA, pour donner accès au GPU à Ollama :

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Sans GPU, le rerank prend environ 15 à 45 s par recherche (20 documents évalués par le modèle).
Il peut être désactivé dans le front (case « Rerank par LLM »).

## API

`POST /search`

```json
{ "query": "rénovation énergétique de logements", "clusters": 3, "size": 10, "rerank": true, "min_score": 0.4 }
```

Réponse : les clusters trouvés, les résultats (`line`, `data_index`, `es_score`, `rerank_score`),
les résultats écartés par le rerank et le temps passé dans chaque étape.

Autres endpoints : `GET /health`, `GET /clusters`.

## Développement et debug

- Backend en local : `cd fastapi-backend && uv sync && uv run uvicorn main:app --reload`
  (ou la configuration VSCode « Python Debugger: FastAPI »).
- Debug dans le docker : le backend démarre avec `debugpy` sur le port 5678. Lancer la configuration
  VSCode « Python Debugger: Remote Attach (docker) » depuis le dossier `fastapi-backend`.
- Front en local : `cd react-front && npm install && npm run dev` (http://localhost:5173).
- Indexation en local : `cd index_data && uv run main.py` (ElasticSearch et Ollama doivent tourner).

## Variables d'environnement

| Variable | Défaut | Service |
|---|---|---|
| `ELASTICSEARCH_HOST` | `http://localhost:9200` | indexer, backend |
| `OLLAMA_HOST` | `http://localhost:11434` | indexer, backend |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | indexer, backend |
| `RERANK_MODEL` | `qwen3.5:0.8b` | backend |
| `N_CLUSTERS` | `20` | indexer |
| `FORCE_REINDEX` | `false` | indexer |
| `MIN_RERANK_SCORE` | `0.4` | backend |
| `KNN_BOOST` | `10` | backend |
