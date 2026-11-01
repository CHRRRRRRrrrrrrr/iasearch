"""Rerank par LLM (TD3).

Pour chaque document, on demande au modèle "chat" (qwen3.5:0.8b) si le document répond à la
question, avec une sortie contrainte à un JSON {"answer": "yes" | "no"}. On lit ensuite les
logprobs du token de la réponse : score = p_yes / (p_yes + p_no).
Les documents sont envoyés dans des requêtes séparées, en parallèle.
"""

import asyncio
import math
import os

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3.5:0.8b")
RERANK_CONCURRENCY = int(os.getenv("RERANK_CONCURRENCY", "4"))

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
}

SYSTEM_PROMPT = (
    "You are a search relevance judge. The documents are French public procurement notices. "
    "Given a search query and a document, answer \"yes\" if the document is relevant to the query "
    "(it matches the subject the user is looking for), otherwise answer \"no\". "
    "Reply only with JSON matching the schema."
)


def _normalize_token(token: str) -> str:
    return token.strip().strip('"').strip().lower()


def _answer_score(logprobs: list[dict]) -> float | None:
    """Trouve le token de la réponse, additionne les probabilités des variantes de yes / no
    parmi les top_logprobs, et retourne p_yes / (p_yes + p_no)."""
    for entry in logprobs:
        if _normalize_token(entry.get("token", "")) not in ("yes", "no"):
            continue
        candidates = entry.get("top_logprobs") or [entry]
        p_yes = sum(math.exp(c["logprob"]) for c in candidates if _normalize_token(c["token"]) == "yes")
        p_no = sum(math.exp(c["logprob"]) for c in candidates if _normalize_token(c["token"]) == "no")
        if p_yes + p_no == 0:
            return None
        return p_yes / (p_yes + p_no)
    return None


async def score_document(client: httpx.AsyncClient, query: str, document: str) -> float:
    response = await client.post(
        "/api/chat",
        json={
            "model": RERANK_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nDocument: {document}\n\nIs the document relevant to the query?"},
            ],
            "format": ANSWER_SCHEMA,
            "think": False,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 10,
            "options": {"temperature": 0, "num_predict": 16},
        },
    )
    response.raise_for_status()
    score = _answer_score(response.json().get("logprobs") or [])
    return 0.0 if score is None else score


async def rerank(query: str, documents: list[dict], text_field: str = "line") -> list[dict]:
    """Ajoute un champ "rerank_score" à chaque document et les trie par score décroissant."""
    semaphore = asyncio.Semaphore(RERANK_CONCURRENCY)

    async with httpx.AsyncClient(base_url=OLLAMA_HOST, timeout=120) as client:

        async def scored(document: dict) -> dict:
            async with semaphore:
                return {**document, "rerank_score": await score_document(client, query, document[text_field])}

        results = await asyncio.gather(*(scored(document) for document in documents))

    return sorted(results, key=lambda document: document["rerank_score"], reverse=True)
