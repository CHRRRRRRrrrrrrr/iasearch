FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# installation des dépendances seules d'abord, pour profiter du cache Docker
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

EXPOSE 8000 5678

# debugpy écoute sur le port 5678 pour le "Remote Attach" de VSCode (TD2-3)
CMD ["uv", "run", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
