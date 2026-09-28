# deploy/kit/Dockerfile - образ фасада veles (corporate-профиль deploy-kit).
# Собирается docker compose (build). Внутри: CLI/фасад + embedding (torch, faiss).
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY veles ./veles

RUN pip install --no-cache-dir ".[st,faiss]"

EXPOSE 8400

CMD ["python", "-m", "veles", "serve", "--host", "0.0.0.0", "--port", "8400"]
