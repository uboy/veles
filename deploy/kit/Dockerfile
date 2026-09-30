# deploy/kit/Dockerfile - образ фасада veles (corporate-профиль deploy-kit).
# Собирается docker compose (build). Внутри: CLI/фасад + embedding (torch, faiss).
#
# Слои: deps до кода - обновление veles/ не перекачивает зависимости (кэш слоя
# RUN живёт, пока pyproject не менялся). TORCH_CPU=1 (дефолт) ставит CPU-сборку
# torch заранее - без неё pip на Linux тянет CUDA-вариант (~2.5 ГБ с nvidia-*),
# безGPU-хостам он не нужен; GPU-хостам - собрать с --build-arg TORCH_CPU=0.
FROM python:3.11-slim

WORKDIR /app

ARG TORCH_CPU=1

COPY pyproject.toml ./
RUN if [ "$TORCH_CPU" = "1" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY veles ./veles

RUN pip install --no-cache-dir ".[st,faiss]"

EXPOSE 8400

CMD ["python", "-m", "veles", "serve", "--host", "0.0.0.0", "--port", "8400"]
