# deploy/kit/Dockerfile - образ фасада veles (corporate-профиль deploy-kit).
# Собирается docker compose (build). Внутри: CLI/фасад + embedding (torch, faiss).
#
# Слои: ВСЕ зависимости (включая st/faiss) ставятся до копирования кода - правка
# veles/ переиспользует кэш deps-слоя целиком (пересборка занимает секунды, без
# перекачки sentence-transformers/faiss - сотни МБ). Хитрость: deps-слой ставит
# пакет "." без кода (pyproject уже скопирован, wheel собирается без модулей) -
# это резолвит и качает зависимости; сам код ставится следом с --no-deps.
# TORCH_CPU=1 (дефолт) ставит CPU-сборку torch заранее - без неё pip на Linux
# тянет CUDA-вариант (~2.5 ГБ с nvidia-*), безGPU-хостам он не нужен;
# GPU-хостам - собрать с --build-arg TORCH_CPU=0.
FROM python:3.11-slim

WORKDIR /app

ARG TORCH_CPU=1

COPY pyproject.toml ./
RUN if [ "$TORCH_CPU" = "1" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    pip install --no-cache-dir ".[st,faiss]"

COPY veles ./veles

RUN pip install --no-cache-dir --no-deps .

EXPOSE 8400

CMD ["python", "-m", "veles", "serve", "--host", "0.0.0.0", "--port", "8400"]
