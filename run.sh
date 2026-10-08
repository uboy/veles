#!/usr/bin/env bash
# run.sh - запуск veles одной командой на Linux/macOS.
#
#   ./run.sh
#
# Скрипт сам делает всё необходимое и безопасно перезапускается:
#   1) проверяет Python 3.11+;
#   2) создаёт виртуальное окружение .venv и ставит пакет (один раз);
#   3) копирует config.example.env в .env, если своего ещё нет
#      (адрес языковой модели подставляется в .env - или оставьте как есть,
#      база работает и без неё, как поиск с цитатами);
#   4) при первом запуске скачивает модель поиска (~450 МБ, дальше из кэша);
#   5) поднимает сервер: http://127.0.0.1:8400
#
# Документы добавляются после старта: .venv/bin/veles add файл.md
# или через очередь загрузки (docs/RUN-WINDOWS.md, раздел 2 - путь тот же).
set -euo pipefail

cd "$(dirname "$0")"

log() { echo "[veles] $*"; }

# 1. Python 3.11+
PY=""
for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then
        v="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        case "$v" in
            3.11|3.12|3.13) PY="$c"; break ;;
        esac
    fi
done
[[ -n "$PY" ]] || { echo "[veles] нужен Python 3.11-3.13 (python.org)"; exit 1; }
log "python: $($PY -V) ($(command -v "$PY"))"

# 2. Окружение и пакет
if [[ ! -x .venv/bin/python ]]; then
    log "создаю окружение .venv"
    "$PY" -m venv .venv
fi
PYBIN=.venv/bin/python
if ! "$PYBIN" -c "import veles, sentence_transformers, faiss" 2>/dev/null; then
    log "ставлю пакет и зависимости (несколько минут, один раз)"
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -e ".[st,faiss]"
fi

# 3. Файл настроек
if [[ ! -f .env ]]; then
    cp config.example.env .env
    log "создан .env - при желании подставьте адрес языковой модели (VELES_LLM_BASE_URL)"
    log "без неё база работает как поиск с цитатами; менять что-то прямо сейчас не нужно"
fi

# 4. Модель поиска: скачать заранее, чтобы первый вопрос не ждал
if ! "$PYBIN" -c "import os,sys; os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1');
from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" 2>/dev/null; then
    log "скачиваю модель поиска (~450 МБ, один раз)"
    "$PYBIN" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
fi

# 5. Сервер
HOST="${VELES_HOST:-127.0.0.1}"
PORT="${VELES_PORT:-8400}"
log "стартую сервер: http://${HOST}:${PORT} (остановка: Ctrl+C)"
exec "$PYBIN" -m veles serve --host "$HOST" --port "$PORT"
