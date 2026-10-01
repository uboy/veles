#!/usr/bin/env bash
# install.sh - установка veles из deploy-kit.
#
# Профили:
#   corporate - docker compose: фасад + Open WebUI (+ мост по --with-bridge).
#   personal  - venv + фасад на своей машине (Linux/macOS), WebUI по --with-webui.
#   lite      - только CLI (venv), без демонов и портов.
#
#   --demo    - демо в один клик: база сразу наполнена готовым мини-индексом
#               (5 фикшн-документов ООО «Ромашка»), после установки можно
#               сразу спрашивать. Профиль без --profile = lite.
#
# Примеры:
#   ./install.sh --demo --yes                    # демо на CLI (lite), без вопросов
#   ./install.sh --profile lite --dir ~/veles
#   ./install.sh --profile corporate --yes --with-bridge
#
# Ключи LLM спрашиваются при установке и пишутся только в локальный .env (chmod 600)
# - в архив они не попадают никогда.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ]; then
    die "нужен bash 4+, сейчас: ${BASH_VERSION:-нет}. Запустите через bash install.sh"
fi

REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)  # deploy/kit -> корень репозитория/архива

PROFILE="" IN_DIR="" DRY_RUN=0 ASSUME_YES=0 DEMO=0
WITH_WEBUI=0 WITH_BRIDGE=0 WITH_VOXFUSION=0
IDENTITY_HEADERS=0   # opt-in: режим headers вместо дефолтного jwt (corporate)

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --profile)          PROFILE="${2:-}"; shift 2 ;;
        --dir)              IN_DIR="${2:-}"; shift 2 ;;
        --demo)             DEMO=1; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        --yes|-y)           ASSUME_YES=1; shift ;;
        --with-webui)       WITH_WEBUI=1; shift ;;
        --with-bridge)      WITH_BRIDGE=1; shift ;;
        --with-voxfusion)   WITH_VOXFUSION=1; shift ;;
        --identity-headers) IDENTITY_HEADERS=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *) die "неизвестный аргумент: $1 (см. --help)" ;;
    esac
done

export VELES_ASSUME_YES="$ASSUME_YES"

# демо по умолчанию - lite (CLI, без docker и портов); явный --profile сильнее
if [ "$DEMO" = 1 ] && [ -z "$PROFILE" ]; then
    PROFILE=lite
    info "--demo: профиль не задан - ставлю lite (corporate: --profile corporate --demo)"
fi

if [ "$WITH_VOXFUSION" = 1 ]; then
    # честно: kit не ставит транскрипцию - VoxFusion отдельный сервис со своим репо
    warn "--with-voxfusion только помечает намерение в .env: транскрипция ставится отдельно
       (репозиторий VoxFusion, URL затем вписывается в VOXFUSION_API_URL). См. README, раздел Транскрипция"
fi

# ---------- выбор профиля ----------

if [ -z "$PROFILE" ]; then
    if [ ! -t 0 ]; then
        die "профиль не задан и stdin не терминал. Укажите: --profile corporate|personal|lite"
    fi
    cat <<'MENU'
Выберите профиль установки:
  1) corporate - docker compose: фасад + Open WebUI, необязательны мост и транскрипция.
                 Для сервера компании.
  2) personal  - venv + фасад на своей машине (Linux/macOS), WebUI по флагу.
                 Личное использование, без docker.
  3) lite      - только CLI в venv: veles ask/status, без демонов и портов.
MENU
    read -r -p "Профиль [1/2/3, пусто = 3]: " reply || die "нет ввода"
    case "$reply" in
        1|corporate) PROFILE=corporate ;;
        2|personal)  PROFILE=personal ;;
        ""|3|lite)   PROFILE=lite ;;
        *) die "не понял ответ: $reply" ;;
    esac
fi

case "$PROFILE" in
    corporate|personal|lite) ;;
    *) die "профиль: corporate|personal|lite, задано: $PROFILE" ;;
esac

# corporate ставит всё в каталог кита (compose, .env, данные): относительные пути
# bind-маунтов в docker-compose.yml считаются от каталога compose-файла
if [ "$PROFILE" = "corporate" ] && [ -n "$IN_DIR" ] && [ "$IN_DIR" != "$SCRIPT_DIR" ]; then
    warn "--dir для corporate не применяется: установка целиком в $SCRIPT_DIR"
fi
IN_DIR="${IN_DIR:-$SCRIPT_DIR}"
[ "$PROFILE" = "corporate" ] && IN_DIR="$SCRIPT_DIR"
mkdir -p "$IN_DIR"
IN_DIR=$(cd "$IN_DIR" && pwd)
ENV_FILE="$IN_DIR/.env"

info "профиль: $PROFILE; каталог установки: $IN_DIR"
[ "$DRY_RUN" = 1 ] && warn "dry-run: изменения на диск и в docker не вносятся"

# ---------- preflight ----------

if [ "$DEMO" = 1 ]; then
    for d in index canonical documents; do
        [ -d "$SCRIPT_DIR/demo/$d" ] || die "--demo: в ките нет demo/$d ($SCRIPT_DIR/demo) - кит собран без демо-контента"
    done
    ok "демо-контент: $(find "$SCRIPT_DIR/demo/documents" -name '*.md' | wc -l) документов, индекс $(du -sh "$SCRIPT_DIR/demo/index" 2>/dev/null | awk '{print $1}')"
fi

PY=""
if [ "$PROFILE" != "corporate" ]; then
    if PY=$(find_python_min 3.11); then
        ok "python: $PY ($("$PY" -V 2>&1))"
    else
        die "python 3.11+ не найден в PATH. Установите python3.11+ (пакет python3.11/python3) и повторите"
    fi
fi

if ! command -v curl >/dev/null 2>&1; then
    if [ "$PROFILE" = "lite" ]; then
        warn "curl не найден - verify-install.sh будет пропускать HTTP-проверки (для lite это некритично)"
    else
        die "curl не найден - нужен для проверки установки (verify-install.sh)"
    fi
fi

SECRET_MAKER=""
if command -v openssl >/dev/null 2>&1; then
    SECRET_MAKER="openssl"
    ok "openssl: есть (генерация секретов)"
elif PY=$(find_python_min 3.11); then
    SECRET_MAKER="$PY"
    warn "openssl не найден - секреты сгенерирует python"
else
    die "ни openssl, ни python - нечем генерировать секреты. Установите openssl"
fi

disk_needed_mb=1024
[ "$PROFILE" = "lite" ] || disk_needed_mb=5120
# демо в lite/personal тянет sentence-transformers (torch) в venv
if [ "$DEMO" = 1 ] && [ "$PROFILE" != "corporate" ]; then
    disk_needed_mb=4096
fi
avail=$(disk_free_mb "$IN_DIR")
if [ -n "$avail" ] && [ "$avail" -lt "$disk_needed_mb" ]; then
    die "мало места: свободно ${avail}МБ, нужно ≥${disk_needed_mb}МБ ($IN_DIR)"
fi
ok "диск: свободно ${avail:-?}МБ (нужно ≥${disk_needed_mb}МБ)"

if [ "$PROFILE" != "lite" ]; then
    ram=$(ram_total_mb)
    if [ -n "$ram" ] && [ "$ram" -gt 0 ] && [ "$ram" -lt 4096 ]; then
        die "мало памяти: ${ram}МБ, нужно ≥4096МБ (embedding-модель и WebUI)"
    fi
    [ -n "$ram" ] && [ "$ram" -gt 0 ] && ok "память: ${ram}МБ"
fi

FACADE_PORT=$(env_get "$ENV_FILE" VELES_FACADE_PORT); FACADE_PORT=${FACADE_PORT:-8400}
WEBUI_PORT=$(env_get "$ENV_FILE" VELES_WEBUI_PORT);   WEBUI_PORT=${WEBUI_PORT:-8300}

COMPOSE=""
if [ "$PROFILE" = "corporate" ]; then
    command -v docker >/dev/null 2>&1 || die "docker не найден. Установите docker + compose v2 и повторите"
    COMPOSE=$(docker compose version --short 2>/dev/null || true)
    [ -n "$COMPOSE" ] || die "docker compose v2 не найден. Нужен плагин docker compose (v2)"
    docker info >/dev/null 2>&1 || die "docker-демон не отвечает (docker info упал). Запустите docker и повторите"
    ok "docker compose: v$COMPOSE"
    # фасад наружу не публикуется (только внутри compose-сети), наружу - WebUI
    check_port_free "$WEBUI_PORT" "Open WebUI"
fi

if [ "$PROFILE" = "personal" ]; then
    check_port_free "$FACADE_PORT" "фасад veles"
    [ "$WITH_WEBUI" = 1 ] && check_port_free "$WEBUI_PORT" "Open WebUI"
fi

# ---------- конфигурация (.env) ----------

ask_var() { # ask_var <KEY> <prompt> <default> [secret] - не переспрашивает, если уже в .env
    local key="$1" prompt="$2" def="${3:-}" cur="" secret="${4:-0}"
    cur=$(env_get "$ENV_FILE" "$key")
    if [ -n "$cur" ] && [ "$cur" != "GENERATE" ]; then
        info "$key: оставляю существующее значение"
        return 0
    fi
    if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then
        env_set "$ENV_FILE" "$key" "$def"
        return 0
    fi
    # secret=1: ввод без эха - ключ не остаётся в scrollback/терминальном логе
    if [ "$secret" = "1" ]; then
        read -rs -p "$prompt: " reply || reply=""
        echo
    else
        read -r -p "$prompt${def:+ [$def]}: " reply || reply=""
    fi
    env_set "$ENV_FILE" "$key" "${reply:-$def}"
}

gen_or_reuse() { # gen_or_reuse <KEY> - GENERATE/пусто -> секрет
    local key="$1" cur=""
    cur=$(env_get "$ENV_FILE" "$key")
    if [ -n "$cur" ] && [ "$cur" != "GENERATE" ]; then return 0; fi
    secret=$(gen_secret "$PY") || die "не могу сгенерировать секрет для $key"
    env_set "$ENV_FILE" "$key" "$secret"
}

if [ "$DRY_RUN" = 1 ]; then
    info "dry-run: конфигурация записана не будет"
else
    if [ -f "$ENV_FILE" ]; then
        info "найден существующий .env - существующие значения сохраняются"
    fi

    env_set "$ENV_FILE" VELES_KIT_PROFILE "$PROFILE"
    env_set "$ENV_FILE" VELES_KIT_DEMO "$DEMO"
    env_set "$ENV_FILE" VELES_FACADE_PORT "$FACADE_PORT"
    env_set "$ENV_FILE" VELES_WEBUI_PORT "$WEBUI_PORT"
    env_set "$ENV_FILE" VELES_DATA "$IN_DIR/data"
    env_set "$ENV_FILE" VELES_WITH_WEBUI "$WITH_WEBUI"
    env_set "$ENV_FILE" VELES_WITH_BRIDGE "$WITH_BRIDGE"
    env_set "$ENV_FILE" VELES_WITH_VOXFUSION "$WITH_VOXFUSION"
    [ "$PROFILE" = "corporate" ] || env_set "$ENV_FILE" VELES_CORPUS_ROOT "$IN_DIR/corpus"

    # ревью владельца 29.09 (демо сервер): дефолт 127.0.0.1 рождает ссылки,
    # которые не открываются ни у кого, кроме самой машины установки. Для
    # corporate-демо дефолт = LAN-адрес хоста, и фасад публикуется на нём же
    # (override-файл ниже) - иначе ссылки из веб-чата ведут в никуда.
    public_default="http://127.0.0.1:$FACADE_PORT"
    if [ "$PROFILE" = "corporate" ] && [ "$DEMO" = 1 ]; then
        lan_ip=$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' | head -1)
        [ -n "$lan_ip" ] || lan_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        [ -n "$lan_ip" ] && public_default="http://$lan_ip:$FACADE_PORT"
    fi
    ask_var VELES_PUBLIC_URL "Публичный URL фасада (как его видят клиенты)" "$public_default"
    # K21: эта настройка = ОБЩИЕ группы (видны всем пользователям). corporate-дефолт all:
    # "personal" здесь делал личную группу владельца общей - пользователи видели весь корпус
    if [ "$PROFILE" = "corporate" ]; then groups_default="all"; else groups_default="personal"; fi
    ask_var VELES_ALLOWED_GROUPS "Общие группы доступа через запятую" "$groups_default"
    if [ "$PROFILE" = "corporate" ] && [ "$DEMO" = 1 ] && [ ! -f docker-compose.override.yml ]; then
        # фасад в corporate не публикуется (безопасность), но демо-ссылки
        # /v1/source должны открываться с машин клиентов - публикуем 8400
        # только в демо-режиме (токен+подпись защищают)
        cat > docker-compose.override.yml <<EOF
services:
  veles-facade:
    ports:
      - "$FACADE_PORT:$FACADE_PORT"
EOF
        ok "демо: фасад опубликован на :$FACADE_PORT (docker-compose.override.yml) - иначе ссылки источников не открываются"
    fi

    if [ "$PROFILE" = "corporate" ]; then
        # дефолт - jwt (подписанный заголовок, секрет генерится и отдаётся WebUI);
        # headers - только явный opt-in --identity-headers
        mode=$(env_get "$ENV_FILE" VELES_IDENTITY_MODE)
        if [ "$IDENTITY_HEADERS" = 1 ]; then
            mode=headers
            warn "headers-режим: личность приходит plain-заголовками. Заголовкам можно подделать личность при прямом доступе к фасаду в обход Open WebUI - нужен сетевой ACL (фасад закрыт от пользователей)."
        elif [ -z "$mode" ] || [ "$mode" = "static" ]; then
            mode=jwt
        fi
        env_set "$ENV_FILE" VELES_IDENTITY_MODE "$mode"
        if [ "$mode" = "jwt" ]; then
            gen_or_reuse VELES_IDENTITY_JWT_SECRET
            jwtsec=$(env_get "$ENV_FILE" VELES_IDENTITY_JWT_SECRET)
            env_set "$ENV_FILE" FORWARD_USER_INFO_HEADER_JWT_SECRET "$jwtsec"
        fi
        env_set "$ENV_FILE" ENABLE_FORWARD_USER_INFO_HEADERS "true"
        ask_var VELES_ADMIN_EMAILS "Админы (email через запятую, пусто = нет)" ""
    else
        env_set "$ENV_FILE" VELES_IDENTITY_MODE "${VELES_IDENTITY_MODE:-static}"
    fi

    # секреты: существующие не трогаем
    gen_or_reuse VELES_API_TOKEN
    gen_or_reuse VELES_LINK_SECRET
    if [ "$PROFILE" != "lite" ]; then
        gen_or_reuse WEBUI_SECRET_KEY
    fi

    # ключ LLM: только ввод пользователя, в архив не встраивается
    if [ "$PROFILE" != "lite" ]; then
        ask_var VELES_LLM_BASE_URL "Базовый URL LLM (OpenAI-совместимый)" "https://api.z.ai/api/coding/paas/v4"
        ask_var VELES_LLM_MODEL "Модель LLM для синтеза ответов" "glm-5.3-flash"
        cur=$(env_get "$ENV_FILE" VELES_LLM_API_KEY)
        if [ -z "$cur" ] && [ -t 0 ] && [ "$ASSUME_YES" != 1 ]; then
            read -rs -p "Ключ LLM (пишется только в .env, пусто = пропустить): " llm_key || llm_key=""
            echo
            env_set "$ENV_FILE" VELES_LLM_API_KEY "$llm_key"
        elif [ -z "$cur" ]; then
            warn "ключ LLM не задан (VELES_LLM_API_KEY) - фасад будет отвечать фрагментами без синтеза; впишите ключ в $ENV_FILE"
        fi
    fi

    if [ "$WITH_BRIDGE" = 1 ]; then
        ask_var TELEGRAM_BOT_TOKEN "Токен Telegram-бота (пусто = настроите потом)" "" 1
        ask_var TG_PROXY "Прокси для Telegram (http://user:pass@host:port, пусто = напрямую)" ""
        env_set "$ENV_FILE" PAIR_MODE "1"
    fi

    chmod 600 "$ENV_FILE"
    ok ".env записан: $ENV_FILE (права 600)"

    # fail-closed: corporate без аутентификации не поднимается
    if [ "$PROFILE" = "corporate" ]; then
        mode=$(env_get "$ENV_FILE" VELES_IDENTITY_MODE)
        token=$(env_get "$ENV_FILE" VELES_API_TOKEN)
        case "$mode" in
            headers|jwt) ;;
            *) die "fail-closed: corporate требует VELES_IDENTITY_MODE=headers|jwt, сейчас: '${mode:-пусто}'. static для корпоративного контура запрещён"
        esac
        [ -n "$token" ] || die "fail-closed: пустой VELES_API_TOKEN - контур без аутентификации подниматься не будет"
        ok "fail-closed: identity-режим $mode, bearer-токен задан"
    fi
fi

# ---------- демо-контент (K15) ----------

copy_demo_content() {
    local data_dir="$IN_DIR/data"
    mkdir -p "$data_dir/canonical"
    if [ -f "$data_dir/index/meta.json" ]; then
        warn "существующий индекс сохранён - демо-индекс не копировался ($data_dir/index)"
    else
        # ревью K15 P1: mkdir обязателен - cp -a src/. dst/ без созданного dst
        # падает на coreutils 8.32 (Ubuntu 22.04 LTS), на 9.4 (24.04) работает
        mkdir -p "$data_dir/index"
        cp -a "$SCRIPT_DIR/demo/index/." "$data_dir/index/"
        ok "демо-индекс: $data_dir/index ($(du -sh "$SCRIPT_DIR/demo/index" | awk '{print $1}'))"
    fi
    if [ -n "$(ls -A "$data_dir/canonical" 2>/dev/null)" ]; then
        warn "канон не пуст - демо-канон не копировался ($data_dir/canonical)"
    else
        cp -a "$SCRIPT_DIR/demo/canonical/." "$data_dir/canonical/"
        ok "демо-канон: $data_dir/canonical"
    fi
    if [ "$PROFILE" = "corporate" ]; then
        info "corporate: индекс и канон уедут в контейнер через bind ./data; корпус в контейнер не монтируется"
    else
        mkdir -p "$IN_DIR/corpus/demo"
        cp -a "$SCRIPT_DIR/demo/documents/." "$IN_DIR/corpus/demo/"
        ok "демо-документы (оригиналы): $IN_DIR/corpus/demo"
    fi
}

# прогрев embedding-модели: для эмбеддинга ВОПРОСА модель нужна и локально;
# скачаем на установке (~130 МБ, HuggingFace, кеш ~/.cache/huggingface),
# чтобы первый демо-вопрос не упал и не ждал скачивания
warmup_embedder() {
    info "прогрев embedding-модели: скачиваю с HuggingFace (~130 МБ, один раз, кеш ~/.cache/huggingface)"
    if "$IN_DIR/.venv/bin/python" -c "from sentence_transformers import SentenceTransformer; from veles.config import EMBED_MODEL; SentenceTransformer(EMBED_MODEL)" >/dev/null 2>&1; then
        ok "embedding-модель готова локально - вопросы пойдут без сети"
    else
        warn "прогрев не удался (нет сети до huggingface.co?) - модель скачается при первом вопросе"
    fi
}

if [ "$DEMO" = 1 ] && [ "$DRY_RUN" != 1 ]; then
    copy_demo_content
fi

# ---------- установка ----------

install_lite() {
    info "создаю venv: $IN_DIR/.venv"
    "$PY" -m venv "$IN_DIR/.venv"
    "$IN_DIR/.venv/bin/pip" install -q -U pip
    if [ "$DEMO" = 1 ]; then
        # демо-вопросу нужен эмбеддер (sentence-transformers/torch);
        # torch ставим CPU-сборкой: PyPI-вариант тянет CUDA-библиотеки (~5 ГБ),
        # которые на чистой машине демо не нужны. Не скачался - PyPI fallback.
        info "ставлю veles + embedding-зависимости для демо (CPU-torch + sentence-transformers)"
        if ! "$IN_DIR/.venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu; then
            warn "CPU-torch с download.pytorch.org не скачался - torch придёт из PyPI (с CUDA, крупнее на ~5 ГБ)"
        fi
        "$IN_DIR/.venv/bin/pip" install -q -e "$REPO_DIR[st]"
    else
        "$IN_DIR/.venv/bin/pip" install -q -e "$REPO_DIR"
    fi
    ok "CLI установлен: $IN_DIR/.venv/bin/veles"
    # обёртка CLI с окружением установки: без неё `veles ask` не видит индекс -
    # VELES_DATA живёт в .env, который CLI сам не читает (поймано приёмкой K15
    # в чистом контейнере; verify-install проходил, потому что грузит .env сам)
    cat > "$IN_DIR/veles" <<WRAPPER
#!/usr/bin/env bash
# CLI veles с окружением установки (env из .env рядом): ./veles ask|add|status
set -euo pipefail
here=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)
set -a; source "\$here/.env"; set +a
exec "\$here/.venv/bin/veles" "\$@"
WRAPPER
    chmod +x "$IN_DIR/veles"
    ok "CLI с окружением: $IN_DIR/veles (ask|add|status)"
    if [ "$DEMO" = 1 ]; then
        warmup_embedder
    fi
}

install_personal() {
    install_lite
    info "ставлю embedding/facade-зависимости (torch тянется дольше всего)"
    "$IN_DIR/.venv/bin/pip" install -q -e "$REPO_DIR[st,faiss]"
    # фасад на 127.0.0.1; с docker-WebUI - на всех интерфейсах (контейнер ходит через host)
    hostbind=127.0.0.1
    if [ "$WITH_WEBUI" = 1 ] && command -v docker >/dev/null 2>&1; then
        hostbind=0.0.0.0
    fi
    unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    if command -v systemctl >/dev/null 2>&1 && systemctl --user show veles-facade >/dev/null 2>&1; then
        mkdir -p "$unit_dir"
        cat > "$unit_dir/veles-facade.service" <<UNIT
[Unit]
Description=Veles Knowledge API facade
After=network-online.target

[Service]
WorkingDirectory=$REPO_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$IN_DIR/.venv/bin/veles serve --host $hostbind --port $FACADE_PORT
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
UNIT
        systemctl --user daemon-reload
        systemctl --user enable --now veles-facade
        ok "systemd user-юнит veles-facade запущен (автостарт при входе; для фона без входа: sudo loginctl enable-linger $USER)"
    else
        cat > "$IN_DIR/run-facade.sh" <<RUN
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
set -a; source "$ENV_FILE"; set +a
exec "$IN_DIR/.venv/bin/veles" serve --host $hostbind --port $FACADE_PORT
RUN
        chmod +x "$IN_DIR/run-facade.sh"
        info "systemd user-сессии нет - запускайте фасад так: $IN_DIR/run-facade.sh"
    fi
    if [ "$WITH_WEBUI" = 1 ]; then
        if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
            mkdir -p "$IN_DIR/data/webui"
            docker rm -f veles-webui >/dev/null 2>&1 || true
            docker run -d --name veles-webui --restart unless-stopped \
                -p "$WEBUI_PORT:8080" --add-host=host.docker.internal:host-gateway \
                --env-file "$ENV_FILE" -v "$IN_DIR/data/webui:/app/backend/data" \
                ghcr.io/open-webui/open-webui:main >/dev/null
            ok "Open WebUI: контейнер veles-webui на порту $WEBUI_PORT"
        else
            info "docker нет - ставлю open-webui в venv (большая загрузка)"
            "$IN_DIR/.venv/bin/pip" install -q open-webui
            cat > "$IN_DIR/run-webui.sh" <<RUN
#!/usr/bin/env bash
set -euo pipefail
exec "$IN_DIR/.venv/bin/open-webui" serve --port $WEBUI_PORT
RUN
            chmod +x "$IN_DIR/run-webui.sh"
            info "WebUI запуск: $IN_DIR/run-webui.sh"
        fi
    fi
}

install_corporate() {
    cd "$SCRIPT_DIR"
    compose_profiles=()
    [ "$WITH_BRIDGE" = 1 ] && compose_profiles+=(--profile bridge)
    info "docker compose up -d (образ фасада собирается при первой установке)"
    docker compose -f docker-compose.yml ${compose_profiles[@]+"${compose_profiles[@]}"} up -d --build
    ok "стек поднят: veles-facade (внутри compose-сети), open-webui :$WEBUI_PORT"
}

if [ "$DRY_RUN" = 1 ]; then
    info "dry-run завершён: preflight пройден, дальше было бы: .env + $PROFILE"
    exit 0
fi

printf 'veles deploy-kit\nпрофиль: %s\nустановлен: %s\n' "$PROFILE" "$(date -Iseconds)" > "$IN_DIR/.veles-kit"

case "$PROFILE" in
    lite)      install_lite ;;
    personal)  install_personal ;;
    corporate) install_corporate ;;
esac

# ---------- итог ----------

echo
info "Установка завершена. Самопроверка:"
if "$SCRIPT_DIR/verify-install.sh" --dir "$IN_DIR"; then
    echo
    ok "Проверка пройдена."
else
    echo
    warn "Часть проверок не прошла - смотрите таблицу выше. Установка на месте; частые причины: не задан ключ LLM, не загружен индекс базы, WebUI ещё стартует (перезапустите verify-install.sh через минуту)."
fi

echo
info "Управление:"
if [ "$PROFILE" = "corporate" ]; then
    info "  статус/логи:  cd $SCRIPT_DIR && docker compose ps / docker compose logs -f veles-facade"
    info "  остановить:   cd $SCRIPT_DIR && docker compose down"
    info "  удалить всё:  $SCRIPT_DIR/uninstall.sh --purge-data"
    info "  порты: WebUI на ${WEBUI_PORT} (наружу), фасад - только внутри compose-сети:8400;"
    info "         VELES_FACADE_PORT на corporate не меняет порт контейнера (зашит в compose)"
elif [ "$PROFILE" = "personal" ]; then
    info "  статус:       systemctl --user status veles-facade (или $IN_DIR/run-facade.sh)"
    info "  удалить всё:  $SCRIPT_DIR/uninstall.sh --dir $IN_DIR --purge-data"
else
    info "  CLI:          $IN_DIR/veles status"
    info "  вопросы:      $IN_DIR/veles ask \"вопрос\""
    info "  удалить всё:  $SCRIPT_DIR/uninstall.sh --dir $IN_DIR --purge-data"
fi

if [ "$DEMO" = 1 ]; then
    echo
    info "Демо готово: база наполнена 5 документами (ООО «Ромашка», фикшн). Проверенные вопросы:"
    sed 's/^/    /' "$SCRIPT_DIR/demo/questions.txt"
    if [ "$PROFILE" = "lite" ]; then
        info "Как спросить:   $IN_DIR/veles ask \"вопрос\""
        info "Свой документ:  $IN_DIR/veles add файл.txt --title \"Название\" - и он сразу в поиске"
    elif [ "$PROFILE" = "personal" ]; then
        public_url=$(env_get "$ENV_FILE" VELES_PUBLIC_URL)
        info "Как спросить:   веб-чат ${public_url} (или curl POST ${public_url}/v1/knowledge/query)"
        info "Свой документ:  curl -H \"Authorization: Bearer \$(grep '^VELES_API_TOKEN=' $ENV_FILE | cut -d= -f2)\" -F file=@документ.pdf ${public_url}/v1/documents"
    else
        info "Как спросить:   Open WebUI на http://<адрес сервера>:${WEBUI_PORT} (модель veles-knowledge)"
        info "Свой документ:  через чат Telegram-бота (--with-bridge, перетащить файл) или POST /v1/documents изнутри контура"
    fi
    info "Удалить демо целиком: $SCRIPT_DIR/uninstall.sh --dir $IN_DIR --purge-data --yes"
fi
