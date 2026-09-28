#!/usr/bin/env bash
# verify-install.sh - самопроверка установки veles deploy-kit.
#
#   ./verify-install.sh [--dir PATH] [--profile P]
#
# Таблица OK/FAIL/SKIP по компонентам профиля; код 0 - нет FAIL.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

IN_DIR="" PROFILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dir)     IN_DIR="${2:-}"; shift 2 ;;
        --profile) PROFILE="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "неизвестный аргумент: $1" ;;
    esac
done

IN_DIR="${IN_DIR:-$SCRIPT_DIR}"
IN_DIR=$(cd "$IN_DIR" && pwd)
ENV_FILE="$IN_DIR/.env"

if [ -f "$ENV_FILE" ]; then load_env "$ENV_FILE"; fi
PROFILE="${PROFILE:-${VELES_KIT_PROFILE:-}}"
[ -n "$PROFILE" ] || die "профиль не определён: нет $ENV_FILE, задайте --profile"
case "$PROFILE" in corporate|personal|lite) ;; *) die "неизвестный профиль: $PROFILE" ;; esac

FACADE_PORT="${VELES_FACADE_PORT:-8400}"
WEBUI_PORT="${VELES_WEBUI_PORT:-8300}"
VELES_ASSUME_YES="${VELES_ASSUME_YES:-0}"

failures=0
report() { # report OK|FAIL|SKIP <что> <детали>
    local status="$1" what="$2" details="${3:-}"
    case "$status" in
        OK)   printf '%s[OK]%s   %-28s %s\n'   "$C_OK"   "$C_OFF" "$what" "$details" ;;
        FAIL) printf '%s[FAIL]%s %-28s %s\n'   "$C_ERR"  "$C_OFF" "$what" "$details"; failures=$((failures + 1)) ;;
        SKIP) printf '%s[SKIP]%s %-28s %s\n'   "$C_WARN" "$C_OFF" "$what" "$details" ;;
    esac
}

echo "Проверка установки veles: профиль $PROFILE ($IN_DIR)"
echo "----------------------------------------------------------------"

VENV="$IN_DIR/.venv"
PY_BIN="$VENV/bin/python"
VELES_BIN="$VENV/bin/veles"

# --- базовое ---
if [ -f "$ENV_FILE" ]; then
    report OK ".env" "$ENV_FILE (600: $(stat -c '%a' "$ENV_FILE" 2>/dev/null || stat -f '%Lp' "$ENV_FILE" 2>/dev/null || echo '?'))"
else
    report FAIL ".env" "нет файла $ENV_FILE"
fi

if [ -x "$VELES_BIN" ]; then
    report OK "CLI" "$VELES_BIN"
else
    report FAIL "CLI" "нет $VELES_BIN"
fi

if [ "$PROFILE" != "lite" ]; then
    if [ -n "$(env_get "$ENV_FILE" VELES_LLM_API_KEY)" ]; then
        report OK "ключ LLM" "задан в .env"
    else
        report SKIP "ключ LLM" "не задан - синтез ответов выключен (впишите VELES_LLM_API_KEY в .env)"
    fi
fi

# --- lite: CLI отвечает ---
if [ "$PROFILE" = "lite" ]; then
    if [ -x "$VELES_BIN" ] && "$VELES_BIN" status >/tmp/veles-verify-status.$$ 2>&1; then
        report OK "CLI: veles status" "$(head -1 /tmp/veles-verify-status.$$)"
    else
        report FAIL "CLI: veles status" "$(tail -1 /tmp/veles-verify-status.$$ 2>/dev/null || echo 'не выполнен')"
    fi
    rm -f /tmp/veles-verify-status.$$
fi

# --- фасад ---
if [ "$PROFILE" != "lite" ]; then
    if [ "$PROFILE" = "corporate" ]; then
        # corporate: фасад не публикуется наружу - проверяем изнутри compose-сети
        facade_code() { # facade_code <path> <auth:0|1> - HTTP-код из контейнера фасада
            (cd "$SCRIPT_DIR" && docker compose -f docker-compose.yml exec -T veles-facade python - "$1" "$2" <<'PYEOF'
import os, sys, urllib.request
path, auth = sys.argv[1], sys.argv[2] == "1"
req = urllib.request.Request("http://127.0.0.1:8400" + path)
if auth:
    req.add_header("Authorization", "Bearer " + os.environ.get("VELES_API_TOKEN", ""))
try:
    print(urllib.request.urlopen(req, timeout=5).status)
except Exception as e:
    print(getattr(e, "code", 0))
PYEOF
            ) 2>/dev/null
        }
        facade_where="внутри compose-сети"
        hz=$(facade_code /healthz 0)
        token=$(env_get "$ENV_FILE" VELES_API_TOKEN)
        models=$(facade_code /v1/models 1)
        models_body=$( (cd "$SCRIPT_DIR" && docker compose -f docker-compose.yml exec -T veles-facade \
            python -c "import os,urllib.request; r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8400/v1/models', headers={'Authorization':'Bearer '+os.environ.get('VELES_API_TOKEN','')}), timeout=5); print(r.read().decode())" 2>/dev/null) || true)
    else
        base="http://127.0.0.1:$FACADE_PORT"
        facade_where="$base"
        hz=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$base/healthz" 2>/dev/null || echo 000)
        token=$(env_get "$ENV_FILE" VELES_API_TOKEN)
        models=$(curl -s -o /tmp/veles-verify-models.$$ -w '%{http_code}' --max-time 10 \
            -H "Authorization: Bearer $token" "$base/v1/models" 2>/dev/null || echo 000)
        models_body=$(cat /tmp/veles-verify-models.$$ 2>/dev/null || true)
        rm -f /tmp/veles-verify-models.$$
    fi

    if [ "$hz" = "200" ]; then
        report OK "фасад /healthz" "$facade_where"
    else
        report FAIL "фасад /healthz" "код ${hz:-000} ($facade_where)"
    fi
    if [ "$models" = "200" ] && printf '%s' "$models_body" | grep -q "veles-knowledge"; then
        report OK "фасад /v1/models" "bearer принят, veles-knowledge в списке"
    else
        report FAIL "фасад /v1/models" "код ${models:-000} (bearer/token или фасад не поднят)"
    fi

    if [ -f "$IN_DIR/data/index/meta.json" ]; then
        report OK "индекс базы" "$IN_DIR/data/index"
    else
        report SKIP "индекс базы" "не найден - наполните базу (README: «Наполнение базы»)"
    fi
fi

# --- Open WebUI ---
if [ "$PROFILE" = "corporate" ] || { [ "$PROFILE" = "personal" ] && [ "${VELES_WITH_WEBUI:-0}" = "1" ]; }; then
    if http_ok "http://127.0.0.1:$WEBUI_PORT/health"; then
        report OK "Open WebUI /health" "http://127.0.0.1:$WEBUI_PORT"
    else
        report FAIL "Open WebUI /health" "порт $WEBUI_PORT (первый старт тянет образ и миграции - повторите через минуту)"
    fi
fi

# --- Telegram-мост ---
if [ "${VELES_WITH_BRIDGE:-0}" = "1" ] || docker ps --format '{{.Names}}' 2>/dev/null | grep -q veles-telegram-bridge; then
    tg_token=$(env_get "$ENV_FILE" TELEGRAM_BOT_TOKEN)
    if [ -z "$tg_token" ]; then
        report SKIP "мост getMe" "TELEGRAM_BOT_TOKEN не задан"
    else
        proxy=$(env_get "$ENV_FILE" TG_PROXY)
        proxy_args=()
        [ -n "$proxy" ] && proxy_args=(--proxy "$proxy")
        me=$(curl -s --max-time 15 ${proxy_args[@]+"${proxy_args[@]}"} \
            "https://api.telegram.org/bot$tg_token/getMe" 2>/dev/null || echo '')
        if printf '%s' "$me" | grep -q '"ok":true'; then
            report OK "мост getMe" "$(printf '%s' "$me" | grep -o '"username":"[^"]*"' || echo 'токен принят')"
        else
            report FAIL "мост getMe" "Telegram недоступен или токен неверен${proxy:+ (прокси: $proxy)}"
        fi
    fi
fi

echo "----------------------------------------------------------------"
if [ "$failures" -eq 0 ]; then
    echo "${C_OK}Итог: FAIL нет.${C_OFF} Компоненты профиля $PROFILE работают (SKIP = осознанно не настроено)."
    exit 0
else
    echo "${C_ERR}Итог: FAIL: $failures.${C_OFF} Смотрите строки FAIL выше."
    exit 1
fi
