#!/usr/bin/env bash
# uninstall.sh - снятие установки veles из deploy-kit.
#
# По умолчанию останавливает сервисы профиля, но сохраняет данные и .env
# (VELES_DATA, tracker, WebUI-данные) - повторная install.sh поднимет как было.
#
#   --purge-data      удалить и данные: печатает перечень путей и размеров.
#                     Без --yes - двойное подтверждение.
#   --dir PATH        каталог установки (для lite/personal; corporate - каталог кита)
#   --profile P       не читать профиль из .env, а взять принудительно
#   --yes             не спрашивать подтверждений
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ]; then
    die "нужен bash 4+, сейчас: ${BASH_VERSION:-нет}. Запустите через bash uninstall.sh"
fi

PURGE=0 ASSUME_YES=0 PROFILE="" IN_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --purge-data) PURGE=1; shift ;;
        --yes|-y)     ASSUME_YES=1; shift ;;
        --profile)    PROFILE="${2:-}"; shift 2 ;;
        --dir)        IN_DIR="${2:-}"; shift 2 ;;
        -h|--help)    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "неизвестный аргумент: $1 (см. --help)" ;;
    esac
done
export VELES_ASSUME_YES="$ASSUME_YES"

IN_DIR="${IN_DIR:-$SCRIPT_DIR}"
# mkdir ПОСЛЕ валидации: опечатка в --dir не должна создавать пустой каталог
case "$IN_DIR" in
    /|"$HOME") die "--dir $IN_DIR запрещён: не стираю корень или домашний каталог" ;;
esac
IN_DIR=$(cd "$IN_DIR" 2>/dev/null && pwd) || die "каталог --dir $IN_DIR не существует"
ENV_FILE="$IN_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    load_env "$ENV_FILE"
fi
PROFILE="${PROFILE:-${VELES_KIT_PROFILE:-}}"
if [ -z "$PROFILE" ]; then
    die "профиль не определён: нет $ENV_FILE и не задан --profile corporate|personal|lite"
fi
case "$PROFILE" in
    corporate|personal|lite) ;;
    *) die "профиль: corporate|personal|lite, в .env: $PROFILE" ;;
esac
FACADE_PORT="${VELES_FACADE_PORT:-8400}"
WEBUI_PORT="${VELES_WEBUI_PORT:-8300}"

info "профиль: $PROFILE; каталог установки: $IN_DIR"

# ---------- остановка ----------

stop_corporate() {
    command -v docker >/dev/null 2>&1 || { warn "docker нет - пропускаю остановку контейнеров"; return 0; }
    cd "$SCRIPT_DIR"
    local profs=()
    [ "${VELES_WITH_BRIDGE:-0}" = "1" ] && profs+=(--profile bridge)
    docker compose -f docker-compose.yml ${profs[@]+"${profs[@]}"} down --remove-orphans || \
        warn "docker compose down вернул ошибку (стек, возможно, уже снят)"
    ok "compose-стек остановлен и удалён (образы остались; при --purge-data удалятся каталоги данных)"
}

stop_personal() {
    if command -v systemctl >/dev/null 2>&1 && systemctl --user show veles-facade >/dev/null 2>&1; then
        systemctl --user disable --now veles-facade 2>/dev/null || true
        rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/veles-facade.service"
        systemctl --user daemon-reload 2>/dev/null || true
        ok "systemd user-юнит veles-facade остановлен и удалён"
    fi
    if docker info >/dev/null 2>&1; then
        docker rm -f veles-webui >/dev/null 2>&1 && ok "контейнер veles-webui удалён" || true
    fi
    # процессы фасада, запущенные run-facade.sh из этого каталога
    local pids
    pids=$(pgrep -f "$IN_DIR/.venv/bin/veles serve" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
        sleep 1
        ok "остановлены процессы фасада: $pids"
    fi
}

stop_lite() {
    local pids
    pids=$(pgrep -f "$IN_DIR/.venv/bin/veles" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
        warn "были запущены CLI-процессы - остановлены: $pids"
    fi
    info "lite: демонов нет, останавливать нечего"
}

case "$PROFILE" in
    corporate) stop_corporate ;;
    personal)  stop_personal ;;
    lite)      stop_lite ;;
esac

# ---------- данные ----------

if [ "$PURGE" = 1 ]; then
    # распакованный архив снимается целиком: маркер - COMMIT.txt с полем kit_version
    # (кладёт make-kit.sh). Чужой каталог со случайным COMMIT.txt не считается китом,
    # git-чекаут репозитория (COMMIT.txt не трекается) не трогается никогда.
    KIT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
    IS_KIT_ROOT=0
    if [ "$KIT_ROOT" != "/" ] && [ -f "$KIT_ROOT/COMMIT.txt" ] \
        && grep -q '^kit_version:' "$KIT_ROOT/COMMIT.txt" 2>/dev/null; then
        IS_KIT_ROOT=1
    fi

    echo
    info "Будут удалены (--purge-data):"
    local_total=0
    for p in "$IN_DIR/data" "$IN_DIR/.venv" "$ENV_FILE" "$IN_DIR/corpus"; do
        [ -e "$p" ] || continue
        printf '  %-12s %s\n' "$(human_size "$p")" "$p"
        local_total=1
    done
    if [ "$IS_KIT_ROOT" = 1 ]; then
        printf '  %-12s %s (весь распакованный кит)\n' "$(human_size "$KIT_ROOT")" "$KIT_ROOT"
        local_total=1
    fi
    [ "$local_total" = 0 ] && info "  (нечего удалять - уже чисто)"

    if [ "$ASSUME_YES" != 1 ]; then
        echo
        confirm "Удалить перечисленное? Данные базы, .env с ключами и venv будут потеряны." || die "отменено пользователем"
        confirm "Подтвердите второй раз: удалить безвозвратно?" || die "отменено пользователем"
    fi

    rm -rf "$IN_DIR/data" "$IN_DIR/.venv" "$IN_DIR/corpus"
    rm -f "$ENV_FILE"
    if [ "$IS_KIT_ROOT" = 1 ]; then
        info "снимаю распакованный кит: $KIT_ROOT"
        find "$KIT_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
        rmdir "$KIT_ROOT" 2>/dev/null || warn "каталог $KIT_ROOT не пуст - посмотрите глазами"
    else
        info "исходники $KIT_ROOT не тронуты (это не распакованный архив make-kit)"
    fi
    ok "данные удалены"
else
    info "без --purge-data: данные и .env сохранены ($IN_DIR/data, $ENV_FILE)"
    info "полное удаление: $SCRIPT_DIR/uninstall.sh --dir $IN_DIR --purge-data"
fi

# ---------- проверка результата ----------

echo
info "Проверка после снятия:"
leftover=0
for p in "$ENV_FILE" "$IN_DIR/.venv" "$IN_DIR/data"; do
    if [ -e "$p" ]; then
        warn "  остался: $p"
        leftover=1
    fi
done
[ "$leftover" = 0 ] && ok "  файлов установки не осталось: $IN_DIR"

port_bad=0
for port in "$FACADE_PORT" "$WEBUI_PORT"; do
    [ "$PROFILE" = "lite" ] && break
    if port_in_use "$port"; then
        warn "  порт $port всё ещё занят (наш ли это процесс - проверьте: ss -tlnp | grep $port)"
        port_bad=1
    fi
done
[ "$port_bad" = 0 ] && [ "$PROFILE" != "lite" ] && ok "  порты $FACADE_PORT/$WEBUI_PORT свободны"
[ "$PROFILE" = "lite" ] && ok "  порты не использовались"

[ "$leftover" = 0 ] && [ "$port_bad" = 0 ]
