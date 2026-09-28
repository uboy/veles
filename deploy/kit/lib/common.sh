# lib/common.sh - общие функции deploy-kit veles.
# Подключается из install.sh / uninstall.sh / verify-install.sh:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
# Каждый скрипт сам ставит set -euo pipefail.

# ---------- вывод ----------

if [ -t 1 ]; then
    C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'
    C_INFO=$'\033[36m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
    C_OK=""; C_WARN=""; C_ERR=""; C_INFO=""; C_DIM=""; C_OFF=""
fi

info()  { printf '%s[veles]%s %s\n' "$C_INFO" "$C_OFF" "$*"; }
ok()    { printf '%s[ok]%s %s\n'    "$C_OK"   "$C_OFF" "$*"; }
warn()  { printf '%s[внимание]%s %s\n' "$C_WARN" "$C_OFF" "$*"; }
err()   { printf '%s[ошибка]%s %s\n' "$C_ERR"  "$C_OFF" "$*" >&2; }

die() { err "$*"; exit 1; }

# ---------- python ----------

# find_python_min 3.11 - печатает путь к первому подходящему интерпретатору
# (>= указанной версии), иначе код 1. Уважает VELES_PYTHON как подсказку.
find_python_min() {
    local min="${1:-3.11}" c
    local candidates=()
    [ -n "${VELES_PYTHON:-}" ] && candidates+=("$VELES_PYTHON")
    candidates+=(python3.13 python3.12 python3.11 python3 python)
    # ${arr[@]+...} - совместимость с bash 4.0-4.3 при пустом массиве (set -u)
    for c in ${candidates[@]+"${candidates[@]}"}; do
        command -v "$c" >/dev/null 2>&1 || continue
        if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            command -v "$c"
            return 0
        fi
    done
    return 1
}

# ---------- порты ----------

# port_in_use 8400 - 0 если порт кто-то слушает.
port_in_use() {
    local port="$1" host="${2:-127.0.0.1}"
    (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null || return 1
    exec 3>&- 3<&- || true
    return 0
}

# check_port_free 8400 "фасад" - die, если занято.
check_port_free() {
    local port="$1" what="$2"
    if port_in_use "$port"; then
        die "порт $port занят (нужен для: $what). Освободите порт или задайте другой в .env"
    fi
}

# ---------- диск и память ----------

# disk_free_mb <каталог>
disk_free_mb() {
    df -Pm "$1" 2>/dev/null | awk 'NR==2 {print $4}'
}

# ram_total_mb - 0 если определить не удалось (проверка тогда пропускается).
ram_total_mb() {
    if command -v free >/dev/null 2>&1; then
        free -m 2>/dev/null | awk '/^Mem:/ {print $2; exit}'
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.memsize 2>/dev/null | awk '{print int($1 / 1024 / 1024)}'
    else
        echo 0
    fi
}

# ---------- секреты ----------

# gen_secret [python] - 64 hex-символа. openssl, иначе python secrets.
gen_secret() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    elif [ -n "${1:-}" ]; then
        "$1" -c 'import secrets; print(secrets.token_hex(32))'
    else
        return 1
    fi
}

# ---------- .env ----------

# env_get <файл> <KEY> - значение без кавычек, пусто если нет.
env_get() {
    local file="$1" key="$2" line value
    [ -f "$file" ] || return 0
    line=$(grep -m1 -E "^${key}=" "$file" 2>/dev/null || true)
    [ -n "$line" ] || return 0
    value="${line#*=}"
    value="${value%\"}"; value="${value#\"}"
    printf '%s' "$value"
}

# env_has <файл> <KEY> - 0 если KEY есть и не пуст.
env_has() {
    [ -n "$(env_get "$1" "$2")" ]
}

# env_set <файл> <KEY> <VALUE> - записать/заменить строку KEY=VALUE.
# Без sed в replacement: '&','\','|' в значении sed раскрывает как спецсимволы
# и тихо портит секрет (ревью P2-1) - перезапись через grep -v + mv.
env_set() {
    local file="$1" key="$2" value="$3"
    touch "$file"
    chmod 600 "$file" 2>/dev/null || true
    if grep -qE "^${key}=" "$file" 2>/dev/null; then
        grep -vE "^${key}=" "$file" > "${file}.tmp" || true
        printf '%s=%s\n' "$key" "$value" >> "${file}.tmp"
        mv "${file}.tmp" "$file"
        chmod 600 "$file" 2>/dev/null || true
    else
        printf '%s=%s\n' "$key" "$value" >> "$file"
    fi
}

# load_env <файл> - экспортировать все KEY=VALUE (без eval, значения дословно).
load_env() {
    local file="$1" line key value
    [ -f "$file" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|\#*) continue ;; esac
        key="${line%%=*}"
        value="${line#*=}"
        value="${value%\"}"; value="${value#\"}"
        [ "$key" = "$line" ] && continue
        export "$key=$value"
    done < "$file"
}

# ---------- интерактив ----------

# confirm <вопрос> - 0 при "y". Уважает VELES_ASSUME_YES=1.
confirm() {
    [ "${VELES_ASSUME_YES:-0}" = "1" ] && return 0
    local reply
    read -r -p "$1 [y/N] " reply || return 1
    case "$reply" in y|Y|да|Д|yes|YES) return 0 ;; *) return 1 ;; esac
}

# ---------- прочее ----------

# human_size <путь> - размер каталога/файла читаемо.
human_size() {
    du -sh "$1" 2>/dev/null | awk '{print $1}'
}

# http_ok <url> [curl-extra-args...] - 0 если HTTP код 2xx/3xx.
http_ok() {
    local url="$1"; shift
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$@" "$url" 2>/dev/null || echo 000)
    case "$code" in 2*|3*) return 0 ;; *) return 1 ;; esac
}
