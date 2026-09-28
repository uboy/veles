#!/usr/bin/env bash
# make-kit.sh - сборка архива deploy-kit из git HEAD.
#
#   ./make-kit.sh            -> scratch/veles-deploy-kit-<версия>.tar.gz
#
# Внутри: весь репозиторий из git archive HEAD (без .git, данных, venv, секретов -
# их в git и нет) + COMMIT.txt (версия, sha, дата) + исполняемые биты скриптов.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

command -v git >/dev/null 2>&1 || { echo "git не найден" >&2; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "это не git-репозиторий" >&2; exit 1; }

VER=$(git describe --tags --always --dirty 2>/dev/null || echo unknown)
SHA=$(git rev-parse HEAD)
DATE=$(git log -1 --format=%cI)
SUBJ=$(git log -1 --format=%s)

# архив собирается из HEAD: незакоммиченные правки в кит не попадают
if [ -n "$(git status --porcelain)" ]; then
    echo "внимание: рабочее дерево грязное - архив собирается из HEAD ($SHA)," >&2
    echo "незакоммиченные изменения в кит не попадут" >&2
fi

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
PKG_DIR="$STAGE/veles-deploy-kit-$VER"
mkdir -p "$PKG_DIR"

git archive HEAD | tar -x -C "$PKG_DIR"

# в поставку заказчику не едет внутренняя кухня: рунбук прод-топологии стенд,
# eval-отчёты, E2E-чек-листы деплоя, бенчмарки (ревью P2-5). README кита - свой.
for excl in docs deploy/стенд evals e2e bench; do
    rm -rf "$PKG_DIR/$excl"
done

cat > "$PKG_DIR/COMMIT.txt" <<EOF
kit_version: $VER
commit: $SHA
date: $DATE
subject: $SUBJ
built: $(date -Iseconds)
EOF

# защита от CRLF (архив могут собирать на Windows) и прав на исполнение
find "$PKG_DIR" -type f \( -name '*.sh' -o -name '*.py' -o -name '*.service' \) -exec sed -i 's/\r$//' {} +
chmod +x "$PKG_DIR/deploy/kit/"*.sh "$PKG_DIR/deploy/kit/lib/"*.sh 2>/dev/null || true

# в архиве не должно быть секретов: ни файлов, ни содержимого
if find "$PKG_DIR" \( -name '.env' -o -name '*.env' -o -name 'secrets*' \) | grep -q .; then
    echo "в архиве найдены файлы, похожие на секреты - сборка прервана" >&2
    exit 1
fi
# content-скан: ключи и внутренняя инфраструктура не должны уехать заказчику
# (паттерны разорваны классами [c], чтобы не совпадать с собственным текстом)
LEAKS=$(grep -rIlE 'sk-[a-zA-Z0-9]{16,}|BEGIN [A-Z ]*PRIVATE KEY|vps-ae[z]a|d3v1-se\.duck[d]ns|10[.]13[.]105[.]' "$PKG_DIR" 2>/dev/null || true)
if [ -n "$LEAKS" ]; then
    echo "content-скан нашёл кандидатов в секреты/внутреннюю инфру - сборка прервана:" >&2
    echo "$LEAKS" >&2
    echo "если это ложные срабатывания - вычисти текст из репозитория, а не отключай скан" >&2
    exit 1
fi

mkdir -p "$ROOT/scratch"
OUT="$ROOT/scratch/veles-deploy-kit-$VER.tar.gz"
tar -czf "$OUT" -C "$STAGE" "veles-deploy-kit-$VER"

echo "архив: $OUT"
echo "  версия: $VER"
echo "  commit: $SHA ($DATE)"
echo "  файлов: $(tar -tzf "$OUT" | wc -l)"
