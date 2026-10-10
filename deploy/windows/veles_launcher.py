"""Точка входа veles.exe: без аргументов - сервер, с аргументами - CLI veles.

Запуск veles.exe: поднимает сервер на 127.0.0.1:8400. «veles.exe add файл.md»
и прочие команды работают как обычный CLI.

Рабочий каталог: если рядом с каталогом exe есть соседний каталог `data`
(раскладка установщика: ...\\veles\\app + ...\\veles\\data), работа идёт в нём -
там .env, документы и индекс, и деинсталлятор может удалить программу, не
тронув данные. Иначе (портативный запуск из распакованного архива) - в
каталоге exe, как раньше.

Корень данных для veles.config задаётся переменной VELES_DATA: у замороженного
exe модули лежат в app\\_internal, и путь «рядом с модулем» указал бы внутрь
программы. В .env этот путь менять не нужно (если нужен особый каталог данных -
задайте системную переменную окружения VELES_DATA, она важнее).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_workdir(exe_dir: Path) -> Path:
    """Каталог данных: соседний `data`, если есть, иначе сам каталог exe."""
    data_dir = exe_dir.parent / "data"
    return data_dir if data_dir.is_dir() else exe_dir


def apply_data_env(workdir: Path) -> None:
    """VELES_DATA = рабочий каталог; существующую переменную не перекрываем."""
    os.environ.setdefault("VELES_DATA", str(workdir))


def _fix_console_encoding() -> None:
    """Читаемый кириллический вывод в консоли Windows: консоль по умолчанию
    в cp866, а Python пишет UTF-8 - переключаем консоль на UTF-8. Без консоли
    (скрытый сервис) команда просто ничего не меняет."""
    if sys.platform == "win32" and sys.stdout is not None and sys.stdout.isatty():
        os.system("chcp 65001 >NUL")


def main() -> int:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
    else:
        exe_dir = Path(__file__).parent
    workdir = resolve_workdir(exe_dir)
    os.chdir(workdir)
    apply_data_env(workdir)
    _fix_console_encoding()
    from veles.cli import main as cli_main

    if len(sys.argv) == 1:
        sys.argv += ["serve", "--host", "127.0.0.1", "--port", "8400"]
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
