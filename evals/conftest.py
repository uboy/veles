"""Pytest-сборка eval-скриптов.

eval-скрипты репо запускаются напрямую («python evals/<имя>.py», финальная
строка VALIDATION-PASS, код 0), но повторный прогон валидации бывает и через
pytest. Без хука «pytest evals/» собирал 0 тестов («no tests ran», exit 5):
имя файла не матчит дефолтный python_files. Хук добавляет к сборке ТОЛЬКО
index-format-validate.py - у остальных eval-скриптов импорт имеет тяжёлые
побочные эффекты (env-подстановка, поднятие uvicorn/LDAP), собирать их
тестами нельзя. meta-compact-validate.py соблюдает ту же дисциплину (env только
в main(), подпроцессы без побочных эффектов) - тоже собирается.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_COLLECTABLE = {"index-format-validate.py", "meta-compact-validate.py"}


def pytest_collect_file(file_path, parent):
    # явный файл («pytest evals/index-format-validate.py») ядро собирает само
    # (init-path собирается независимо от python_files) - хук тут добавил бы
    # ВТОРОЙ коллектор и гонял валидацию дважды. Полный прогон без явных
    # аргументов («pytest», «pytest .») хук тоже не трогает: иначе 65s
    # валидация молча попадает в постоянный набор сюиты. Хук собирает файл
    # только когда ЯВНЫМ аргументом передан каталог evals (или подкаталог).
    if file_path.name not in _COLLECTABLE or parent.session.isinitpath(file_path):
        return None
    config = parent.session.config
    explicit = {Path(a).resolve() for a in config.args} - {config.rootpath}
    p = Path(file_path).resolve()
    if not any(arg == p or arg in p.parents for arg in explicit):
        return None
    return pytest.Module.from_parent(parent, path=file_path)
