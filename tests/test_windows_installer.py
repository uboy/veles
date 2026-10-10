"""Установщик Windows: лаунчер выбирает рабочий каталог (программа vs данные).

Раскладка установщика: ...\\veles\\app (программа) + ...\\veles\\data (данные,
.env, документы, индекс). Лаунчер работает в data, если тот существует, иначе
в каталоге exe (портативный запуск из архива). От этого зависит, что именно
удаляет uninstall.ps1 без -PurgeData.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DEPLOY_WINDOWS = Path(__file__).resolve().parents[1] / "deploy" / "windows"
sys.path.insert(0, str(_DEPLOY_WINDOWS))

import veles_launcher  # noqa: E402


@pytest.fixture()
def app_dir(tmp_path: Path) -> Path:
    app = tmp_path / "veles" / "app"
    app.mkdir(parents=True)
    return app


def test_workdir_is_data_when_present(app_dir: Path) -> None:
    (app_dir.parent / "data").mkdir()
    assert veles_launcher.resolve_workdir(app_dir) == app_dir.parent / "data"


def test_workdir_is_exe_dir_when_no_data(app_dir: Path) -> None:
    assert veles_launcher.resolve_workdir(app_dir) == app_dir


def test_workdir_ignores_data_file(app_dir: Path) -> None:
    # сосед с именем data, но это файл, а не каталог раскладки - не данные
    (app_dir.parent / "data").write_text("не каталог", encoding="utf-8")
    assert veles_launcher.resolve_workdir(app_dir) == app_dir


def test_workdir_data_is_used_even_if_empty(app_dir: Path) -> None:
    (app_dir.parent / "data").mkdir()
    assert veles_launcher.resolve_workdir(app_dir) == app_dir.parent / "data"


def test_apply_data_env_sets_veles_data(app_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = app_dir.parent / "data"
    workdir.mkdir()
    monkeypatch.delenv("VELES_DATA", raising=False)
    veles_launcher.apply_data_env(workdir)
    import os

    assert os.environ["VELES_DATA"] == str(workdir)


def test_apply_data_env_keeps_existing(app_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VELES_DATA", r"D:\мои-данные-veles")
    veles_launcher.apply_data_env(app_dir)
    import os

    assert os.environ["VELES_DATA"] == r"D:\мои-данные-veles"
