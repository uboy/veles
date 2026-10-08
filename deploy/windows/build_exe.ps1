# build_exe.ps1 - сборка veles.exe для Windows (один каталог + zip).
# Результат: dist\veles\veles.exe (запуск без аргументов = сервер на 8400),
# упакованный в dist\veles-windows.zip вместе с config.example.env и доками.
# Требования: Python 3.11+ в PATH; сборка выполняется на Windows.
# Примечание: размер каталога ~1.5-2 ГБ (torch + модель поиска ставятся
# рядом при первом старте, в exe не пакуются).
$ErrorActionPreference = 'Stop'
$RepoDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # deploy\windows -> корень

Write-Host "[exe] venv сборки"
py -3.11 -m venv build-venv
.\build-venv\Scripts\pip install --quiet --upgrade pip
.\build-venv\Scripts\pip install --quiet -e "$RepoDir[st,faiss]" pyinstaller

Write-Host "[exe] PyInstaller"
.\build-venv\Scripts\pyinstaller --noconfirm --clean `
  --name veles `
  --hidden-import sentence_transformers `
  --hidden-import torch `
  --hidden-import transformers `
  --paths "$RepoDir" `
  --collect-submodules veles `
  --collect-all sentence_transformers `
  --collect-all transformers `
  --collect-all tokenizers `
  --collect-all safetensors `
  "$RepoDir\deploy\windows\veles_launcher.py"

Write-Host "[exe] комплект поставки"
Copy-Item "$RepoDir\config.example.env" dist\veles\
Copy-Item "$RepoDir\docs\RUN-WINDOWS.md", "$RepoDir\docs\REQUIREMENTS.md" dist\veles\
Copy-Item "$RepoDir\README.md" dist\veles\
Compress-Archive -Path dist\veles -DestinationPath dist\veles-windows.zip -Force
Write-Host "[exe] готово: dist\veles\veles.exe и dist\veles-windows.zip"
