# build_exe.ps1 - сборка veles.exe для Windows (один каталог + zip).
# Результат: dist\veles\veles.exe (запуск без аргументов = сервер на 8400),
# упакованный в dist\veles-windows.zip вместе с установщиком и доками.
# Требования: Python 3.11+ в PATH; сборка выполняется на Windows.
# Примечание: torch входит в поставку (exe ~800 МБ); при первом старте
# скачивается только модель поиска (~450 МБ, кэш пользователя).
$ErrorActionPreference = 'Stop'
$RepoDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # deploy\windows -> корень
# venv сборки, dist и build создаются в корне репозитория, а не в каталоге запуска
Set-Location $RepoDir

Write-Host "[exe] venv сборки"
py -3.11 -m venv build-venv
if ($LASTEXITCODE -ne 0) { Write-Host "ОШИБКА: py -3.11 не найден или venv не создан"; exit 1 }
.\build-venv\Scripts\pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Host "ОШИБКА: pip upgrade"; exit 1 }
.\build-venv\Scripts\pip install --quiet -e "$RepoDir[st,faiss]" pyinstaller
if ($LASTEXITCODE -ne 0) { Write-Host "ОШИБКА: pip install"; exit 1 }

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
if ($LASTEXITCODE -ne 0) { Write-Host "ОШИБКА: pyinstaller"; exit 1 }

Write-Host "[exe] комплект поставки"
Copy-Item "$RepoDir\config.example.env" dist\veles\
Copy-Item "$RepoDir\docs\RUN-WINDOWS.md", "$RepoDir\docs\REQUIREMENTS.md" dist\veles\
Copy-Item "$RepoDir\README.md" dist\veles\
# zip-комплект: в корне архива - установщики (cmd для двойного клика) и инструкция,
# рядом папка veles с программой; install.ps1 ищет программу по умолчанию в .\veles
Compress-Archive -Path dist\veles, `
  "$RepoDir\deploy\windows\install.cmd", "$RepoDir\deploy\windows\install.ps1", `
  "$RepoDir\deploy\windows\service.ps1", "$RepoDir\deploy\windows\uninstall.ps1", `
  "$RepoDir\docs\RUN-WINDOWS.md" `
  -DestinationPath dist\veles-windows.zip -Force
if ($LASTEXITCODE -ne 0) { Write-Host "ОШИБКА: Compress-Archive"; exit 1 }
Write-Host "[exe] готово: dist\veles\veles.exe и dist\veles-windows.zip (установка в один клик)"
