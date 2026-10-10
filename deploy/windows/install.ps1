# install.ps1 - установка veles "в один клик" для Windows (без прав администратора).
#
# Что делает:
#   1) копирует программу в %LOCALAPPDATA%\veles\app;
#   2) создаёт каталог данных %LOCALAPPDATA%\veles\data и файл настроек .env
#      (если .env уже есть - не трогает);
#   3) добавляет veles в автозапуск при входе пользователя (можно отключить -NoAutostart);
#   4) запускает сервер скрыто и проверяет, что он отвечает.
#
# Запуск: правый клик по install.ps1 -> "Выполнить с помощью PowerShell",
# или в консоли: powershell -ExecutionPolicy Bypass -File install.ps1
#
# Опции: -NoStart (не запускать сейчас), -NoAutostart (не добавлять в автозапуск),
#        -PayloadDir <путь> (каталог с veles.exe; по умолчанию папка veles рядом со скриптом),
#        -InstallRoot <путь> (по умолчанию %LOCALAPPDATA%\veles).
param(
  [string]$PayloadDir = "",
  [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA "veles"),
  [switch]$NoStart,
  [switch]$NoAutostart
)
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $PayloadDir) { $PayloadDir = Join-Path $ScriptDir "veles" }
$ExePath = Join-Path $PayloadDir "veles.exe"
if (-not (Test-Path $ExePath)) {
  Write-Host "ОШИБКА: не найден $ExePath"
  Write-Host "Распакуйте архив поставки целиком и запускайте install.ps1 из него:"
  Write-Host "рядом со скриптом должна лежать папка veles с файлом veles.exe."
  exit 1
}

$AppDir = Join-Path $InstallRoot "app"
$DataDir = Join-Path $InstallRoot "data"
$LogDir = Join-Path $InstallRoot "logs"
$AppService = Join-Path $AppDir "service.ps1"

Write-Host "[1/5] Останавливаю прежний veles (если запущен)"
if (Test-Path $AppService) {
  & $AppService stop -InstallRoot $InstallRoot | Out-Null
}

Write-Host "[2/5] Копирую программу в $AppDir (это минута-две, файлов много)"
robocopy "$PayloadDir" "$AppDir" /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) {
  Write-Host "ОШИБКА: копирование не удалось (robocopy код $LASTEXITCODE)"
  exit 1
}
# скрипты управления живут и в установленном каталоге: пользователь может
# удалить распакованный архив, а управление и удаление остаются рабочими
Copy-Item (Join-Path $ScriptDir "service.ps1") $AppDir -Force
Copy-Item (Join-Path $ScriptDir "uninstall.ps1") $AppDir -Force

Write-Host "[3/5] Готовлю каталог данных $DataDir"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$EnvFile = Join-Path $DataDir ".env"
if (Test-Path $EnvFile) {
  Write-Host "      .env уже есть - не трогаю: $EnvFile"
} else {
  $ExampleEnv = Join-Path $AppDir "config.example.env"
  if (Test-Path $ExampleEnv) {
    Copy-Item $ExampleEnv $EnvFile
    Write-Host "      Создан $EnvFile"
    Write-Host "      Там уже всё настроено; если хотите свою модель LLM - впишите её адрес и ключ в этот файл."
  } else {
    Write-Host "      ВНИМАНИЕ: config.example.env не найден в поставке - создайте .env сами: $EnvFile"
  }
}

if (-not $NoAutostart) {
  Write-Host "[4/5] Добавляю в автозапуск (при входе в Windows; в Диспетчере задач называется 'Veles')"
  # vbs-обёртка: wscript не имеет консоли, поэтому при входе в Windows окно
  # не появляется вообще (обёртка powershell.exe мигала бы чёрным окном)
  $VbsPath = Join-Path $AppDir "veles-background.vbs"
  $VbsLine = 'CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""' + $AppService + '"" start", 0, False'
  Set-Content -Path $VbsPath -Value $VbsLine -Encoding Unicode
  $RunKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
  if (-not (Test-Path $RunKey)) { New-Item -Path $RunKey -Force | Out-Null }
  Set-ItemProperty -Path $RunKey -Name 'Veles' -Value ('wscript.exe "' + $VbsPath + '"')
} else {
  Write-Host "[4/5] Автозапуск пропущен (-NoAutostart); существующая запись автозапуска снимается"
  Remove-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name 'Veles' -ErrorAction SilentlyContinue
}
# запись в "Установленные приложения" (Параметры -> Приложения) - при любой установке
$UninstKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Veles'
if (-not (Test-Path $UninstKey)) { New-Item -Path $UninstKey -Force | Out-Null }
$AppExe = Join-Path $AppDir "veles.exe"
$SizeKb = [math]::Round((Get-ChildItem $AppDir -Recurse -File | Measure-Object Length -Sum).Sum / 1KB)
Set-ItemProperty -Path $UninstKey -Name 'DisplayName' -Value 'Veles (локальная база знаний)'
Set-ItemProperty -Path $UninstKey -Name 'DisplayVersion' -Value '1.0'
Set-ItemProperty -Path $UninstKey -Name 'InstallLocation' -Value $InstallRoot
Set-ItemProperty -Path $UninstKey -Name 'DisplayIcon' -Value $AppExe
Set-ItemProperty -Path $UninstKey -Name 'Publisher' -Value 'veles'
Set-ItemProperty -Path $UninstKey -Name 'EstimatedSize' -Value $SizeKb -Type DWord
Set-ItemProperty -Path $UninstKey -Name 'NoModify' -Value 1 -Type DWord
Set-ItemProperty -Path $UninstKey -Name 'NoRepair' -Value 1 -Type DWord
Set-ItemProperty -Path $UninstKey -Name 'UninstallString' -Value ('powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + (Join-Path $AppDir 'uninstall.ps1') + '"')

if (-not $NoStart) {
  Write-Host "[5/5] Запускаю сервер"
  & $AppService start -InstallRoot $InstallRoot
  if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "УСТАНОВКА ЗАВЕРШЕНА НЕ ПОЛНОСТЬЮ: программа установлена, но сервер не стартовал."
    Write-Host "Разберитесь по журналу ($AppService log) и запустите: $AppService start"
    exit 1
  }
} else {
  Write-Host "[5/5] Запуск пропущен (-NoStart); запустить позже: service.ps1 start"
}

Write-Host ""
Write-Host "Установка завершена: $InstallRoot"
Write-Host "  - база знаний:  http://127.0.0.1:8400"
Write-Host "  - добавление документа:  $AppDir\veles.exe add путь\к\файлу.md"
Write-Host "  - вопрос в командной строке:  $AppDir\veles.exe ask ""вопрос"""
Write-Host "  - управление:  $AppDir\service.ps1 start|stop|status|log"
Write-Host "  - удаление программы (данные останутся):  $AppDir\uninstall.ps1"
