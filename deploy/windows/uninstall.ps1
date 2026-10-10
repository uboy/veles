# uninstall.ps1 - удалить veles: остановить сервер, снять автозапуск, удалить программу.
#
# По умолчанию данные остаются: документы, индекс и настройки
# (%LOCALAPPDATA%\veles\data) - на случай, если установите снова.
# Удалить и их: uninstall.ps1 -PurgeData
param(
  [switch]$PurgeData,
  [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA "veles")
)
$ErrorActionPreference = 'Stop'

$AppDir = Join-Path $InstallRoot "app"

if (-not (Test-Path $AppDir)) {
  Write-Host "veles не установлен ($AppDir не найден) - удалять нечего."
  exit 0
}
# защита от кривого пути: трогаем только каталог установки внутри LOCALAPPDATA
$lap = $env:LOCALAPPDATA.TrimEnd('\') + '\'
$root = $InstallRoot.TrimEnd('\')
if (-not $root.StartsWith($lap, [StringComparison]::OrdinalIgnoreCase) -or $root -eq $env:LOCALAPPDATA.TrimEnd('\')) {
  Write-Host "ОШИБКА: каталог установки должен быть своим подкаталогом внутри $env:LOCALAPPDATA (сейчас: $InstallRoot)."
  exit 1
}

Write-Host "[1/3] Останавливаю сервер"
& (Join-Path $AppDir "service.ps1") stop -InstallRoot $InstallRoot
if ($LASTEXITCODE -ne 0) {
  Write-Host "ОШИБКА: сервер не удалось остановить - удаление отменено, ничего не тронуто."
  Write-Host "Догазавершите процесс вручную (Диспетчер задач -> veles.exe) и повторите uninstall.ps1."
  exit 1
}

Write-Host "[2/3] Снимаю автозапуск и запись в «Установленных приложениях»"
Remove-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name 'Veles' -ErrorAction SilentlyContinue
Remove-Item -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Veles' -Recurse -ErrorAction SilentlyContinue

Write-Host "[3/3] Удаляю программу"
# прочь из удаляемого каталога, иначе Remove-Item упрётся в собственный CWD
Set-Location $env:TEMP
if ($PurgeData) {
  Remove-Item -LiteralPath $root -Recurse -Force
  Write-Host "Удалено полностью: $root (включая данные)"
} else {
  # -Exclude c -Recurse в PowerShell 5.1 ненадёжен - удаляем явно app и logs,
  # data (документы, индекс, .env) не трогаем
  Remove-Item -LiteralPath (Join-Path $root "app") -Recurse -Force
  Remove-Item -LiteralPath (Join-Path $root "logs") -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host "Программа и автозапуск удалены."
  Write-Host "Данные остались: $(Join-Path $root 'data') (документы, индекс, .env)."
  Write-Host "Удалить и их: uninstall.ps1 -PurgeData"
}
