# service.ps1 - управление сервером veles без консольного окна.
#
# Использование: service.ps1 start|stop|status|log
#   start   запустить сервер скрыто (если уже работает - ничего не делает)
#   stop    остановить (только процесс, установленный в app-каталоге)
#   status  что сейчас: процесс, ответ сервера, автозапуск
#   log     последние строки журнала
#
# Установленные пути: программа %LOCALAPPDATA%\veles\app, данные
# %LOCALAPPDATA%\veles\data, журнал %LOCALAPPDATA%\veles\logs.
param(
  [Parameter(Position = 0)][string]$Action = "status",
  [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA "veles")
)
$ErrorActionPreference = 'Stop'

$AppDir = Join-Path $InstallRoot "app"
$DataDir = Join-Path $InstallRoot "data"
$LogDir = Join-Path $InstallRoot "logs"
$ExePath = Join-Path $AppDir "veles.exe"
$OutLog = Join-Path $LogDir "veles.out.log"
$ErrLog = Join-Path $LogDir "veles.err.log"
$BaseUrl = "http://127.0.0.1:8400"

function Test-Health {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri "$BaseUrl/healthz"
    return ($r.StatusCode -eq 200)
  } catch { return $false }
}

function Get-VelesProcess {
  # только наш процесс: путь к exe обязан лежать ВНУТРИ app-каталога установки
  $prefix = $AppDir.TrimEnd('\') + '\'
  Get-Process -Name veles -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) }
}

switch ($Action) {
  "start" {
    if (Test-Health) { Write-Host "veles уже запущен: $BaseUrl"; exit 0 }
    if (-not (Test-Path $ExePath)) {
      Write-Host "ОШИБКА: veles не установлен ($ExePath не найден). Запустите install.ps1."
      exit 1
    }
    if (Get-VelesProcess) {
      Write-Host "Процесс veles уже есть, но сервер не отвечает (возможно, ещё поднимается)."
      Write-Host "Подождите минуту и проверьте: service.ps1 status; журнал: service.ps1 log"
      Write-Host "Перезапустить насильно: service.ps1 stop, затем service.ps1 start"
      exit 1
    }
    New-Item -ItemType Directory -Force -Path $LogDir, $DataDir | Out-Null
    Start-Process -FilePath $ExePath -WorkingDirectory $DataDir -WindowStyle Hidden `
      -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog
    Write-Host "veles запускается скрыто; самый первый старт скачивает модель поиска - это может занять несколько минут."
    $ok = $false
    foreach ($i in 1..60) {
      Start-Sleep -Seconds 2
      if (Test-Health) { $ok = $true; break }
      if (-not (Get-VelesProcess)) {
        Write-Host "ОШИБКА: процесс завершился при старте. Последние строки журнала:"
        if (Test-Path $ErrLog) { Get-Content $ErrLog -Tail 15 -Encoding UTF8 }
        elseif (Test-Path $OutLog) { Get-Content $OutLog -Tail 15 -Encoding UTF8 }
        exit 1
      }
      if ($i -eq 10) { Write-Host "...всё ещё запускается (журнал: $OutLog)" }
    }
    if ($ok) { Write-Host "Готово: $BaseUrl"; exit 0 }
    Write-Host "Сервер не ответил за время ожидания. Журнал: service.ps1 log (файлы $OutLog, $ErrLog)"
    exit 1
  }
  "stop" {
    $p = Get-VelesProcess
    if (-not $p) {
      Write-Host "veles не запущен (процессов из $AppDir нет; чужие процессы с таким именем не трогаю)."
      exit 0
    }
    $p | Stop-Process -Force
    Start-Sleep -Seconds 1
    if (Get-VelesProcess) { Write-Host "ОШИБКА: процесс не завершился."; exit 1 }
    Write-Host "Остановлен."
    exit 0
  }
  "status" {
    $health = Test-Health
    $proc = @(Get-VelesProcess)
    $run = (Get-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -ErrorAction SilentlyContinue).Veles
    Write-Host ("Процесс:    " + $(if ($proc.Count) { "запущен (PID " + (($proc | ForEach-Object Id) -join ", ") + ")" } else { "не запущен" }))
    Write-Host ("Сервер:     " + $(if ($health) { "отвечает, $BaseUrl" } else { "не отвечает" }))
    Write-Host ("Автозапуск: " + $(if ($run) { "включён (ключ Veles в HKCU ...Run)" } else { "выключен" }))
    if ($health) { exit 0 } else { exit 1 }
  }
  "log" {
    if (Test-Path $OutLog) { Get-Content $OutLog -Tail 40 -Encoding UTF8 } else { Write-Host "Журнала пока нет: $OutLog" }
    if ((Test-Path $ErrLog) -and ((Get-Item $ErrLog).Length -gt 0)) {
      Write-Host "--- поток ошибок (последние строки) ---"
      Get-Content $ErrLog -Tail 10 -Encoding UTF8
    }
  }
  default {
    Write-Host "Использование: service.ps1 start|stop|status|log"
    exit 2
  }
}
