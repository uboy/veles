# uninstall.ps1 - снятие установки veles с Windows.
#
#   .\uninstall.ps1                                  # удалить venv и файлы установки, данные сохранить
#   .\uninstall.ps1 -PurgeData                       # + данные и .env (спросит дважды)
#   .\uninstall.ps1 -PurgeData -Yes                  # без вопросов
[CmdletBinding()]
param(
    [string]$InstallDir = '',

    [switch]$PurgeData,

    [switch]$Yes
)

$ErrorActionPreference = 'Stop'

$KitDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DeployKitDir = Split-Path -Parent $KitDir
if (-not $InstallDir) { $InstallDir = $DeployKitDir }
$EnvFile = Join-Path $InstallDir '.env'
$VenvDir = Join-Path $InstallDir '.venv'

function Write-Step([string]$Message) { Write-Host "[veles] $Message" }

# остановить процессы этого venv (python и veles.exe - консольный скрипт pip;
# фильтр Name like 'python%' пропускал живой veles.exe, Remove-Item .venv падал
# на залоченных файлах - ревью P2-8: фильтр только по CommandLine)
$venvRoot = (Resolve-Path $InstallDir -ErrorAction SilentlyContinue).Path
if ($venvRoot) {
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "$venvRoot*" -and $_.ProcessId -ne $PID }
    foreach ($p in $procs) {
        Write-Step "останавливаю процесс $($p.ProcessId): $($p.CommandLine)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

if (Test-Path $VenvDir) {
    Remove-Item -Recurse -Force $VenvDir
    Write-Step "venv удалён: $VenvDir"
} else {
    Write-Step "venv не найден (уже чисто)"
}

if ($PurgeData) {
    $targets = @()
    foreach ($t in (Join-Path $InstallDir 'data'), (Join-Path $InstallDir 'corpus'), $EnvFile, (Join-Path $InstallDir '.veles-kit')) {
        if (Test-Path $t) { $targets += $t }
    }
    if ($targets.Count -gt 0) {
        Write-Step "будут удалены:"
        foreach ($t in $targets) {
            $size = if (Test-Path $t -PathType Container) {
                '{0:N1} МБ' -f ((Get-ChildItem $t -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1MB)
            } else { '<1 МБ' }
            Write-Host ("  {0,10}  {1}" -f $size, $t)
        }
        if (-not $Yes) {
            $a1 = Read-Host 'Удалить перечисленное? Данные базы и .env с ключами будут потеряны [y/N]'
            if ($a1 -notmatch '^[yY]') { Write-Step 'отменено'; exit 1 }
            $a2 = Read-Host 'Подтвердите второй раз: удалить безвозвратно [y/N]'
            if ($a2 -notmatch '^[yY]') { Write-Step 'отменено'; exit 1 }
        }
        foreach ($t in $targets) { Remove-Item -Recurse -Force $t }
        Write-Step 'данные удалены'
    } else {
        Write-Step 'данных не осталось'
    }
} else {
    Write-Step "данные и .env сохранены: $InstallDir"
    Write-Step "полное удаление: .\uninstall.ps1 -InstallDir '$InstallDir' -PurgeData"
}

$left = Get-ChildItem $InstallDir -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in '.env', '.venv', 'data', 'corpus', '.veles-kit' }
if ($PurgeData -and -not $left) {
    Write-Step "проверка: файлов установки в $InstallDir не осталось"
} elseif ($left) {
    Write-Step "осталось (не удалялось): $($left.Name -join ', ')"
}
