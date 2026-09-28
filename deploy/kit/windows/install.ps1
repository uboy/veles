# install.ps1 - установка veles на Windows (профили lite и personal).
# venv + CLI + фасад (запуск run-facade.ps1); WebUI - опционально в venv.
#
# Примеры:
#   .\install.ps1                          # lite: только CLI, без демонов (как в bash-версии)
#   .\install.ps1 -Profile personal        # venv + фасад + CLI
#   .\install.ps1 -WithWebUI               # + open-webui в venv
#
# Ключи LLM вводятся при установке и пишутся только в локальный .env - в архив не попадают.
[CmdletBinding()]
param(
    [ValidateSet('personal', 'lite')]
    # дефолт lite - тот же, что в bash-версии (выравнивание расхождения ps1/bash, ревью P2-8)
    [string]$Profile = 'lite',

    [string]$InstallDir = '',

    [switch]$WithWebUI,

    [switch]$DryRun,

    [switch]$Yes
)

$ErrorActionPreference = 'Stop'

$KitDir = Split-Path -Parent $MyInvocation.MyCommand.Path          # ...\deploy\kit\windows
$DeployKitDir = Split-Path -Parent $KitDir                          # ...\deploy\kit
$RepoDir = Split-Path -Parent (Split-Path -Parent $DeployKitDir)    # корень репозитория/архива
if (-not $InstallDir) { $InstallDir = $DeployKitDir }
$EnvFile = Join-Path $InstallDir '.env'
$VenvDir = Join-Path $InstallDir '.venv'

function Write-Step([string]$Message) { Write-Host "[veles] $Message" }
function Stop-Install([string]$Message) { Write-Host "[ошибка] $Message" -ForegroundColor Red; exit 1 }

function Find-Python {
    # py -3.11 / -3.12 / -3.13, затем python из PATH; нужен 3.11+
    $candidates = @()
    foreach ($v in '3.13', '3.12', '3.11') {
        if (Get-Command py -ErrorAction SilentlyContinue) { $candidates += "py -$v" }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += 'python' }
    foreach ($c in $candidates) {
        $parts = $c -split ' '
        # кандидат 'python' без суффикса: $parts[1..0] дал бы лишний позиционный
        # аргумент ("python python -c ..." - ревью P1-2, доказано прогоном)
        $extra = @()
        if ($parts.Count -gt 1) { $extra = $parts[1..($parts.Count - 1)] }
        try {
            $ver = & $parts[0] @($extra) -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
        } catch { continue }
        if (-not $ver) { continue }
        $major, $minor = $ver.Trim().Split('.')
        if ([int]$major -gt 3 -or ([int]$major -eq 3 -and [int]$minor -ge 11)) {
            return @{ Exe = $parts[0]; Args = @($extra); Version = $ver.Trim() }
        }
    }
    return $null
}

function New-Secret {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    ($bytes | ForEach-Object { $_.ToString('x2') }) -join ''
}

function Set-EnvVar([string]$File, [string]$Key, [string]$Value) {
    $lines = @()
    if (Test-Path $File) { $lines = Get-Content $File -Encoding UTF8 }
    $found = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^$Key=") { $lines[$i] = "$Key=$Value"; $found = $true }
    }
    if (-not $found) { $lines += "$Key=$Value" }
    Set-Content -Path $File -Value $lines -Encoding UTF8
}

function Get-EnvVar([string]$File, [string]$Key) {
    if (-not (Test-Path $File)) { return '' }
    $line = Get-Content $File -Encoding UTF8 | Where-Object { $_ -match "^$Key=" } | Select-Object -First 1
    if ($line) { return ($line -split '=', 2)[1] }
    return ''
}

# ---------- preflight ----------

Write-Step "профиль: $Profile; каталог установки: $InstallDir"

$py = Find-Python
if (-not $py) { Stop-Install "python 3.11+ не найден. Установите python 3.11+ (python.org или winget) и повторите" }
Write-Step "python: $($py.Version)"

$freeMb = 0
$drive = (Get-PSDrive -Name ($InstallDir.Substring(0, 1)) -ErrorAction SilentlyContinue)
if ($drive) { $freeMb = [int]($drive.Free / 1MB) }
$needMb = 1024
if ($Profile -eq 'personal') { $needMb = 5120 }
if ($freeMb -and $freeMb -lt $needMb) { Stop-Install "мало места: ${freeMb}МБ свободно, нужно >=${needMb}МБ" }
Write-Step "диск: свободно ${freeMb}МБ (нужно >=${needMb}МБ)"

if ($DryRun) {
    Write-Step "dry-run: preflight пройден, установка не выполнялась"
    exit 0
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# ---------- .env ----------

$existing = Test-Path $EnvFile
if ($existing) { Write-Step "найден существующий .env - значения сохраняются, дополняю недостающие" }

Set-EnvVar $EnvFile 'VELES_KIT_PROFILE' $Profile
Set-EnvVar $EnvFile 'VELES_DATA' (Join-Path $InstallDir 'data')
Set-EnvVar $EnvFile 'VELES_CORPUS_ROOT' (Join-Path $InstallDir 'corpus')
Set-EnvVar $EnvFile 'VELES_FACADE_PORT' '8400'
Set-EnvVar $EnvFile 'VELES_IDENTITY_MODE' 'static'
Set-EnvVar $EnvFile 'VELES_WITH_WEBUI' ($(if ($WithWebUI) { '1' } else { '0' }))

foreach ($secretKey in 'VELES_API_TOKEN', 'VELES_LINK_SECRET') {
    if (-not (Get-EnvVar $EnvFile $secretKey)) { Set-EnvVar $EnvFile $secretKey (New-Secret) }
}
if ($Profile -eq 'personal') {
    if (-not (Get-EnvVar $EnvFile 'WEBUI_SECRET_KEY')) { Set-EnvVar $EnvFile 'WEBUI_SECRET_KEY' (New-Secret) }
    if (-not (Get-EnvVar $EnvFile 'VELES_LLM_BASE_URL')) { Set-EnvVar $EnvFile 'VELES_LLM_BASE_URL' 'https://api.z.ai/api/coding/paas/v4' }
    if (-not (Get-EnvVar $EnvFile 'VELES_LLM_MODEL')) { Set-EnvVar $EnvFile 'VELES_LLM_MODEL' 'glm-5.3-flash' }
    if (-not (Get-EnvVar $EnvFile 'VELES_LLM_API_KEY') -and -not $Yes) {
        $llmKey = Read-Host 'Ключ LLM (пишется только в .env, пусто = пропустить)'
        Set-EnvVar $EnvFile 'VELES_LLM_API_KEY' $llmKey
    }
}

# ---------- venv и пакет ----------

Write-Step "создаю venv: $VenvDir"
& $py.Exe @($py.Args) -m venv $VenvDir
$pip = Join-Path $VenvDir 'Scripts\pip.exe'
$veles = Join-Path $VenvDir 'Scripts\veles.exe'

& $pip install --quiet --upgrade pip
if ($Profile -eq 'personal') {
    Write-Step "ставлю veles[st,faiss] (torch - самая долгая часть)"
    & $pip install --quiet -e "$RepoDir[st,faiss]"
} else {
    & $pip install --quiet -e $RepoDir
}
if (-not (Test-Path $veles)) { Stop-Install "CLI не появился после pip install - смотрите вывод pip выше" }
Write-Step "CLI: $veles"

# ---------- фасад и WebUI ----------

if ($Profile -eq 'personal') {
    $runFacade = Join-Path $InstallDir 'run-facade.ps1'
    @"
# run-facade.ps1 - запуск фасада veles (сгенерирован install.ps1)
`$ErrorActionPreference = 'Stop'
Set-Location '$RepoDir'
Get-Content '$EnvFile' | Where-Object { `$_ -match '^\w+=' } | ForEach-Object {
    `$k, `$v = `$_ -split '=', 2
    Set-Item -Path "env:`$k" -Value `$v
}
& '$veles' serve --host 127.0.0.1 --port 8400
"@ | Set-Content -Path $runFacade -Encoding UTF8
    Write-Step "фасад: запускайте $runFacade (порт 8400, только localhost)"

    if ($WithWebUI) {
        Write-Step "ставлю open-webui в venv (большая загрузка)"
        & $pip install --quiet open-webui
        $runWebui = Join-Path $InstallDir 'run-webui.ps1'
        @"
# run-webui.ps1 - запуск Open WebUI (сгенерирован install.ps1)
& '$(Join-Path $VenvDir 'Scripts\open-webui.exe')' serve --port 8300
"@ | Set-Content -Path $runWebui -Encoding UTF8
        Write-Step "WebUI: запускайте $runWebui (порт 8300)"
    }
}

Write-Step "проверка: $veles status"
& $veles status

Write-Host ''
Write-Host 'Установка завершена. Управление:'
Write-Host "  CLI:          $veles status"
if ($Profile -eq 'personal') {
    Write-Host "  фасад:        $(Join-Path $InstallDir 'run-facade.ps1')"
    Write-Host "  самопроверка: см. README.md в deploy\kit (verify-install.sh - Linux)"
}
Write-Host "  удаление:     $(Join-Path $KitDir 'uninstall.ps1') -InstallDir '$InstallDir' -PurgeData"
