param(
    [string]$CatalogRoot = "F:\Work\FileService.Webapi\FileService.Webapi\upload",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8100,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$serviceRoot = Join-Path $projectRoot 'service'
$venvRoot = Join-Path $projectRoot '.venv'
$python = Join-Path $venvRoot 'Scripts\python.exe'
$dbPath = Join-Path $projectRoot 'data\sqlite\visual-search.db'
$modelCache = Join-Path $projectRoot 'data\models'

function Test-Http([string]$Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $CatalogRoot)) {
    throw "CatalogRoot does not exist: $CatalogRoot"
}

if (-not (Test-Path -LiteralPath $python)) {
    $created = $false
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & py -3.11 -m venv $venvRoot
        $created = $LASTEXITCODE -eq 0
        if (-not $created) {
            & py -3.12 -m venv $venvRoot
            $created = $LASTEXITCODE -eq 0
        }
    }
    if (-not $created) { throw 'Failed to create Python venv. Install Python 3.11 or 3.12 first. Python 3.14 is not supported by the pinned torch build.' }

    $requirements = Join-Path $serviceRoot 'requirements.txt'
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
    & $python -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'dependency install failed.' }
}

New-Item -ItemType Directory -Path (Split-Path -Parent $dbPath) -Force | Out-Null
New-Item -ItemType Directory -Path $modelCache -Force | Out-Null

$healthUrl = "http://${HostName}:$Port/health/live"
if (-not (Test-Http $healthUrl)) {
    $env:PYTHONPATH = '.'
    $env:VISUAL_SEARCH_STORE_BACKEND = 'sqlite'
    $env:VISUAL_SEARCH_SQLITE_PATH = $dbPath
    $env:VISUAL_SEARCH_CATALOG_ROOT = $CatalogRoot
    $env:VISUAL_SEARCH_CATALOG_ROOTS = $CatalogRoot
    $env:VISUAL_SEARCH_DEVICE = 'cpu'
    $env:VISUAL_SEARCH_MODEL_ID = 'facebook/dinov2-base'
    $env:VISUAL_SEARCH_MODEL_VERSION = 'dinov2-base-v1'
    $env:VISUAL_SEARCH_PUBLIC_IMAGE_BASE_URL = '/v1/catalog/image?path='
    Remove-Item Env:\HF_ENDPOINT -ErrorAction SilentlyContinue
    $env:HF_HOME = $modelCache
    Start-Process -FilePath $python -WorkingDirectory $serviceRoot -WindowStyle Hidden `
        -ArgumentList '-m','uvicorn','app.main:app','--host',$HostName,'--port',"$Port"
}

$deadline = (Get-Date).AddMinutes(5)
do {
    Start-Sleep -Seconds 2
    try {
        $health = Invoke-RestMethod "http://${HostName}:$Port/health/ready"
    } catch {
        $health = $null
    }
} until ($health.status -eq 'ok' -or (Get-Date) -gt $deadline)

if ($health.status -ne 'ok') {
    throw 'Visual search startup timed out. Run uvicorn in foreground to inspect logs.'
}

$adminUrl = "http://${HostName}:$Port/admin"
Write-Host "Visual Search is ready: $adminUrl" -ForegroundColor Green
Write-Host "CatalogRoot: $CatalogRoot"
Write-Host "SQLite: $dbPath"
if (-not $NoBrowser) {
    Start-Process $adminUrl
}
