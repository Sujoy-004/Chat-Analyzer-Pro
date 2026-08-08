# make_nlp_env.ps1
#
# Creates a disposable NLP venv at a short path (default: TEMP) and
# installs the project with its [nlp] extras into it. Deep venv paths
# crash `pip install torch` on Windows (WinError 206). CI / stage D1
# will call this helper to prepare the NLP environment.

param(
    [string]$BasePath = $env:TEMP
)

$ErrorActionPreference = "Stop"

$venvDir = Join-Path $BasePath "chat-analyzer-nlp"
$projectRoot = $PSScriptRoot | Split-Path -Parent

if (Test-Path -LiteralPath $venvDir) {
    Write-Host "[INFO] Removing existing env at $venvDir"
    Remove-Item -LiteralPath $venvDir -Recurse -Force
}

Write-Host "[1/3] Creating venv at $venvDir"
py -m venv $venvDir
if ($LASTEXITCODE -ne 0) {
    throw "[error] Failed to create venv at $venvDir"
}

$python = Join-Path $venvDir 'Scripts\python.exe'

Write-Host "[2/3] Installing package + NLP extras"
try {
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip"
    }
    & $python -m pip install -e "$projectRoot[nlp]"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install project with [nlp] extras"
    }
}
catch {
    Write-Host "[error] pip install failed: $($_.Exception.Message)"
    exit 1
}

Write-Host "[3/3] Done"
Write-Host ""
Write-Host "To activate the env and run the analyzer:"
Write-Host "    & '$venvDir\Scripts\Activate.ps1'"
Write-Host "    chat-analyzer path\to\export.txt"