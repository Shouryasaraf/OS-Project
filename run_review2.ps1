param(
    [switch]$Test,
    [string]$Trace
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $pythonExe = $pythonCommand.Source
} elseif (Test-Path -LiteralPath $bundledPython) {
    $pythonExe = $bundledPython
} else {
    throw 'Python 3.10+ was not found. Install Python or run this project inside Codex.'
}

Push-Location $repoRoot
try {
    $env:PYTHONPATH = Join-Path $repoRoot 'src'
    if ($Test) {
        & $pythonExe -m unittest discover -s tests -v
    } elseif ($Trace) {
        & $pythonExe -m adaptive_prefetch replay $Trace
    } else {
        & $pythonExe -m adaptive_prefetch demo
    }
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
