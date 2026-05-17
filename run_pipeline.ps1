Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExistingVenv = Join-Path $ScriptDir 'venv'
$VenvDir = if (Test-Path $ExistingVenv) { $ExistingVenv } else { Join-Path $ScriptDir '.venv' }

Write-Host "[pipeline] working dir: $ScriptDir"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Error "Python not found in PATH. Install Python 3.8+ and retry."
  exit 1
}

Write-Host "[pipeline] python: $(python --version)"

$CreatedVenv = $false
if (-not (Test-Path $VenvDir)) {
  Write-Host "[pipeline] creating virtualenv at $VenvDir"
  python -m venv $VenvDir
  $CreatedVenv = $true
}

Write-Host "[pipeline] activating venv"
& (Join-Path $VenvDir 'Scripts\Activate.ps1')

if ($CreatedVenv) {
  Write-Host "[pipeline] installing requirements"
  python -m pip install --upgrade pip
  $req = Join-Path $ScriptDir 'requirements.txt'
  if (Test-Path $req) { python -m pip install -r $req }
} else {
  Write-Host "[pipeline] using existing virtualenv; dependency install skipped"
}

Write-Host "[pipeline] setting numeric thread limits"
if (-not (Test-Path Env:LOKY_MAX_CPU_COUNT)) { $env:LOKY_MAX_CPU_COUNT = '1' }
if (-not (Test-Path Env:OMP_NUM_THREADS)) { $env:OMP_NUM_THREADS = '1' }
if (-not (Test-Path Env:OPENBLAS_NUM_THREADS)) { $env:OPENBLAS_NUM_THREADS = '1' }
if (-not (Test-Path Env:MKL_NUM_THREADS)) { $env:MKL_NUM_THREADS = '1' }
if (-not (Test-Path Env:KMA_FETCH_AWS)) { $env:KMA_FETCH_AWS = '0' }

$PipelineArgs = @($args)
if ($PipelineArgs.Count -eq 0) {
  $PipelineArgs = @('--fetch', 'all')
}

Write-Host "[pipeline] running unified pipeline"
python (Join-Path $ScriptDir 'src\pipeline.py') @PipelineArgs

Write-Host "[pipeline] done. Artifacts saved under: artifacts/ and output/"
