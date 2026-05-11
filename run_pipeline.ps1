Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ScriptDir '.venv'

Write-Host "[pipeline] working dir: $ScriptDir"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Error "Python not found in PATH. Install Python 3.8+ and retry."
  exit 1
}

Write-Host "[pipeline] python: $(python --version)"

if (-not (Test-Path $VenvDir)) {
  Write-Host "[pipeline] creating virtualenv at $VenvDir"
  python -m venv $VenvDir
}

Write-Host "[pipeline] activating venv"
& (Join-Path $VenvDir 'Scripts\Activate.ps1')

Write-Host "[pipeline] upgrading pip and installing requirements"
python -m pip install --upgrade pip
$req = Join-Path $ScriptDir 'requirements.txt'
if (Test-Path $req) { python -m pip install -r $req }

Write-Host "[pipeline] setting numeric thread limits"
if (-not (Test-Path Env:LOKY_MAX_CPU_COUNT)) { $env:LOKY_MAX_CPU_COUNT = '1' }
if (-not (Test-Path Env:OMP_NUM_THREADS)) { $env:OMP_NUM_THREADS = '1' }
if (-not (Test-Path Env:OPENBLAS_NUM_THREADS)) { $env:OPENBLAS_NUM_THREADS = '1' }
if (-not (Test-Path Env:MKL_NUM_THREADS)) { $env:MKL_NUM_THREADS = '1' }

Write-Host "[pipeline] running weather fetch (src\weather_api.py)"
python (Join-Path $ScriptDir 'src\weather_api.py')

Write-Host "[pipeline] preparing model input (src\data_prep.py)"
python (Join-Path $ScriptDir 'src\data_prep.py')

Write-Host "[pipeline] running training pipeline (train.py)"
python (Join-Path $ScriptDir 'train.py')

Write-Host "[pipeline] optional plotting (plot_model_comparison.py)"
$plot = Join-Path $ScriptDir 'plot_model_comparison.py'
if (Test-Path $plot) {
  python $plot
}

Write-Host "[pipeline] done. Artifacts saved under: artifacts/ and output/"
