param(
  [ValidateSet("minimal","standard","full","dev")]
  [string]$Mode = "full",
  [switch]$Cuda,
  [string]$Venv = "venv",
  [switch]$SkipCorpora,
  [switch]$SkipModels
)

$ErrorActionPreference = "Stop"

Write-Host "== EDMG installer ==" -ForegroundColor Cyan
Write-Host "Mode: $Mode  CUDA: $Cuda  Venv: $Venv" -ForegroundColor Cyan

python scripts\edmg_installer.py install --mode $Mode $(if ($Cuda) { "--cuda" } else { "" }) --venv $Venv $(if ($SkipCorpora) { "--skip-corpora" } else { "" }) $(if ($SkipModels) { "--skip-models" } else { "" })

Write-Host "`nDone. To run:" -ForegroundColor Green
Write-Host "  .\$Venv\Scripts\activate" -ForegroundColor Green
Write-Host "  python -m enhanced_deforum_music_generator ui --port 7860" -ForegroundColor Green
