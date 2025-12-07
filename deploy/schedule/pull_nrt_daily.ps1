Param(
  [string]$RepoRoot = "$PSScriptRoot\..\..",
  [string]$Ym = "current",
  [string]$Since = "-P1D",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$python = "$Env:PYTHON"; if (-not $python) { $python = "python" }

Push-Location $RepoRoot

# Ensure logs dir
$newRunId = (Get-Date -Format "yyyyMMddTHHmmss")
$healthOut = "reports/health/health_$(Get-Date -Format 'yyyyMMdd').json"

# 1) Health probe
& $python -m ArcticRoute.api.cli health.check --out $healthOut | Out-Host

# 2) NRT ingest
$dry = $false; if ($DryRun) { $dry = $true }
$ingestArgs = @("-m", "ArcticRoute.api.cli", "ingest.nrt.pull", "--ym", $Ym, "--what", "ice,wave", "--since", $Since)
if ($dry) { $ingestArgs += "--dry-run" }
& $python @ingestArgs | Out-Host

# 3) Fuse (stacking)
& $python -m ArcticRoute.api.cli risk.fuse --ym $Ym --method stacking | Out-Host

# 4) Report (calibration)
& $python -m ArcticRoute.api.cli report.build --ym $Ym --include calibration | Out-Host

Pop-Location

