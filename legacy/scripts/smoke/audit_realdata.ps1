# Smoke: Real-Data audit quick profile
$ErrorActionPreference = "Stop"
python -m ArcticRoute.api.cli audit.real --profile real.quick
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# Assert a representative artifact if present
$path = "ArcticRoute/data_processed/risk/risk_fused_202412.nc"
if (Test-Path $path) {
  python -m ArcticRoute.api.cli audit.assert --path $path
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "Audit smoke finished OK"








