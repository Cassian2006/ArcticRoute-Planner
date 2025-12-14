$ErrorActionPreference = "Stop"
$root = Resolve-Path "."
$app = Resolve-Path "$root\ArcticRoute\apps\app_min.py"
Write-Host "Launching Streamlit with:" $app
streamlit run $app
