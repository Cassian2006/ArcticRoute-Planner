$ErrorActionPreference = "Stop"
$root = Resolve-Path "."
$app = Resolve-Path "$root\ArcticRoute\apps\gis_viewer.py"
Write-Host "Launching GIS Viewer with:" $app
streamlit run $app






