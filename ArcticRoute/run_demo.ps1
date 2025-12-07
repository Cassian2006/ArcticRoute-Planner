chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
#!/usr/bin/env pwsh

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Push-Location $PSScriptRoot
try {
    $cfgPath = Join-Path $PSScriptRoot 'config/runtime.yaml'
    if (-not (Test-Path $cfgPath)) {
        Write-Host "[run_demo] missing config/runtime.yaml" -ForegroundColor Yellow
        exit 1
    }

    $cfgRaw = $cfgPath.Replace('\', '\\')
    $envRel = (& python -c "from pathlib import Path; import yaml; cfg=Path(r'$cfgRaw'); data=yaml.safe_load(cfg.read_text(encoding='utf-8')) or {}; print(data.get('data', {}).get('env_nc') or 'data_processed/env_clean.nc')").Trim()
    if (-not $envRel) { $envRel = 'data_processed/env_clean.nc' }
    if ([System.IO.Path]::IsPathRooted($envRel)) { $envPath = $envRel } else { $envPath = Join-Path $PSScriptRoot $envRel }
    if (-not (Test-Path $envPath)) {
        Write-Host "[run_demo] missing environment file: $envPath" -ForegroundColor Yellow
        exit 1
    }

    $corrRel = (& python -c "from pathlib import Path; import yaml; cfg=Path(r'$cfgRaw'); data=yaml.safe_load(cfg.read_text(encoding='utf-8')) or {}; print(data.get('data', {}).get('corridor_prob') or data.get('behavior', {}).get('corridor_path') or 'data_processed/corridor_prob.nc')").Trim()
    if (-not $corrRel) { $corrRel = 'data_processed/corridor_prob.nc' }
    if ([System.IO.Path]::IsPathRooted($corrRel)) { $corrPath = $corrRel } else { $corrPath = Join-Path $PSScriptRoot $corrRel }
    if (-not (Test-Path $corrPath)) {
        Write-Host "[run_demo] missing corridor_prob.nc, generating placeholder..." -ForegroundColor Yellow
        $placeholderArgs = @('scripts/make_placeholder_corridor.py')
        $placeholder = Start-Process python -ArgumentList $placeholderArgs -NoNewWindow -Wait -PassThru
        if ($placeholder.ExitCode -ne 0 -or -not (Test-Path $corrPath)) {
            Write-Host "[run_demo] failed to generate corridor placeholder. Run: python scripts/make_placeholder_corridor.py" -ForegroundColor Red
            if ($placeholder.ExitCode -ne 0) {
                exit $placeholder.ExitCode
            } else {
                exit 1
            }
        }
    }

    $planArgs = @(
        '-m','api.cli','plan',
        '--cfg','config/runtime.yaml',
        '--beta','3','--gamma','0.3','--p','1.1','--tidx','0',
        '--beta-a','0.3','--tag','demo_full'
    )

    if (-not ($planArgs -contains '--accident-density')) {
        $accPath = Join-Path $PSScriptRoot 'data_processed/accident_density_static.nc'
        if (Test-Path $accPath) {
            $planArgs += @('--accident-density', 'data_processed/accident_density_static.nc', '--acc-mode', 'static')
        } else {
            Write-Host "[run_demo] 未发现事故密度文件，按无事故风险执行" -ForegroundColor Yellow
        }
    }
    Write-Host "[run_demo] running planner..." -ForegroundColor Cyan
    $proc = Start-Process python -ArgumentList $planArgs -NoNewWindow -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-Host "[run_demo] planner failed with exit code $($proc.ExitCode)" -ForegroundColor Red
        exit $proc.ExitCode
    }

    $reportPath = Join-Path 'outputs' 'run_report_demo_full.json'
    if (Test-Path $reportPath) {
        Write-Host "✅ 规划成功" -ForegroundColor Green
    } else {
        Write-Host "[run_demo] run report not found: $reportPath" -ForegroundColor Yellow
        exit 1
    }

    $imagePath = Join-Path 'outputs' 'route_on_risk_demo_full.png'
    if (Test-Path $imagePath) {
        Write-Host "[run_demo] opening $imagePath" -ForegroundColor Cyan
        Start-Process $imagePath | Out-Null
    } else {
        Write-Host "[run_demo] image not found: $imagePath" -ForegroundColor Yellow
    }

    Write-Host "[run_demo] showing first 10 lines of $reportPath" -ForegroundColor Cyan
    Get-Content -Path $reportPath -TotalCount 10

    Write-Host "[run_demo] 完整报告请查看 outputs 目录" -ForegroundColor Cyan
    exit 0
}
finally {
    Pop-Location
}
