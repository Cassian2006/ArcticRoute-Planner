$ErrorActionPreference = 'Stop'
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot

function Join-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Rel
    )

    return (Join-Path -Path $PSScriptRoot -ChildPath $Rel)
}

function Assert-FilesExist {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $RelPaths,

        [string] $Stage = ''
    )

    foreach ($rel in $RelPaths) {
        $full = Join-ProjectPath $rel
        if (-not (Test-Path -LiteralPath $full)) {
            $msg = "[ai_demo] missing file: $rel"
            if ($Stage) {
                $msg = "$msg (stage=$Stage)"
            }
            throw $msg
        }
    }
}

function Resolve-First {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Pattern
    )

    $r = Get-ChildItem -LiteralPath $Pattern -ErrorAction SilentlyContinue
    if ($null -eq $r) {
        return $null
    }
    return ($r | Select-Object -First 1 -ExpandProperty FullName)
}

function Ensure-PlaceholderScript {
    param(
        [Parameter(Mandatory = $true)]
        [string] $ScriptPath
    )

    if (Test-Path -LiteralPath $ScriptPath) {
        return
    }

    Write-Host ("占位脚本缺失，正在自动创建：{0}" -f $ScriptPath) -ForegroundColor Yellow

    $placeholderSource = @'
# -*- coding: utf-8 -*-
"""Generate a placeholder corridor_prob.nc aligned with env_clean.nc."""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

import numpy as np


def _write_with_xarray(env_path: pathlib.Path, output_path: pathlib.Path) -> bool:
    try:
        import xarray as xr  # type: ignore
    except Exception:
        return False

    with xr.open_dataset(env_path) as ds:
        data_vars: Iterable[str] = list(ds.data_vars)
        if not data_vars:
            raise RuntimeError("env dataset has no data variables to infer dimensions from")
        key = data_vars[0]
        zeros = xr.zeros_like(ds[key]).astype("float32")
        ds_out = zeros.to_dataset(name="corridor_prob")
        ds_out.to_netcdf(output_path)
    return True


def _write_with_netcdf4(env_path: pathlib.Path, output_path: pathlib.Path) -> None:
    try:
        from netCDF4 import Dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - fallback path
        raise RuntimeError("xarray/netCDF4 unavailable, cannot create placeholder corridor") from exc

    with Dataset(env_path, "r") as src, Dataset(output_path, "w") as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

        for name, var in src.variables.items():
            if len(var.dimensions) == 1:
                out_var = dst.createVariable(name, var.datatype, var.dimensions)
                out_var.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs()})
                out_var[:] = var[:]

        template = None
        for name, var in src.variables.items():
            if len(var.dimensions) >= 1 and name not in dst.variables:
                template = var
                break

        if template is None:
            raise RuntimeError("Unable to infer corridor dimensions from env_clean.nc")

        corr = dst.createVariable("corridor_prob", "f4", template.dimensions, zlib=False)
        corr.setncatts(
            {
                "long_name": "Corridor probability (placeholder)",
                "description": "Auto-generated zero corridor for AI demo self-heal",
            }
        )
        corr[:] = np.zeros(template.shape, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create placeholder corridor_prob.nc")
    parser.add_argument("--env-path", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    env_path = args.env_path.resolve()
    output_path = args.output.resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _write_with_xarray(env_path, output_path):
        _write_with_netcdf4(env_path, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"[placeholder] failed: {exc}", file=sys.stderr)
        sys.exit(1)
'@

    New-Item -ItemType Directory -Path (Split-Path -Parent $ScriptPath) -Force | Out-Null
    Set-Content -LiteralPath $ScriptPath -Value $placeholderSource -Encoding UTF8
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string] $CommandString,

        [Parameter(Mandatory = $true)]
        [string] $LogPath
    )

    Write-Host ("执行命令：{0}" -f $CommandString)
    Remove-Item -LiteralPath $LogPath -ErrorAction SilentlyContinue
    cmd.exe /c $CommandString 2>&1 | Tee-Object -FilePath $LogPath | Out-Null
    if (Test-Path -LiteralPath $LogPath) {
        $script:LastCommandLines = Get-Content -LiteralPath $LogPath -Encoding UTF8
    } else {
        $script:LastCommandLines = @()
    }
    return $LASTEXITCODE
}

$script:LastCommandLines = @()

$AccFixCommands = @(
    'python scripts\incidents_ingest.py ...',
    'python scripts\incidents_align_to_grid.py ...',
    'python scripts\accident_density_grid.py ...'
)

function Show-StepFailure {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Stage,

        [Parameter(Mandatory = $true)]
        [System.Exception] $Error,

        [string[]] $Suggestions
    )

    Write-Host ("{0}阶段失败：{1}" -f $Stage, $Error.Message) -ForegroundColor Red
    if ($Suggestions) {
        foreach ($suggestion in $Suggestions) {
            if (-not [string]::IsNullOrWhiteSpace($suggestion)) {
                Write-Host $suggestion -ForegroundColor Yellow
            }
        }
    }
    exit 1
}

$runtimePath = Join-ProjectPath 'config\runtime.yaml'
$envCleanPath = Join-ProjectPath 'data_processed\env_clean.nc'
$defaultCorridorPath = Join-ProjectPath 'data_processed\corridor_prob.nc'
$accidentStaticPath = Join-ProjectPath 'data_processed\accident_density_static.nc'
$outputsDir = Join-ProjectPath 'outputs'

$fatalIssues = @()

if (-not (Test-Path -LiteralPath $runtimePath)) {
    $fatalIssues += @{
        Color = 'Red'
        Message = "缺少 config\runtime.yaml，请先准备运行时配置。"
    }
}

if (-not (Test-Path -LiteralPath $envCleanPath)) {
    $fatalIssues += @{
        Color = 'Yellow'
        Message = "缺少 data_processed\env_clean.nc，请运行 scripts\maintenance\fix_env_path.py 后重试。"
        Suggestions = @('python scripts/maintenance/fix_env_path.py')
    }
}

if ($fatalIssues.Count -gt 0) {
    foreach ($issue in $fatalIssues) {
        Write-Host $issue.Message -ForegroundColor $issue.Color
        if ($issue.ContainsKey('Suggestions') -and $issue.Suggestions) {
            foreach ($cmd in $issue.Suggestions) {
                if (-not [string]::IsNullOrWhiteSpace($cmd)) {
                    Write-Host ("建议命令：{0}" -f $cmd) -ForegroundColor Yellow
                }
            }
        }
    }
    exit 1
}

if ([string]::IsNullOrWhiteSpace($env:MOONSHOT_API_KEY)) {
    Write-Host "未检测到 MOONSHOT_API_KEY，将使用规则版建议（不调用 LLM）。" -ForegroundColor Yellow
    $UseLLM = $false
} else {
    Write-Host "检测到 MOONSHOT_API_KEY，将启用 LLM 建议。" -ForegroundColor Green
    $UseLLM = $true
}

$corridorPath = $defaultCorridorPath
$corridorPathFromConfig = $null

try {
    $probeScript = @'
import json
import pathlib
import sys
try:
    import yaml  # type: ignore
except Exception:
    print(json.dumps({"corridor": None}))
    sys.exit(0)
runtime_path = pathlib.Path(sys.argv[1])
try:
    with runtime_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
except Exception:
    print(json.dumps({"corridor": None}))
    sys.exit(0)
data_cfg = config.get("data") or {}
corridor_value = data_cfg.get("corridor_prob")
print(json.dumps({"corridor": corridor_value}))
'@
    $tempProbe = [System.IO.Path]::GetTempFileName()
    Set-Content -LiteralPath $tempProbe -Value $probeScript -Encoding UTF8
    $probeOutput = & python $tempProbe $runtimePath
    $probeExit = $LASTEXITCODE
    Remove-Item -LiteralPath $tempProbe -Force
    if ($probeExit -eq 0 -and -not [string]::IsNullOrWhiteSpace($probeOutput)) {
        try {
            $probeJson = $probeOutput.Trim() | ConvertFrom-Json
            if ($null -ne $probeJson.corridor -and -not [string]::IsNullOrWhiteSpace([string]$probeJson.corridor)) {
                $corridorPathFromConfig = [string]$probeJson.corridor
            }
        } catch {
            # 忽略 JSON 解析错误
        }
    }
} catch {
    # 忽略配置解析异常
}

if (-not [string]::IsNullOrWhiteSpace($corridorPathFromConfig)) {
    try {
        if ([System.IO.Path]::IsPathRooted($corridorPathFromConfig)) {
            $corridorPath = [System.IO.Path]::GetFullPath($corridorPathFromConfig)
        } else {
            $corridorPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $corridorPathFromConfig))
        }
    } catch {
        $corridorPath = Join-ProjectPath $corridorPathFromConfig
    }
}

$corridorAvailable = $false
if (Test-Path -LiteralPath $corridorPath) {
    $corridorAvailable = $true
} else {
    Write-Host "[ai_demo] missing corridor_prob.nc, generating placeholder..." -ForegroundColor Yellow
    $corridorDir = Split-Path -Path $corridorPath -Parent
    if (-not [string]::IsNullOrWhiteSpace($corridorDir) -and -not (Test-Path -LiteralPath $corridorDir)) {
        New-Item -ItemType Directory -Path $corridorDir -Force | Out-Null
    }

    $placeholderScriptPath = Join-ProjectPath 'scripts\make_placeholder_corridor.py'
    Ensure-PlaceholderScript -ScriptPath $placeholderScriptPath

    $placeholderCmd = "python `"$placeholderScriptPath`" --env-path `"$envCleanPath`" --output `"$corridorPath`""
    $placeholderOutput = @()
    Invoke-Expression "$placeholderCmd 2>&1" | Tee-Object -Variable placeholderOutput | Out-Null
    $placeholderExit = $LASTEXITCODE
    if ($placeholderExit -ne 0) {
        Show-StepFailure -Stage "走廊卷自查" -Error ([System.Exception]::new(("占位走廊生成失败，退出码 {0}" -f $placeholderExit))) -Suggestions @()
    }

    if (Test-Path -LiteralPath $corridorPath) {
        $corridorAvailable = $true
    } else {
        Show-StepFailure -Stage "走廊卷自查" -Error ([System.Exception]::new(("仍未找到走廊文件：{0}" -f $corridorPath))) -Suggestions @()
    }
}

$AccArg = ''
$accidentAvailable = $false
if (Test-Path -LiteralPath $accidentStaticPath) {
    $AccArg = '--accident-density data_processed\accident_density_static.nc --acc-mode static'
    $accidentAvailable = $true
    Write-Host "已自动启用事故密度：data_processed\accident_density_static.nc"
} else {
    Write-Host "未发现静态事故密度文件，将跳过事故相关参数。" -ForegroundColor Yellow
    Write-Host "[hint] 若需启用事故密度可运行：" -ForegroundColor Yellow
    foreach ($cmd in $AccFixCommands) {
        Write-Host ("  {0}" -f $cmd) -ForegroundColor Yellow
    }
}

Write-Host "即将执行的配置摘要："
Write-Host ("  UseLLM: {0}" -f $(if ($UseLLM) { "是" } else { "否" }))
Write-Host ("  事故密度: {0}" -f $(if ($accidentAvailable) { "已配置" } else { "未配置" }))
Write-Host ("  走廊文件: {0}" -f $(if ($corridorAvailable) { $corridorPath } else { "缺失" }))
Write-Host ("  配置文件: {0}" -f (Resolve-Path -LiteralPath $runtimePath))

$logDirectory = Join-Path $outputsDir 'logs'
if (-not (Test-Path -LiteralPath $logDirectory)) {
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
}

$gammaValue = if ($corridorAvailable) { '0.3' } else { '0' }
$baselineLog = Join-Path $logDirectory ("base_ai_demo_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

try {
    Write-Host ''
    Write-Host ("基线规划日志将写入：{0}" -f $baselineLog)

    $accSegment = if ([string]::IsNullOrWhiteSpace($AccArg)) { '' } else { " $AccArg" }
    $planBase = "python -m api.cli plan --cfg config/runtime.yaml --beta 3 --gamma $gammaValue --p 1.1 --tidx 0$accSegment --tag base_ai_demo"

    $baselineExit = Invoke-LoggedCommand -CommandString $planBase -LogPath $baselineLog
    if ($baselineExit -ne 0) {
        throw ("基线规划命令失败，退出码 {0}（查看日志：{1}）" -f $baselineExit, $baselineLog)
    }

    $BaseReportRel = 'outputs\run_report_base_ai_demo.json'
    $BasePngRel = 'outputs\route_on_risk_base_ai_demo.png'
    Assert-FilesExist @($BaseReportRel, $BasePngRel) -Stage 'baseline'

    Write-Host "✅ 基线规划成功" -ForegroundColor Green
} catch {
    $errMessage = $Error[0].Exception.Message
    Write-Host $errMessage -ForegroundColor Red
    $errorLogPath = Join-ProjectPath 'outputs\logs\run_ai_demo_error.log'
    $tail = $script:LastCommandLines | Select-Object -Last 30
    if ($tail) {
        $tail | Out-File -FilePath $errorLogPath -Encoding UTF8
        Write-Host ("错误日志已保存至：{0}" -f $errorLogPath) -ForegroundColor Yellow
    }
    if (-not (Test-Path -LiteralPath $envCleanPath)) {
        Write-Host "[hint] 可先运行：python scripts\maintenance\fix_env_path.py" -ForegroundColor Yellow
    }
    if (-not (Test-Path -LiteralPath $accidentStaticPath)) {
        Write-Host "[hint] 若需启用事故密度可运行：" -ForegroundColor Yellow
        foreach ($cmd in $AccFixCommands) {
            Write-Host ("  {0}" -f $cmd) -ForegroundColor Yellow
        }
    }
    exit 1
}

$advisorLog = Join-Path $logDirectory ("ai_advice_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

$AdvisedBeta = 3
$AdvisedGamma = 0.3
$AdvisedP = 1.1
$AdvisedBetaA = 0.3
$adviceSource = "规则"
$AdviceFromLLM = $false

try {
    Write-Host ''
    Write-Host ("AI 建议日志将写入：{0}" -f $advisorLog)

    $advisorCmd = "python -m api.cli ai advise --cfg config/runtime.yaml --use-llm $UseLLM --tag ai_demo"
    $advisorExit = Invoke-LoggedCommand -CommandString $advisorCmd -LogPath $advisorLog
    if ($advisorExit -ne 0) {
        throw ("AI 建议命令失败，退出码 {0}（查看日志：{1}）" -f $advisorExit, $advisorLog)
    }

    $advicePathRel = 'outputs\advice_ai_demo.json'
    $advicePath = Join-ProjectPath $advicePathRel
    if (Test-Path -LiteralPath $advicePath) {
        try {
            $adviceRaw = Get-Content -LiteralPath $advicePath -Raw -Encoding UTF8
            $adviceData = $adviceRaw | ConvertFrom-Json
        } catch {
            Write-Host ("解析 AI 建议 JSON 失败，将使用默认参数。错误：{0}" -f $_.Exception.Message) -ForegroundColor Yellow
            $adviceData = $null
        }
    } else {
        Write-Host ("未找到建议文件：{0}，将使用默认参数。" -f $advicePath) -ForegroundColor Yellow
        $adviceData = $null
    }

    if ($null -ne $adviceData) {
        $hasMissingAdvice = $false

        if ($null -ne $adviceData.beta) {
            $AdvisedBeta = [double]$adviceData.beta
        } else {
            $hasMissingAdvice = $true
        }

        if ($null -ne $adviceData.gamma) {
            $AdvisedGamma = [double]$adviceData.gamma
        } else {
            $hasMissingAdvice = $true
        }

        if ($null -ne $adviceData.p) {
            $AdvisedP = [double]$adviceData.p
        } else {
            $hasMissingAdvice = $true
        }

        if ($null -ne $adviceData.beta_a) {
            $AdvisedBetaA = [double]$adviceData.beta_a
        } else {
            $hasMissingAdvice = $true
        }

        if ($null -ne $adviceData.from_llm) {
            $AdviceFromLLM = [bool]$adviceData.from_llm
            $adviceSource = if ($AdviceFromLLM) { 'LLM' } else { "规则" }
        } else {
            $hasMissingAdvice = $true
        }

        if ($hasMissingAdvice) {
            Write-Host "AI 建议 JSON 缺少部分字段，已回退到默认值。" -ForegroundColor Yellow
        }
    }

    $sourceLabel = if ($AdviceFromLLM) { 'LLM' } else { "规则" }
    Write-Host ("✅ 建议已生成（来源：{0}）" -f $sourceLabel) -ForegroundColor Green
    Write-Host ("采用建议参数：beta={0}, gamma={1}, p={2}, beta_a={3}" -f $AdvisedBeta, $AdvisedGamma, $AdvisedP, $AdvisedBetaA)
} catch {
    $suggestions = @()
    if (-not (Test-Path -LiteralPath $accidentStaticPath)) {
        $suggestions += "[hint] 若需启用事故密度可运行："
        $suggestions += $AccFixCommands
    }
    Show-StepFailure -Stage "获取 AI 建议" -Error $_.Exception -Suggestions $suggestions
}

$advisedLog = Join-Path $logDirectory ("advised_ai_demo_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

try {
    Write-Host ''
    Write-Host ("建议规划日志将写入：{0}" -f $advisedLog)

    $invariant = [System.Globalization.CultureInfo]::InvariantCulture
    $betaStr = [System.String]::Format($invariant, '{0}', $AdvisedBeta)
    $gammaStr = [System.String]::Format($invariant, '{0}', $AdvisedGamma)
    $pStr = [System.String]::Format($invariant, '{0}', $AdvisedP)
    $betaAStr = [System.String]::Format($invariant, '{0}', $AdvisedBetaA)

    $accSegment = if ([string]::IsNullOrWhiteSpace($AccArg)) { '' } else { " $AccArg" }
    $planAdvised = "python -m api.cli plan --cfg config/runtime.yaml --beta $betaStr --gamma $gammaStr --p $pStr --tidx 0$accSegment --beta-a $betaAStr --tag advised_ai_demo"

    $advisedExit = Invoke-LoggedCommand -CommandString $planAdvised -LogPath $advisedLog
    if ($advisedExit -ne 0) {
        throw ("建议参数规划失败，退出码 {0}（查看日志：{1}）" -f $advisedExit, $advisedLog)
    }

    $AdvReportRel = 'outputs\run_report_advised_ai_demo.json'
    $AdvPngRel = 'outputs\route_on_risk_advised_ai_demo.png'
    Assert-FilesExist @($AdvReportRel, $AdvPngRel) -Stage 'advised'

    Write-Host "✅ 建议参数复跑成功" -ForegroundColor Green
} catch {
    $suggestions = @()
    if (-not (Test-Path -LiteralPath $accidentStaticPath)) {
        $suggestions += "[hint] 若需启用事故密度可运行："
        $suggestions += $AccFixCommands
    }
    Show-StepFailure -Stage "建议参数复跑" -Error $_.Exception -Suggestions $suggestions
}

$compareLog = Join-Path $logDirectory ("ai_explain_compare_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$compareOutputPath = Join-Path $outputsDir 'exp_compare_ai_demo.md'

try {
    Write-Host ''
    Write-Host ("对比解释日志将写入：{0}" -f $compareLog)

    $compareCmd = "python -m api.cli ai explain-compare --a outputs/run_report_base_ai_demo.json --b outputs/run_report_advised_ai_demo.json --out outputs/exp_compare_ai_demo.md"
    $compareExit = Invoke-LoggedCommand -CommandString $compareCmd -LogPath $compareLog
    if ($compareExit -ne 0) {
        throw ("对比解释命令失败，退出码 {0}（查看日志：{1}）" -f $compareExit, $compareLog)
    }

    if (-not (Test-Path -LiteralPath $compareOutputPath)) {
        throw ("未找到对比解释文件：{0}" -f $compareOutputPath)
    }

    Write-Host ("✅ 对比解释已生成：{0}" -f $compareOutputPath) -ForegroundColor Green
} catch {
    $suggestions = @()
    if (-not (Test-Path -LiteralPath $accidentStaticPath)) {
        $suggestions += "[hint] 若需启用事故密度可运行："
        $suggestions += $AccFixCommands
    }
    Show-StepFailure -Stage "对比解释" -Error $_.Exception -Suggestions $suggestions
}

function Get-ReportMetrics {
    param(
        [Parameter(Mandatory = $true)]
        [string] $ReportPath
    )

    $defaults = @{
        total_cost = $null
        mean_risk = $null
        geodesic_length_km = $null
        'nearest_accident_km.mean' = $null
    }

    if (-not (Test-Path -LiteralPath $ReportPath)) {
        Write-Host ("指标文件缺失：{0}" -f $ReportPath) -ForegroundColor Yellow
        return $defaults
    }

    try {
        $jsonRaw = Get-Content -LiteralPath $ReportPath -Raw -Encoding UTF8
        $jsonData = $jsonRaw | ConvertFrom-Json
    } catch {
        Write-Host ("读取指标失败：{0}，错误：{1}" -f $ReportPath, $_.Exception.Message) -ForegroundColor Yellow
        return $defaults
    }

    $metrics = @{}

    foreach ($key in $defaults.Keys) {
        switch ($key) {
            'nearest_accident_km.mean' {
                if ($null -ne $jsonData.'nearest_accident_km.mean') {
                    $metrics[$key] = [double]$jsonData.'nearest_accident_km.mean'
                } elseif ($null -ne $jsonData.nearest_accident_km -and $null -ne $jsonData.nearest_accident_km.mean) {
                    $metrics[$key] = [double]$jsonData.nearest_accident_km.mean
                } else {
                    $metrics[$key] = $null
                }
            }
            default {
                if ($null -ne $jsonData.$key) {
                    $metrics[$key] = [double]$jsonData.$key
                } else {
                    $metrics[$key] = $null
                }
            }
        }
    }

    return $metrics
}

$baseMetrics = Get-ReportMetrics -ReportPath (Join-Path $outputsDir 'run_report_base_ai_demo.json')
$advisedMetrics = Get-ReportMetrics -ReportPath (Join-Path $outputsDir 'run_report_advised_ai_demo.json')

$metricPreferences = @{
    total_cost = 'lower'
    mean_risk = 'lower'
    geodesic_length_km = 'lower'
    'nearest_accident_km.mean' = 'higher'
}

$tableRows = @()
$invariant = [System.Globalization.CultureInfo]::InvariantCulture

foreach ($metric in $metricPreferences.Keys) {
    $baseVal = $baseMetrics[$metric]
    $advVal = $advisedMetrics[$metric]
    $trend = '-'

    if ($null -ne $baseVal -and $null -ne $advVal) {
        if ($metricPreferences[$metric] -eq 'lower') {
            if ($advVal -lt $baseVal) {
                $trend = "↑"
            } elseif ($advVal -gt $baseVal) {
                $trend = "↓"
            } else {
                $trend = '='
            }
        } else {
            if ($advVal -gt $baseVal) {
                $trend = "↑"
            } elseif ($advVal -lt $baseVal) {
                $trend = "↓"
            } else {
                $trend = '='
            }
        }
    }

    $baseStr = if ($null -ne $baseVal) { [System.String]::Format($invariant, '{0:0.###}', $baseVal) } else { 'N/A' }
    $advStr = if ($null -ne $advVal) { [System.String]::Format($invariant, '{0:0.###}', $advVal) } else { 'N/A' }

    $tableRows += [PSCustomObject]@{
        "指标" = $metric
        Base = $baseStr
        Advised = $advStr
        "趋势" = $trend
    }
}

Write-Host ''
Write-Host "关键指标对比："
$tableRows | Format-Table -AutoSize | Out-String | Write-Host

Write-Host ''
Write-Host "正在打开相关产物..."

Start-Process -FilePath (Join-Path $outputsDir 'route_on_risk_advised_ai_demo.png') | Out-Null
Start-Process -FilePath $compareOutputPath | Out-Null

