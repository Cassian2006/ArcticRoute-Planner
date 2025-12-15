# CMEMS 近实时数据下载脚本 (PowerShell)
# 支持自动滚动更新

param(
    [int]$IntervalMinutes = 0,  # 0 = 仅执行一次，>0 = 每 N 分钟重复
    [switch]$Loop              # 循环模式
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot

function Invoke-Download {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 开始下载 CMEMS 数据..." -ForegroundColor Green
    
    # 执行 Python 脚本
    python scripts/cmems_download.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 下载成功" -ForegroundColor Green
    } else {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 下载失败 (exit code: $LASTEXITCODE)" -ForegroundColor Red
    }
}

# 执行下载
Invoke-Download

# 如果指定了循环模式
if ($Loop -and $IntervalMinutes -gt 0) {
    Write-Host "[INFO] 进入循环模式，每 $IntervalMinutes 分钟执行一次下载" -ForegroundColor Cyan
    
    while ($true) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 等待 $IntervalMinutes 分钟后进行下一次下载..." -ForegroundColor Yellow
        Start-Sleep -Seconds ($IntervalMinutes * 60)
        Invoke-Download
    }
}

