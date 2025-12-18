# Phase 9.1: nextsim HM 诊断脚本
# 抓取 stdout + stderr + exit code，定位空文件问题

$ErrorActionPreference = "Continue"

Write-Host "=== Phase 9.1 nextsim HM Diagnostic Script ===" -ForegroundColor Cyan

# 确保 reports 目录存在
New-Item -ItemType Directory -Force -Path "reports" | Out-Null

# 1) 尝试运行 copernicusmarine describe 命令（nextsim 产品）
Write-Host "`n[1/4] Running copernicusmarine describe for nextsim product..." -ForegroundColor Yellow

$nextsimProduct = "cmems_mod_arc_phy_anfc_nextsim_hm"
$nextsimDataset = "cmems_mod_arc_phy_anfc_6km_detided_PT1H-i"

# 运行命令并捕获输出
$nextsimLog = "reports/cmems_sic_describe.nextsim.log"
$nextsimTmp = "reports/cmems_sic_describe.nextsim.tmp.txt"
$nextsimJson = "reports/cmems_sic_describe.nextsim.json"
$nextsimExitCode = "reports/cmems_sic_describe.nextsim.exitcode.txt"

try {
    # 执行命令并重定向输出
    $process = Start-Process -FilePath "copernicusmarine" `
        -ArgumentList "describe", "--contains", "siconc", "--contains", "nextsim" `
        -RedirectStandardOutput $nextsimTmp `
        -RedirectStandardError $nextsimLog `
        -NoNewWindow -PassThru -Wait
    
    $exitCode = $process.ExitCode
    $exitCode | Out-File -FilePath $nextsimExitCode -Encoding utf8
    
    Write-Host "  Exit code: $exitCode" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Red" })
    
    # 检查输出文件大小
    if (Test-Path $nextsimTmp) {
        $fileSize = (Get-Item $nextsimTmp).Length
        Write-Host "  Output file size: $fileSize bytes" -ForegroundColor $(if ($fileSize -gt 1000) { "Green" } else { "Yellow" })
        
        # 如果输出看起来是 JSON，复制到 .json 文件
        if ($fileSize -gt 100) {
            Copy-Item $nextsimTmp $nextsimJson -Force
        }
    }
} catch {
    Write-Host "  Error executing command: $_" -ForegroundColor Red
    "ERROR: $_" | Out-File -FilePath $nextsimLog -Encoding utf8
    "-1" | Out-File -FilePath $nextsimExitCode -Encoding utf8
}

# 2) 探测 nextsim 产品可用性
Write-Host "`n[2/4] Probing nextsim product availability..." -ForegroundColor Yellow

$nextsimProbe = "reports/cmems_sic_probe_nextsim.txt"

try {
    $probeCmd = "copernicusmarine describe --contains nextsim 2>&1"
    $probeOutput = Invoke-Expression $probeCmd
    $probeOutput | Out-File -FilePath $nextsimProbe -Encoding utf8
    
    if ($probeOutput -match "cmems_mod_arc_phy_anfc_nextsim_hm") {
        Write-Host "  nextsim product FOUND" -ForegroundColor Green
    } else {
        Write-Host "  nextsim product NOT FOUND" -ForegroundColor Red
    }
} catch {
    Write-Host "  Probe failed: $_" -ForegroundColor Red
    "ERROR: $_" | Out-File -FilePath $nextsimProbe -Encoding utf8
}

# 3) 探测标准 L4 产品（回退选项）
Write-Host "`n[3/4] Probing standard L4 product (fallback)..." -ForegroundColor Yellow

$productProbe = "reports/cmems_sic_probe_product.txt"

try {
    $productCmd = "copernicusmarine describe --contains siconc --contains ARCTIC 2>&1"
    $productOutput = Invoke-Expression $productCmd
    $productOutput | Out-File -FilePath $productProbe -Encoding utf8
    
    if ($productOutput -match "ARCTIC_ANALYSISFORECAST_PHY_ICE") {
        Write-Host "  L4 product FOUND (fallback available)" -ForegroundColor Green
    } else {
        Write-Host "  L4 product status unclear" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  Probe failed: $_" -ForegroundColor Red
    "ERROR: $_" | Out-File -FilePath $productProbe -Encoding utf8
}

# 4) 总结
Write-Host "`n[4/4] Diagnostic Summary:" -ForegroundColor Yellow
Write-Host "  - nextsim describe log: $nextsimLog"
Write-Host "  - nextsim describe output: $nextsimTmp"
Write-Host "  - nextsim describe JSON: $nextsimJson"
Write-Host "  - nextsim describe exit code: $nextsimExitCode"
Write-Host "  - nextsim probe: $nextsimProbe"
Write-Host "  - L4 product probe: $productProbe"

Write-Host "`n=== Diagnostic Complete ===" -ForegroundColor Cyan
Write-Host "Next: Run Python script to generate PHASE11_NEXTSIM_DIAG_SUMMARY.txt" -ForegroundColor Green

