# Phase 9.1 诊断脚本 - 定位 nextsim HM describe 空文件问题
# 目标：抓取 stderr + exit code，找出真实错误

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 9.1 诊断：nextsim HM describe 问题" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"

# 创建 reports 目录
$reportsDir = "reports"
if (-not (Test-Path $reportsDir)) {
    New-Item -ItemType Directory -Path $reportsDir | Out-Null
}

# 定义文件路径
$TMP = "$reportsDir\cmems_sic_describe.nextsim.tmp.txt"
$OUT = "$reportsDir\cmems_sic_describe.nextsim.json"
$LOG = "$reportsDir\cmems_sic_describe.nextsim.log"
$EXITCODE_FILE = "$reportsDir\cmems_sic_describe.nextsim.exitcode.txt"

Write-Host "[1/4] 执行 describe 命令并捕获 stdout+stderr..." -ForegroundColor Yellow
Write-Host "命令: copernicusmarine describe --contains cmems_mod_arc_phy_anfc_nextsim_hm --return-fields datasets,variables" -ForegroundColor Cyan
Write-Host ""

# 执行命令，同时捕获 stdout 和 stderr
$output = copernicusmarine describe --contains cmems_mod_arc_phy_anfc_nextsim_hm --return-fields datasets,variables 2>&1
$exitCode = $LASTEXITCODE

# 写入日志
$output | Tee-Object -FilePath $LOG | Out-File -Encoding UTF8 $TMP
$exitCode | Out-File -Encoding UTF8 $EXITCODE_FILE

$tmpSize = (Get-Item $TMP).Length
Write-Host "✓ 输出已保存到 $TMP (大小: $tmpSize 字节)" -ForegroundColor Green
Write-Host "✓ 日志已保存到 $LOG" -ForegroundColor Green
Write-Host "✓ 退出码已保存到 $EXITCODE_FILE (值: $exitCode)" -ForegroundColor Green
Write-Host ""

# 检查输出内容
Write-Host "[2/4] 分析输出内容..." -ForegroundColor Yellow
if ($tmpSize -eq 0) {
    Write-Host "⚠ 输出为空（0 字节）" -ForegroundColor Red
} elseif ($tmpSize -lt 100) {
    Write-Host "⚠ 输出过短（$tmpSize 字节），可能是错误信息" -ForegroundColor Yellow
    Write-Host "内容预览：" -ForegroundColor Cyan
    Get-Content $TMP | Select-Object -First 10
} else {
    Write-Host "✓ 输出长度合理（$tmpSize 字节）" -ForegroundColor Green
    # 尝试解析为 JSON
    try {
        $json = Get-Content $TMP -Raw | ConvertFrom-Json
        Write-Host "✓ 内容是有效的 JSON" -ForegroundColor Green
        Write-Host "  - 数据集数: $($json.datasets.Count)" -ForegroundColor Cyan
        Write-Host "  - 变量数: $($json.variables.Count)" -ForegroundColor Cyan
    } catch {
        Write-Host "✗ 内容不是有效的 JSON" -ForegroundColor Red
        Write-Host "内容预览：" -ForegroundColor Cyan
        Get-Content $TMP | Select-Object -First 20
    }
}
Write-Host ""

# 如果输出看起来有效，则原子替换为 .json
Write-Host "[3/4] 决定是否替换为 .json 文件..." -ForegroundColor Yellow
if ($tmpSize -ge 1000) {
    Write-Host "✓ 输出足够大（>= 1000 字节），执行原子替换" -ForegroundColor Green
    Move-Item -Force $TMP $OUT
    Write-Host "✓ 已将 $TMP 替换为 $OUT" -ForegroundColor Green
    $outSize = (Get-Item $OUT).Length
    Write-Host "  最终文件大小: $outSize 字节" -ForegroundColor Cyan
} else {
    Write-Host "✗ 输出过短（< 1000 字节），保留 .tmp 文件用于诊断" -ForegroundColor Yellow
    Write-Host "  请检查 $LOG 了解详细错误信息" -ForegroundColor Cyan
}
Write-Host ""

# 执行兜底检索
Write-Host "[4/4] 执行兜底检索，确认关键词匹配..." -ForegroundColor Yellow
Write-Host ""

Write-Host "  4a) 检索 'nextsim' 关键词..." -ForegroundColor Cyan
$probeNextsim = copernicusmarine describe --contains nextsim --return-fields datasets 2>&1
$probeNextsimExitCode = $LASTEXITCODE
$probeNextsim | Out-File -Encoding UTF8 "$reportsDir\cmems_sic_probe_nextsim.txt"
Write-Host "     ✓ 结果保存到 cmems_sic_probe_nextsim.txt (退出码: $probeNextsimExitCode)" -ForegroundColor Green
Write-Host ""

Write-Host "  4b) 检索产品 ID 'ARCTIC_ANALYSISFORECAST_PHY_ICE_002_011'..." -ForegroundColor Cyan
$probeProduct = copernicusmarine describe --contains ARCTIC_ANALYSISFORECAST_PHY_ICE_002_011 --return-fields datasets 2>&1
$probeProductExitCode = $LASTEXITCODE
$probeProduct | Out-File -Encoding UTF8 "$reportsDir\cmems_sic_probe_product.txt"
Write-Host "     ✓ 结果保存到 cmems_sic_probe_product.txt (退出码: $probeProductExitCode)" -ForegroundColor Green
Write-Host ""

# 生成诊断报告
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "诊断报告" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "主查询结果：" -ForegroundColor Yellow
Write-Host "  - 命令: copernicusmarine describe --contains cmems_mod_arc_phy_anfc_nextsim_hm" -ForegroundColor White
Write-Host "  - 退出码: $exitCode" -ForegroundColor White
Write-Host "  - 输出大小: $tmpSize 字节" -ForegroundColor White
Write-Host "  - 输出文件: $TMP (或 $OUT 如果已替换)" -ForegroundColor White
Write-Host ""

Write-Host "兜底检索结果：" -ForegroundColor Yellow
Write-Host "  - nextsim 关键词: 退出码 $probeNextsimExitCode" -ForegroundColor White
Write-Host "  - 产品 ID: 退出码 $probeProductExitCode" -ForegroundColor White
Write-Host ""

Write-Host "后续步骤：" -ForegroundColor Cyan
Write-Host "1. 查看 $LOG 了解具体错误信息" -ForegroundColor White
Write-Host "2. 查看 $reportsDir\cmems_sic_probe_*.txt 了解兜底检索结果" -ForegroundColor White
Write-Host "3. 如果主查询失败，检查：" -ForegroundColor White
Write-Host "   - Copernicus 服务是否在线" -ForegroundColor White
Write-Host "   - 网络连接和代理设置" -ForegroundColor White
Write-Host "   - copernicusmarine CLI 版本是否最新" -ForegroundColor White
Write-Host ""

