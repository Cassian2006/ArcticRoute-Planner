# Phase 9 收口脚本 - 确保 PR 可合并
# 执行步骤：
# 1. 确认没有误提交数据/缓存
# 2. 检查 diff 统计
# 3. 还原不必要的 __init__.py 格式调整
# 4. 运行完整测试
# 5. 提交并推送

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 9 收口：PR 合并前检查" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"

# 步骤 1: 确认没有误提交数据/缓存
Write-Host "[1/5] 检查是否有误提交的数据文件..." -ForegroundColor Yellow
$dataFiles = git ls-files | Select-String "data/cmems_cache|ArcticRoute/data_processed|reports/cmems_" | Measure-Object -Line
if ($dataFiles.Lines -eq 0) {
    Write-Host "✓ 确认：没有误提交的数据文件" -ForegroundColor Green
} else {
    Write-Host "✗ 警告：发现 $($dataFiles.Lines) 个数据文件被追踪" -ForegroundColor Red
    git ls-files | Select-String "data/cmems_cache|ArcticRoute/data_processed|reports/cmems_"
}
Write-Host ""

# 步骤 2: 检查 diff 统计
Write-Host "[2/5] 检查 diff 统计..." -ForegroundColor Yellow
Write-Host "当前分支相对于 origin/main 的改动：" -ForegroundColor Cyan
$diffStat = git diff --stat origin/main...HEAD 2>&1
$lines = $diffStat | Measure-Object -Line
Write-Host "$($lines.Lines) 行统计信息"
Write-Host ""

# 提取文件数和行数
$summary = $diffStat | Select-Object -Last 1
Write-Host "摘要：$summary" -ForegroundColor Cyan
Write-Host ""

# 步骤 3: 检查 __init__.py 改动
Write-Host "[3/5] 检查 __init__.py 改动..." -ForegroundColor Yellow
$initFiles = @(
    "ArcticRoute/__init__.py",
    "ArcticRoute/core/__init__.py",
    "ArcticRoute/core/eco/__init__.py"
)

$hasInitChanges = $false
foreach ($file in $initFiles) {
    $diff = git diff origin/main...HEAD -- $file 2>&1
    if ($diff) {
        Write-Host "  - $file 有改动" -ForegroundColor Cyan
        $hasInitChanges = $true
    }
}

if ($hasInitChanges) {
    Write-Host "是否还原这些 __init__.py 的格式调整？(y/n)" -ForegroundColor Yellow
    $response = Read-Host
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "还原 __init__.py 文件..." -ForegroundColor Yellow
        git checkout -- $initFiles
        Write-Host "✓ 已还原" -ForegroundColor Green
    }
} else {
    Write-Host "✓ 没有 __init__.py 改动需要还原" -ForegroundColor Green
}
Write-Host ""

# 步骤 4: 运行测试
Write-Host "[4/5] 运行完整测试..." -ForegroundColor Yellow
Write-Host "执行: python -m pytest -q" -ForegroundColor Cyan
python -m pytest -q
$testResult = $LASTEXITCODE

if ($testResult -eq 0) {
    Write-Host "✓ 所有测试通过" -ForegroundColor Green
} else {
    Write-Host "✗ 测试失败（退出码: $testResult）" -ForegroundColor Red
    Write-Host "请修复测试后再继续" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# 步骤 5: 提交并推送
Write-Host "[5/5] 提交并推送..." -ForegroundColor Yellow
git add -A
$status = git status --short
if ($status) {
    Write-Host "待提交的改动：" -ForegroundColor Cyan
    Write-Host $status
    Write-Host ""
    Write-Host "执行: git commit -m 'chore: reduce diff noise (revert formatting-only __init__ changes)'" -ForegroundColor Cyan
    git commit -m "chore: reduce diff noise (revert formatting-only __init__ changes)" 2>&1 | Out-Null
    Write-Host "✓ 已提交" -ForegroundColor Green
} else {
    Write-Host "✓ 没有待提交的改动" -ForegroundColor Green
}

Write-Host ""
Write-Host "执行: git push" -ForegroundColor Cyan
git push
$pushResult = $LASTEXITCODE

if ($pushResult -eq 0) {
    Write-Host "✓ 已推送到远程" -ForegroundColor Green
} else {
    Write-Host "✗ 推送失败（退出码: $pushResult）" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Phase 9 收口完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "后续步骤：" -ForegroundColor Cyan
Write-Host "1. 访问 GitHub: https://github.com/Cassian2006/ArcticRoute-Planner" -ForegroundColor White
Write-Host "2. 创建 PR 从当前分支到 main" -ForegroundColor White
Write-Host "3. 填写 PR 描述（包含验收点、测试结果等）" -ForegroundColor White
Write-Host ""

