# CMEMS 集成 Git 工作流脚本 (PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "CMEMS 集成 Git 工作流" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. 创建分支
Write-Host ""
Write-Host "[1/5] 创建分支 feat/cmems-planner-integration..." -ForegroundColor Yellow
try {
    git checkout -b feat/cmems-planner-integration 2>$null
} catch {
    git checkout feat/cmems-planner-integration
}

# 2. 运行测试
Write-Host ""
Write-Host "[2/5] 运行测试..." -ForegroundColor Yellow
python -m pytest tests/test_cmems_planner_integration.py -v --tb=short
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 测试失败！" -ForegroundColor Red
    exit 1
}

# 3. 添加所有更改
Write-Host ""
Write-Host "[3/5] 添加所有更改..." -ForegroundColor Yellow
git add -A
git status

# 4. 提交
Write-Host ""
Write-Host "[4/5] 提交更改..." -ForegroundColor Yellow
git commit -m "feat: integrate CMEMS near-real-time env into planner pipeline (core+ui+tests)"

# 5. 推送
Write-Host ""
Write-Host "[5/5] 推送到 GitHub..." -ForegroundColor Yellow
git push -u origin feat/cmems-planner-integration

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ Git 工作流完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "后续步骤:" -ForegroundColor Cyan
Write-Host "1. 访问 GitHub: https://github.com/your-repo/pulls"
Write-Host "2. 创建 Pull Request，合并到 main 分支"
Write-Host "3. 等待 CI/CD 通过"
Write-Host "4. 合并 PR"
Write-Host ""

