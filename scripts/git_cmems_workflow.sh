#!/bin/bash
# CMEMS 集成 Git 工作流脚本

set -e

echo "=========================================="
echo "CMEMS 集成 Git 工作流"
echo "=========================================="

# 1. 创建分支
echo ""
echo "[1/5] 创建分支 feat/cmems-planner-integration..."
git checkout -b feat/cmems-planner-integration 2>/dev/null || git checkout feat/cmems-planner-integration

# 2. 运行测试
echo ""
echo "[2/5] 运行测试..."
python -m pytest tests/test_cmems_planner_integration.py -v --tb=short

# 3. 添加所有更改
echo ""
echo "[3/5] 添加所有更改..."
git add -A
git status

# 4. 提交
echo ""
echo "[4/5] 提交更改..."
git commit -m "feat: integrate CMEMS near-real-time env into planner pipeline (core+ui+tests)"

# 5. 推送
echo ""
echo "[5/5] 推送到 GitHub..."
git push -u origin feat/cmems-planner-integration

echo ""
echo "=========================================="
echo "✅ Git 工作流完成！"
echo "=========================================="
echo ""
echo "后续步骤："
echo "1. 访问 GitHub: https://github.com/your-repo/pulls"
echo "2. 创建 Pull Request，合并到 main 分支"
echo "3. 等待 CI/CD 通过"
echo "4. 合并 PR"
echo ""

