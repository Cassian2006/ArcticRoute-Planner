# CMEMS 集成部署指南

## 📋 前置条件

- Python 3.8+
- Git
- Copernicus Marine Toolbox (`copernicusmarine` CLI)
- CMEMS 账户（可选，但推荐用于完整功能）

## 🔧 安装步骤

### 1. 安装 Copernicus Marine Toolbox

```bash
# 使用 pip
pip install copernicusmarine

# 验证安装
copernicusmarine --version
```

### 2. 配置 CMEMS 认证（可选）

```bash
# 设置环境变量（Linux/macOS）
export COPERNICUSMARINE_USERNAME=your_username
export COPERNICUSMARINE_PASSWORD=your_password

# 或 PowerShell (Windows)
$env:COPERNICUSMARINE_USERNAME = "your_username"
$env:COPERNICUSMARINE_PASSWORD = "your_password"

# 或使用交互式登录
copernicusmarine login
```

### 3. 克隆或更新项目

```bash
# 如果还没有克隆
git clone https://github.com/your-repo/ArcticRoute.git
cd ArcticRoute

# 如果已有项目，更新到最新
git pull origin main
```

### 4. 创建新分支

```bash
git checkout -b feat/cmems-planner-integration
```

### 5. 安装项目依赖

```bash
pip install -r requirements.txt
```

---

## 🚀 快速部署

### 方案 A：自动化部署（推荐）

#### Linux/macOS
```bash
bash scripts/git_cmems_workflow.sh
```

#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/git_cmems_workflow.ps1
```

### 方案 B：手动部署

#### 步骤 1：生成 Describe JSON
```bash
python scripts/gen_describe_json.py
```

**验证**:
```bash
# 检查文件是否生成
ls -lh reports/cmems_*_describe.json

# 检查文件大小（应该 > 1 KB）
wc -l reports/cmems_sic_describe.json
```

#### 步骤 2：解析变量
```bash
python scripts/cmems_resolve.py
```

**验证**:
```bash
# 查看生成的配置
cat reports/cmems_resolved.json

# 应该包含 sic 和 wav 配置
```

#### 步骤 3：刷新数据（可选）
```bash
# 下载最近 2 天的数据
python scripts/cmems_refresh_and_export.py --days 2
```

**验证**:
```bash
# 检查下载的文件
ls -lh data/cmems_cache/

# 检查刷新记录
cat reports/cmems_refresh_last.json
```

#### 步骤 4：同步到 Newenv
```bash
python scripts/cmems_newenv_sync.py
```

**验证**:
```bash
# 检查 newenv 目录
ls -lh ArcticRoute/data_processed/newenv/
```

#### 步骤 5：运行测试
```bash
pytest tests/test_cmems_planner_integration.py -v
```

**预期输出**:
```
test_find_latest_nc PASSED
test_find_latest_nc_not_found PASSED
test_get_sic_variable PASSED
test_get_swh_variable PASSED
test_sync_to_newenv PASSED
test_sync_to_newenv_partial PASSED
test_cmems_latest_routing PASSED
test_fallback_to_real_archive PASSED
test_pick_function PASSED

====== 9 passed in 0.45s ======
```

#### 步骤 6：提交和推送
```bash
# 添加所有更改
git add -A

# 提交
git commit -m "feat: integrate CMEMS near-real-time env into planner pipeline (core+ui+tests)"

# 推送
git push -u origin feat/cmems-planner-integration
```

#### 步骤 7：创建 Pull Request
1. 访问 GitHub 项目页面
2. 点击 "New Pull Request"
3. 选择 `feat/cmems-planner-integration` → `main`
4. 填写 PR 描述
5. 点击 "Create Pull Request"

---

## 🧪 验证部署

### 检查清单

- [ ] Describe JSON 文件已生成（非空）
- [ ] cmems_resolved.json 包含正确的变量
- [ ] 测试全部通过
- [ ] Git 分支已推送
- [ ] PR 已创建

### 运行完整验证

```bash
# 1. 检查文件
echo "=== 检查 Describe JSON ==="
ls -lh reports/cmems_*_describe.json

# 2. 检查配置
echo "=== 检查 cmems_resolved.json ==="
cat reports/cmems_resolved.json

# 3. 运行测试
echo "=== 运行测试 ==="
pytest tests/test_cmems_planner_integration.py -v

# 4. 检查 Git 状态
echo "=== 检查 Git 状态 ==="
git status
git log --oneline -5
```

---

## 🔍 故障排查

### 问题 1：Describe JSON 为空

**症状**: `reports/cmems_sic_describe.json` 文件存在但为空

**解决方案**:
```bash
# 检查 copernicusmarine 是否正确安装
copernicusmarine --version

# 检查网络连接
ping -c 1 api.marine.copernicus.eu

# 手动运行 describe 命令
copernicusmarine describe --contains cmems_mod_arc_phy_anfc_nextsim_hm --return-fields all
```

### 问题 2：变量解析失败

**症状**: `cmems_resolved.json` 为空或缺少变量

**解决方案**:
```bash
# 检查 describe JSON 是否有效
python -c "import json; json.load(open('reports/cmems_sic_describe.json'))"

# 手动运行解析脚本
python scripts/cmems_resolve.py --debug
```

### 问题 3：数据下载失败

**症状**: `cmems_refresh_and_export.py` 返回错误

**解决方案**:
```bash
# 检查 CMEMS 认证
copernicusmarine login

# 检查网络连接和防火墙
copernicusmarine describe --contains cmems_mod_arc_phy_anfc_nextsim_hm

# 增加超时时间
python scripts/cmems_refresh_and_export.py --days 1 --timeout 600
```

### 问题 4：测试失败

**症状**: `pytest` 返回失败

**解决方案**:
```bash
# 运行单个测试以获取更多信息
pytest tests/test_cmems_planner_integration.py::TestCMEMSDataLoading::test_find_latest_nc -v -s

# 检查依赖
pip install -r requirements.txt

# 清除缓存
rm -rf .pytest_cache __pycache__
```

### 问题 5：UI 集成失败

**症状**: Streamlit 应用启动时出错

**解决方案**:
```bash
# 检查 cmems_panel.py 是否正确导入
python -c "from arcticroute.ui.cmems_panel import render_env_source_selector"

# 运行 UI 集成脚本
python scripts/integrate_cmems_ui.py

# 启动 Streamlit 应用
streamlit run run_ui.py --logger.level=debug
```

---

## 📊 部署检查表

### 前置检查
- [ ] Python 版本 >= 3.8
- [ ] Git 已安装
- [ ] Copernicus Marine Toolbox 已安装
- [ ] 网络连接正常

### 部署步骤
- [ ] 克隆/更新项目
- [ ] 创建新分支
- [ ] 安装依赖
- [ ] 生成 Describe JSON
- [ ] 解析变量
- [ ] 运行测试
- [ ] 提交和推送
- [ ] 创建 PR

### 验证步骤
- [ ] Describe JSON 非空
- [ ] cmems_resolved.json 有效
- [ ] 所有测试通过
- [ ] Git 分支已推送
- [ ] PR 已创建

### 可选步骤
- [ ] 刷新 CMEMS 数据
- [ ] 同步到 Newenv
- [ ] 启动 UI 并测试

---

## 🔄 持续集成

### GitHub Actions 配置（可选）

创建 `.github/workflows/cmems-test.yml`:

```yaml
name: CMEMS Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run CMEMS tests
        run: |
          pytest tests/test_cmems_planner_integration.py -v
```

---

## 📞 支持和反馈

如遇到问题，请：
1. 检查本指南的故障排查部分
2. 查看项目的 Issues
3. 创建新的 Issue 并提供详细信息

---

## 📝 更新日志

### v1.0 (2024-12-15)
- ✅ 初始版本
- ✅ 核心功能实现
- ✅ 测试覆盖
- ✅ 文档完善

---

**最后更新**: 2024-12-15  
**维护者**: ArcticRoute Team

