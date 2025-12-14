"""
乱码检测测试 - 防止 UTF-8 编码问题复发

这个测试扫描所有关键文件，确保不存在 mojibake（乱码）特征字符。
"""

import pytest
from pathlib import Path
from arcticroute.core.scenarios import load_all_scenarios


# 乱码特征字符（常见的 UTF-8 误解码特征）
MOJIBAKE_CHARS = {
    'æ',  # 常见乱码
    'ä',  # 常见乱码
    'ç',  # 常见乱码
    'ö',  # 常见乱码
    'ü',  # 常见乱码
}


def test_scenarios_title_no_mojibake():
    """检查所有 scenario 的 title 中没有乱码"""
    scenarios = load_all_scenarios()
    
    for scenario_id, scenario in scenarios.items():
        # 检查 title
        for char in MOJIBAKE_CHARS:
            assert char not in scenario.title, (
                f"Scenario '{scenario_id}' title contains mojibake char '{char}': {scenario.title}"
            )
        
        # 检查 description
        for char in MOJIBAKE_CHARS:
            assert char not in scenario.description, (
                f"Scenario '{scenario_id}' description contains mojibake char '{char}': {scenario.description}"
            )


def test_planner_ui_labels_no_mojibake():
    """检查 planner_minimal.py 中的标签没有乱码"""
    planner_path = Path(__file__).parent.parent / "arcticroute" / "ui" / "planner_minimal.py"
    
    if not planner_path.exists():
        pytest.skip(f"planner_minimal.py not found at {planner_path}")
    
    content = planner_path.read_text(encoding='utf-8')
    
    # 检查 ROUTE_LABELS_ZH 定义
    for char in MOJIBAKE_CHARS:
        assert char not in content, (
            f"planner_minimal.py contains mojibake char '{char}'"
        )


def test_scenarios_yaml_encoding():
    """检查 scenarios.yaml 是否能正确读取为 UTF-8"""
    from arcticroute.core.scenarios import DEFAULT_SCENARIOS_PATH
    
    # 尝试读取 YAML 文件
    yaml_content = DEFAULT_SCENARIOS_PATH.read_text(encoding='utf-8')
    
    # 检查是否有乱码特征
    for char in MOJIBAKE_CHARS:
        assert char not in yaml_content, (
            f"scenarios.yaml contains mojibake char '{char}'"
        )


def test_scenario_titles_are_readable():
    """检查所有 scenario 的 title 是否可读（不包含乱码）"""
    scenarios = load_all_scenarios()
    
    # 检查每个 scenario 的 title 是否有效
    assert len(scenarios) > 0, "No scenarios loaded"
    
    for scenario_id, scenario in scenarios.items():
        # 标题应该不为空
        assert scenario.title, f"Scenario '{scenario_id}' has empty title"
        
        # 标题应该是可打印的字符
        assert all(c.isprintable() or c.isspace() for c in scenario.title), (
            f"Scenario '{scenario_id}' title contains non-printable characters: {scenario.title}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

