"""A-09: Streamlit UI Reconnaissance (只读)

静态分析 apps/app_min.py，提取 UI 结构、控件、缓存、后端调用与导出功能。
输出 reports/recon/ui_map.json。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set


class UiVisitor(ast.NodeVisitor):
    """AST 访问器，用于提取 Streamlit UI 组件与交互。"""

    def __init__(self):
        self.elements: List[Dict[str, Any]] = []
        self.backend_calls: Set[str] = set()
        self.cache_decorators: List[str] = []

    def visit_Call(self, node: ast.Call):
        """访问函数调用节点。"""
        # 检查 st.foo() 形式的调用
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "st":
                self._analyze_st_call(node)

        # 检查 st_folium() 等直接导入的函数
        if isinstance(node.func, ast.Name) and node.func.id == "st_folium":
            self.elements.append({"type": "st_folium", "line": node.lineno})

        # 检查后端调用（示例）
        if isinstance(node.func, ast.Name) and node.func.id in ("astar_on_cost", "submit_task"):
            self.backend_calls.add(node.func.id)

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """访问函数定义，检查缓存装饰器。"""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute) and isinstance(decorator.value, ast.Name):
                if decorator.value.id == "st" and decorator.attr.startswith("cache"):
                    self.cache_decorators.append(f"@{decorator.value.id}.{decorator.attr}")
            elif isinstance(decorator, ast.Name) and decorator.id.startswith("cache"):
                self.cache_decorators.append(f"@{decorator.id}")
        self.generic_visit(node)

    def _analyze_st_call(self, node: ast.Call):
        element_type = node.func.attr
        # 提取第一个字符串参数作为标签/标题
        label = ""
        if node.args and isinstance(node.args[0], ast.Constant):
            label = node.args[0].value
        elif node.keywords:
            for kw in node.keywords:
                if kw.arg == "label" and isinstance(kw.value, ast.Constant):
                    label = kw.value.value
                    break

        details: Dict[str, Any] = {"type": element_type, "label": label, "line": node.lineno}

        # 提取关键参数
        if element_type == "button":
            for kw in node.keywords:
                if kw.arg == "on_click":
                    # 记录回调函数名
                    if isinstance(kw.value, ast.Name):
                        details["on_click"] = kw.value.id

        if element_type == "download_button":
            details["export"] = True

        self.elements.append(details)


def analyze_ui(file_path: Path) -> Dict[str, Any]:
    """主分析函数，结合 AST 和正则表达式。"""
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}

    text = file_path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    visitor = UiVisitor()
    visitor.visit(tree)

    # 正则表达式补充分析
    tabs = re.findall(r"st\.tabs\(\[(.*?)\]\)", text, re.DOTALL)
    tab_names = [name.strip().strip("'") for name in tabs[0].split(",")] if tabs else []

    # 查找 subprocess 调用
    subprocess_calls = re.findall(r"subprocess\.Popen\((.*?)\)", text, re.DOTALL)

    # 汇总结果
    ui_map = {
        "file": str(file_path),
        "structure": {
            "tabs": tab_names,
            "sidebars": len(re.findall(r"with st\.sidebar:", text)),
            "expanders": len(re.findall(r"with st\.expander\(", text)),
            "columns": len(re.findall(r"st\.columns\(", text)),
        },
        "components": visitor.elements,
        "backend_interactions": {
            "direct_calls": sorted(list(visitor.backend_calls)),
            "subprocess_calls": len(subprocess_calls),
            "cache_decorators": sorted(list(set(visitor.cache_decorators))),
        },
        "exports": [el for el in visitor.elements if el.get("export")],
    }
    return ui_map


def main():
    parser = argparse.ArgumentParser(description="Analyze Streamlit UI from a Python file.")
    repo_root = Path(__file__).resolve().parents[1]
    default_app_path = repo_root / "ArcticRoute" / "apps" / "app_min.py"
    default_out_path = repo_root / "reports" / "recon" / "ui_map.json"

    parser.add_argument("--app", default=str(default_app_path), help="Path to the Streamlit app file.")
    parser.add_argument("--out", default=str(default_out_path), help="Output JSON file path.")
    args = parser.parse_args()

    app_path = Path(args.app)
    out_path = Path(args.out)

    ui_map = analyze_ui(app_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ui_map, f, indent=2, ensure_ascii=False)

    print(f"UI map saved to: {out_path}")


if __name__ == "__main__":
    main()

