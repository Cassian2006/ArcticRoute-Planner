"""
AIS JSON 结构探测器 (Phase AIS-A1)

目标：在不假设具体字段名的前提下，对 data_real/ais/raw/2024/*.json 进行结构探测，
输出一个人类可读的"profiling 报告"，用于指导后续管线设计。

用法：
    python -m scripts.inspect_ais_json
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List, Set, Tuple, Optional
import sys


def get_data_root() -> Path:
    """获取数据根目录。"""
    # 优先使用相对于脚本的路径
    script_dir = Path(__file__).parent.parent
    relative_path = script_dir / "data_real"
    
    if relative_path.exists():
        return relative_path
    
    # 否则尝试使用环境变量
    env_root = os.environ.get("ARCTICROUTE_DATA_ROOT")
    if env_root:
        return Path(env_root)
    
    # 最后回退到相对路径
    return relative_path


def estimate_file_size_mb(file_path: Path) -> float:
    """估计文件大小（MB）。"""
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def detect_structure_type(file_path: Path) -> str:
    """
    检测 JSON 文件的顶层结构类型。
    
    Returns:
        "list" / "dict" / "jsonl" / "unknown"
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 尝试读取第一行
            first_line = f.readline().strip()
            
            if not first_line:
                return "unknown"
            
            # 尝试解析第一行
            try:
                obj = json.loads(first_line)
                
                # 如果第一行是单个对象，可能是 JSONL 格式
                if isinstance(obj, dict):
                    # 检查是否所有行都是单个对象
                    f.seek(0)
                    line_count = 0
                    for line in f:
                        if line.strip():
                            try:
                                json.loads(line.strip())
                                line_count += 1
                            except json.JSONDecodeError:
                                break
                    
                    if line_count > 1:
                        return "jsonl"
                    else:
                        # 只有一行是对象，可能是整个文件是单个 dict
                        f.seek(0)
                        full_content = f.read()
                        try:
                            json.loads(full_content)
                            return "dict"
                        except:
                            return "unknown"
                
                elif isinstance(obj, list):
                    return "list"
            except json.JSONDecodeError:
                pass
            
            # 尝试解析整个文件
            f.seek(0)
            try:
                full_obj = json.load(f)
                if isinstance(full_obj, list):
                    return "list"
                elif isinstance(full_obj, dict):
                    return "dict"
            except json.JSONDecodeError:
                pass
    
    except Exception as e:
        print(f"[WARN] Error detecting structure type for {file_path}: {e}")
    
    return "unknown"


def sample_records(file_path: Path, max_samples: int = 10000, sample_rate: float = 0.01) -> List[Dict[str, Any]]:
    """
    从 JSON 文件中抽样记录。
    
    支持三种格式：
    - list: 整个文件是一个 JSON 数组
    - dict: 整个文件是一个 JSON 对象（可能包含数据字段）
    - jsonl: 每行一个 JSON 对象
    
    Args:
        file_path: 文件路径
        max_samples: 最多抽样多少条记录
        sample_rate: 抽样率（仅用于 JSONL 格式）
    
    Returns:
        抽样的记录列表
    """
    records = []
    structure_type = detect_structure_type(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            if structure_type == "list":
                # 整个文件是一个数组
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        # 抽样
                        step = max(1, len(data) // max_samples)
                        records = data[::step][:max_samples]
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSONDecodeError in {file_path}: {e}")
            
            elif structure_type == "dict":
                # 整个文件是一个对象，可能包含数据字段
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # 尝试找到包含列表的字段
                        for key, value in data.items():
                            if isinstance(value, list):
                                step = max(1, len(value) // max_samples)
                                records.extend(value[::step][:max_samples])
                                break
                        
                        # 如果没有找到列表字段，将整个对象作为一条记录
                        if not records:
                            records = [data]
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSONDecodeError in {file_path}: {e}")
            
            elif structure_type == "jsonl":
                # 每行一个 JSON 对象
                line_count = 0
                for line in f:
                    if not line.strip():
                        continue
                    
                    line_count += 1
                    
                    # 根据抽样率决定是否包含
                    if len(records) < max_samples and (line_count % max(1, int(1 / sample_rate))) == 0:
                        try:
                            obj = json.loads(line.strip())
                            records.append(obj)
                        except json.JSONDecodeError:
                            pass
    
    except Exception as e:
        print(f"[ERROR] Error sampling records from {file_path}: {e}")
    
    return records


def infer_field_type(value: Any) -> str:
    """推断字段值的类型。"""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        return "str"
    elif isinstance(value, list):
        return "list"
    elif isinstance(value, dict):
        return "dict"
    else:
        return "unknown"


def analyze_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析抽样的记录，提取字段统计信息。
    
    Returns:
        {
            "total_records": int,
            "common_fields": set,
            "field_types": {field_name: [type1, type2, ...]},
            "field_examples": {field_name: example_value},
            "field_nulls": {field_name: null_count},
        }
    """
    if not records:
        return {
            "total_records": 0,
            "common_fields": set(),
            "field_types": {},
            "field_examples": {},
            "field_nulls": {},
        }
    
    # 收集所有字段
    all_fields = set()
    field_types = defaultdict(set)
    field_examples = {}
    field_nulls = defaultdict(int)
    
    for record in records:
        if not isinstance(record, dict):
            continue
        
        for field_name, value in record.items():
            all_fields.add(field_name)
            
            if value is None:
                field_nulls[field_name] += 1
            else:
                field_type = infer_field_type(value)
                field_types[field_name].add(field_type)
                
                # 保存示例值（只保存简单类型）
                if field_name not in field_examples and field_type in ("str", "int", "float", "bool"):
                    field_examples[field_name] = value
    
    return {
        "total_records": len(records),
        "common_fields": all_fields,
        "field_types": {k: sorted(list(v)) for k, v in field_types.items()},
        "field_examples": field_examples,
        "field_nulls": dict(field_nulls),
    }


def guess_semantic_fields(analysis: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    根据字段名和类型，猜测语义字段。
    
    Returns:
        {
            "timestamp": field_name or None,
            "lat": field_name or None,
            "lon": field_name or None,
            "mmsi": field_name or None,
            "sog": field_name or None,
            "cog": field_name or None,
            "ship_type": field_name or None,
        }
    """
    fields = analysis.get("common_fields", set())
    field_types = analysis.get("field_types", {})
    
    guesses = {
        "timestamp": None,
        "lat": None,
        "lon": None,
        "mmsi": None,
        "sog": None,
        "cog": None,
        "ship_type": None,
    }
    
    # 时间戳字段
    time_keywords = ["timestamp", "ts", "time", "basedatetime", "datetime", "utc", "date"]
    for field in fields:
        field_lower = field.lower()
        if any(kw in field_lower for kw in time_keywords):
            guesses["timestamp"] = field
            break
    
    # 纬度
    lat_keywords = ["lat", "latitude", "y"]
    for field in fields:
        field_lower = field.lower()
        if any(kw in field_lower for kw in lat_keywords):
            guesses["lat"] = field
            break
    
    # 经度
    lon_keywords = ["lon", "longitude", "x"]
    for field in fields:
        field_lower = field.lower()
        if any(kw in field_lower for kw in lon_keywords):
            guesses["lon"] = field
            break
    
    # MMSI
    mmsi_keywords = ["mmsi", "imo", "id"]
    for field in fields:
        field_lower = field.lower()
        if any(kw in field_lower for kw in mmsi_keywords):
            guesses["mmsi"] = field
            break
    
    # 速度 (SOG)
    sog_keywords = ["sog", "speed", "velocity", "knots"]
    for field in fields:
        field_lower = field.lower()
        if any(kw in field_lower for kw in sog_keywords):
            guesses["sog"] = field
            break
    
    # 航向 (COG)
    cog_keywords = ["cog", "heading", "course", "direction"]
    for field in fields:
        field_lower = field.lower()
        if any(kw in field_lower for kw in cog_keywords):
            guesses["cog"] = field
            break
    
    # 船型
    ship_type_keywords = ["ship_type", "vessel_type", "type", "class", "category"]
    for field in fields:
        field_lower = field.lower()
        if any(kw in field_lower for kw in ship_type_keywords):
            guesses["ship_type"] = field
            break
    
    return guesses


def estimate_record_count(file_path: Path) -> int:
    """
    估计文件中的记录条数。
    
    对于大文件，使用采样估计。
    """
    structure_type = detect_structure_type(file_path)
    
    try:
        if structure_type == "list":
            # 读取整个文件并计数
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return len(data)
        
        elif structure_type == "jsonl":
            # 逐行计数
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                count = 0
                for line in f:
                    if line.strip():
                        count += 1
                return count
        
        elif structure_type == "dict":
            # 尝试找到包含列表的字段
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            return len(value)
            return 1
    
    except Exception as e:
        print(f"[WARN] Error estimating record count for {file_path}: {e}")
    
    return 0


def count_decode_errors(file_path: Path) -> int:
    """统计 JSON 解码错误的行数。"""
    error_count = 0
    structure_type = detect_structure_type(file_path)
    
    if structure_type == "jsonl":
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        json.loads(line.strip())
                    except json.JSONDecodeError:
                        error_count += 1
        except Exception as e:
            print(f"[WARN] Error counting decode errors for {file_path}: {e}")
    
    return error_count


def main():
    """主函数：执行 AIS JSON 结构探测。"""
    data_root = get_data_root()
    ais_dir = data_root / "ais" / "2024"
    
    print(f"\n{'='*80}")
    print(f"AIS JSON 结构探测器 (Phase AIS-A1)")
    print(f"{'='*80}")
    print(f"数据根目录: {data_root}")
    print(f"AIS 数据目录: {ais_dir}")
    print()
    
    # 查找所有 JSON 文件
    if not ais_dir.exists():
        print(f"[ERROR] AIS 目录不存在: {ais_dir}")
        return
    
    json_files = sorted(ais_dir.glob("*.json"))
    
    if not json_files:
        print(f"[ERROR] 未找到 JSON 文件在 {ais_dir}")
        return
    
    print(f"找到 {len(json_files)} 个 JSON 文件：")
    for f in json_files:
        print(f"  - {f.name}")
    print()
    
    # ========================================================================
    # 第一步：扫描每个文件的基本信息
    # ========================================================================
    print(f"{'='*80}")
    print("第一步：基本信息扫描")
    print(f"{'='*80}\n")
    
    file_info = {}
    
    for json_file in json_files:
        print(f"扫描: {json_file.name}")
        
        size_mb = estimate_file_size_mb(json_file)
        structure_type = detect_structure_type(json_file)
        record_count = estimate_record_count(json_file)
        error_count = count_decode_errors(json_file)
        
        print(f"  大小: {size_mb:.2f} MB")
        print(f"  结构类型: {structure_type}")
        print(f"  估计记录数: {record_count}")
        if error_count > 0:
            print(f"  解码错误: {error_count}")
        print()
        
        file_info[json_file.name] = {
            "path": json_file,
            "size_mb": size_mb,
            "structure_type": structure_type,
            "record_count": record_count,
            "error_count": error_count,
        }
    
    # ========================================================================
    # 第二步：抽样分析每个文件的字段结构
    # ========================================================================
    print(f"{'='*80}")
    print("第二步：字段结构分析（抽样 100-10000 条记录）")
    print(f"{'='*80}\n")
    
    schema_suggestions = {}
    
    for file_name, info in file_info.items():
        print(f"分析: {file_name}")
        
        # 抽样记录
        records = sample_records(info["path"], max_samples=10000, sample_rate=0.01)
        
        if not records:
            print(f"  [WARN] 无法抽样任何记录")
            print()
            continue
        
        print(f"  抽样记录数: {len(records)}")
        
        # 分析记录
        analysis = analyze_records(records)
        
        print(f"  共同字段数: {len(analysis['common_fields'])}")
        print(f"  字段列表: {sorted(analysis['common_fields'])}")
        print()
        
        # 打印字段类型
        print(f"  字段类型统计:")
        for field_name in sorted(analysis['common_fields']):
            types = analysis['field_types'].get(field_name, [])
            null_count = analysis['field_nulls'].get(field_name, 0)
            example = analysis['field_examples'].get(field_name, "N/A")
            
            print(f"    - {field_name}: {types} (null: {null_count}, example: {example})")
        
        print()
        
        # 猜测语义字段
        guesses = guess_semantic_fields(analysis)
        
        print(f"  语义字段猜测:")
        for semantic_name, field_name in guesses.items():
            if field_name:
                print(f"    - {semantic_name}: {field_name}")
            else:
                print(f"    - {semantic_name}: [未找到]")
        
        print()
        
        # 生成 schema 建议
        schema_suggestion = {}
        for semantic_name, field_name in guesses.items():
            if field_name:
                schema_suggestion[semantic_name] = field_name
        
        schema_suggestions[file_name] = {
            "analysis": analysis,
            "guesses": guesses,
            "schema": schema_suggestion,
        }
        
        # 打印 JSON 风格的 schema 建议
        print(f"  [SCHEMA SUGGESTION] file={file_name}")
        print(f"  {json.dumps(schema_suggestion, indent=2, ensure_ascii=False)}")
        print()
    
    # ========================================================================
    # 第三步：生成统一的全局 schema 建议
    # ========================================================================
    print(f"{'='*80}")
    print("第三步：全局 Schema 建议")
    print(f"{'='*80}\n")
    
    # 收集所有文件中的字段
    all_fields_across_files = defaultdict(list)
    
    for file_name, suggestion in schema_suggestions.items():
        for semantic_name, field_name in suggestion['guesses'].items():
            if field_name:
                all_fields_across_files[semantic_name].append((file_name, field_name))
    
    print("跨文件字段映射统计:")
    for semantic_name in sorted(all_fields_across_files.keys()):
        mappings = all_fields_across_files[semantic_name]
        print(f"  {semantic_name}:")
        for file_name, field_name in mappings:
            print(f"    - {file_name}: {field_name}")
    
    print()
    
    # 推荐统一的字段映射
    unified_schema = {}
    for semantic_name in sorted(all_fields_across_files.keys()):
        mappings = all_fields_across_files[semantic_name]
        # 选择最常见的字段名
        field_names = [fn for _, fn in mappings]
        most_common = Counter(field_names).most_common(1)
        if most_common:
            unified_schema[semantic_name] = most_common[0][0]
    
    print("推荐的统一字段映射:")
    print(json.dumps(unified_schema, indent=2, ensure_ascii=False))
    print()
    
    # ========================================================================
    # 第四步：生成 Markdown 报告
    # ========================================================================
    print(f"{'='*80}")
    print("第四步：生成 Markdown 报告")
    print(f"{'='*80}\n")
    
    report_path = data_root.parent / "reports" / "ais_schema_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_content = generate_markdown_report(
        json_files,
        file_info,
        schema_suggestions,
        unified_schema,
    )
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ 报告已生成: {report_path}")
    print()
    
    # ========================================================================
    # 第五步：数据质量检查
    # ========================================================================
    print(f"{'='*80}")
    print("第五步：数据质量检查")
    print(f"{'='*80}\n")
    
    for file_name, info in file_info.items():
        print(f"质量检查: {file_name}")
        
        if info['error_count'] > 0:
            error_rate = info['error_count'] / max(1, info['record_count'])
            print(f"  ⚠️ 解码错误率: {error_rate*100:.2f}%")
        else:
            print(f"  ✓ 无解码错误")
        
        # 检查字段完整性
        suggestion = schema_suggestions.get(file_name)
        if suggestion:
            guesses = suggestion['guesses']
            missing_fields = [k for k, v in guesses.items() if v is None]
            if missing_fields:
                print(f"  ⚠️ 缺失字段: {missing_fields}")
            else:
                print(f"  ✓ 所有关键字段都已找到")
        
        print()
    
    # ========================================================================
    # 总结
    # ========================================================================
    print(f"{'='*80}")
    print("探测完成！")
    print(f"{'='*80}\n")
    
    print("关键发现:")
    print(f"  - 共 {len(json_files)} 个 AIS JSON 文件")
    print(f"  - 总数据量: {sum(info['size_mb'] for info in file_info.values()):.2f} MB")
    print(f"  - 总记录数: {sum(info['record_count'] for info in file_info.values())}")
    print(f"  - 推荐的统一 schema: {len(unified_schema)} 个关键字段")
    print()
    
    print("后续步骤:")
    print("  1. 查看 reports/ais_schema_report.md 了解详细的字段分析")
    print("  2. 根据 unified_schema 设计数据管线")
    print("  3. 实现字段映射和数据清洗逻辑")
    print()


def generate_markdown_report(
    json_files: List[Path],
    file_info: Dict[str, Dict],
    schema_suggestions: Dict[str, Dict],
    unified_schema: Dict[str, str],
) -> str:
    """生成 Markdown 格式的报告。"""
    
    lines = [
        "# AIS JSON 结构探测报告 (Phase AIS-A1)",
        "",
        f"生成时间: {__import__('datetime').datetime.now().isoformat()}",
        "",
        "## 概述",
        "",
        f"本报告对 `data_real/ais/2024/` 目录下的 {len(json_files)} 个 AIS JSON 文件进行了结构探测。",
        "目的是在不假设具体字段名的前提下，理解数据的字段结构，为后续管线设计提供指导。",
        "",
        "## 文件清单",
        "",
    ]
    
    # 文件清单表格
    lines.append("| 文件名 | 大小 (MB) | 结构类型 | 记录数 | 错误数 |")
    lines.append("|--------|----------|---------|--------|--------|")
    
    for file_name, info in file_info.items():
        lines.append(
            f"| {file_name} | {info['size_mb']:.2f} | {info['structure_type']} | "
            f"{info['record_count']} | {info['error_count']} |"
        )
    
    lines.extend(["", "## 字段统计", ""])
    
    # 每个文件的字段分析
    for file_name, suggestion in schema_suggestions.items():
        lines.append(f"### {file_name}")
        lines.append("")
        
        analysis = suggestion['analysis']
        
        lines.append(f"**抽样记录数**: {analysis['total_records']}")
        lines.append("")
        
        lines.append("**字段列表**:")
        lines.append("")
        
        for field_name in sorted(analysis['common_fields']):
            types = analysis['field_types'].get(field_name, [])
            null_count = analysis['field_nulls'].get(field_name, 0)
            example = analysis['field_examples'].get(field_name, "N/A")
            
            lines.append(f"- **{field_name}**")
            lines.append(f"  - 类型: {', '.join(types)}")
            lines.append(f"  - 空值数: {null_count}")
            lines.append(f"  - 示例: `{example}`")
        
        lines.append("")
    
    # 推荐的统一 schema
    lines.extend([
        "## 推荐的统一 Schema",
        "",
        "基于对所有文件的分析，推荐以下统一的字段映射：",
        "",
    ])
    
    lines.append("```json")
    lines.append(json.dumps(unified_schema, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    
    # 统一列定义
    lines.extend([
        "## 统一列定义",
        "",
        "基于推荐的 schema，建议的统一数据结构如下：",
        "",
        "| 字段名 | 推荐类型 | 说明 |",
        "|--------|---------|------|",
    ])
    
    type_mapping = {
        "timestamp": ("datetime64[ns]", "AIS 记录时间戳"),
        "lat": ("float64", "纬度（度）"),
        "lon": ("float64", "经度（度）"),
        "mmsi": ("int64", "船舶 MMSI 标识"),
        "sog": ("float32", "对地速度（节）"),
        "cog": ("float32", "对地航向（度）"),
        "ship_type": ("category", "船舶类型（如可用）"),
    }
    
    for semantic_name, field_name in unified_schema.items():
        rec_type, desc = type_mapping.get(semantic_name, ("object", ""))
        lines.append(f"| {semantic_name} | {rec_type} | {desc} |")
    
    lines.extend(["", "## 数据质量检查", ""])
    
    # 质量检查结果
    for file_name, info in file_info.items():
        lines.append(f"### {file_name}")
        lines.append("")
        
        if info['error_count'] > 0:
            error_rate = info['error_count'] / max(1, info['record_count'])
            lines.append(f"⚠️ **解码错误**: {info['error_count']} 行 ({error_rate*100:.2f}%)")
        else:
            lines.append("✓ **无解码错误**")
        
        lines.append("")
    
    # 后续步骤
    lines.extend([
        "## 后续步骤",
        "",
        "1. **字段映射**: 根据推荐的 schema，在数据管线中实现字段映射逻辑",
        "2. **数据清洗**: 处理缺失值、异常值、类型转换等",
        "3. **聚合策略**: 决定是否按航次、时间窗口等维度聚合数据",
        "4. **密度/轨迹分析**: 基于清洗后的数据进行后续分析",
        "",
        "## 附录：完整字段分析",
        "",
    ])
    
    # 详细的字段分析
    for file_name, suggestion in schema_suggestions.items():
        lines.append(f"### {file_name} - 详细字段分析")
        lines.append("")
        
        analysis = suggestion['analysis']
        guesses = suggestion['guesses']
        
        lines.append("**语义字段猜测**:")
        lines.append("")
        
        for semantic_name, field_name in guesses.items():
            if field_name:
                lines.append(f"- `{semantic_name}` → `{field_name}`")
            else:
                lines.append(f"- `{semantic_name}` → [未找到]")
        
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()
    
    # ========================================================================
    # 小试运行：调用标准化加载模块
    # ========================================================================
    print(f"\n{'='*80}")
    print("小试运行：AIS 标准化加载模块")
    print(f"{'='*80}\n")
    
    try:
        import pathlib as _pl
        from arcticroute.data.ais_io import AISLoadConfig, load_ais_json_to_df
        
        # 创建配置对象
        cfg = AISLoadConfig(root=_pl.Path(__file__).resolve().parents[1])
        
        print(f"配置参数:")
        print(f"  root: {cfg.root}")
        print(f"  year: {cfg.year}")
        print(f"  lat_min: {cfg.lat_min}, lat_max: {cfg.lat_max}")
        print(f"  lon_min: {cfg.lon_min}, lon_max: {cfg.lon_max}")
        print()
        
        # 加载 AIS 数据
        print("正在加载 AIS 数据...")
        df = load_ais_json_to_df(cfg)
        
        print(f"✓ 加载完成！")
        print(f"  总行数: {len(df)}")
        print(f"  列: {list(df.columns)}")
        print(f"  数据类型:")
        for col in df.columns:
            print(f"    - {col}: {df[col].dtype}")
        print()
        
        # 显示基本统计
        print("基本统计:")
        print(f"  时间范围: {df['ts'].min()} 到 {df['ts'].max()}")
        print(f"  纬度范围: {df['lat'].min():.2f}° 到 {df['lat'].max():.2f}°")
        print(f"  经度范围: {df['lon'].min():.2f}° 到 {df['lon'].max():.2f}°")
        print(f"  MMSI 数量: {df['mmsi'].nunique()}")
        print(f"  SOG 统计: min={df['sog_knots'].min():.2f}, max={df['sog_knots'].max():.2f}, "
              f"mean={df['sog_knots'].mean():.2f}")
        print(f"  COG 统计: min={df['cog_deg'].min():.2f}, max={df['cog_deg'].max():.2f}, "
              f"mean={df['cog_deg'].mean():.2f}")
        print()
        
        # 显示前几行
        print("前 5 行数据:")
        print(df.head())
        print()
        
        # 显示后几行
        print("后 5 行数据:")
        print(df.tail())
        print()
        
        # 缺失值检查
        print("缺失值检查:")
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                print(f"  {col}: {null_count} ({null_count/len(df)*100:.2f}%)")
        print()
        
        print("✓ 小试运行成功！")
        
    except Exception as e:
        import traceback
        print(f"✗ 小试运行失败: {e}")
        traceback.print_exc()

