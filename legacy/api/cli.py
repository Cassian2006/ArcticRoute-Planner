# 统一 CLI 入口代理：安全加载 ArcticRoute.api.cli 中的入口（cli 或 main）
# 保留业务语义，仅作为导入代理，允许 python -m api.cli 直接运行

# 优先尝试导入 cli（click.Group 或可调用函数），否则回退到 main(argv)
try:
    from ArcticRoute.api.cli import cli as _entry  # type: ignore[attr-defined]
except Exception:
    try:
        from ArcticRoute.api.cli import main as _entry  # type: ignore[attr-defined]
    except Exception as e:  # 防御式：任何导入失败都降级为提示并退出
        import sys
        import traceback

        def _entry():  # type: ignore
            print("CLI proxy failed to import ArcticRoute.api.cli:", e)
            traceback.print_exc()
            sys.exit(2)

# 兼容外部直接调用 cli 或 main
cli = _entry  # type: ignore
main = _entry  # type: ignore

if __name__ == "__main__":
    # 允许 python -m api.cli 直接执行
    if callable(_entry):
        _entry()
