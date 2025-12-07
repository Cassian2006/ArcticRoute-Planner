# Make ArcticRoute.apps a proper Python package
# Expose commonly used symbols for convenience
try:
    from .registry import UIRegistry  # noqa: F401
except Exception:
    # registry may not be imported in some environments; keep package importable
    pass
