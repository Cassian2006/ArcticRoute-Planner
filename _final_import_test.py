#!/usr/bin/env python
"""Final import verification"""
import sys

try:
    import arcticroute
    print(f"OK arcticroute imported from: {arcticroute.__file__}")
    
    import arcticroute.core.grid as g
    print(f"OK arcticroute.core.grid imported: {g}")
    print(f"OK Grid2D class available: {g.Grid2D}")
    
    from arcticroute.core.ais_ingest import inspect_ais_csv, load_ais_from_raw_dir
    print(f"OK inspect_ais_csv function available")
    print(f"OK load_ais_from_raw_dir function available")
    
    print("\nSUCCESS: All imports successful!")
    sys.exit(0)
except Exception as e:
    print(f"FAILED: Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
