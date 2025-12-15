#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test UI imports and basic functionality"""

import sys

try:
    # Test imports
    from arcticroute.ui.ais_density_panel import render_ais_density_panel, render_ais_density_summary
    print("[OK] AIS density panel imports successfully")
    
    from arcticroute.ui import planner_minimal
    print("[OK] planner_minimal imports successfully")
    
    from arcticroute.ui import home, eval_results
    print("[OK] Other UI modules import successfully")
    
    # Test basic functionality (no need to run Streamlit)
    from arcticroute.core.ais_density_select import (
        scan_ais_density_candidates,
        select_best_candidate,
        load_and_align_density,
    )
    print("[OK] AIS density select functions import successfully")
    
    print("\n[SUCCESS] All imports successful - UI should not crash")
    sys.exit(0)

except Exception as e:
    print(f"[FAILED] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
