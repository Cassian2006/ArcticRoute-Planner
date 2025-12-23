from arcticroute.ui.ui_doctor import run_ui_doctor

def test_ui_doctor_runs():
    checks = run_ui_doctor()
    assert isinstance(checks, list)
    assert len(checks) >= 3
