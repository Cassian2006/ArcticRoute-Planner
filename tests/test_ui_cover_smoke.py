from pathlib import Path

def test_cover_asset_exists():
    p = Path("arcticroute/ui/assets/arctic_ui_cover.html")
    assert p.exists(), "arctic_ui_cover.html missing under arcticroute/ui/assets"

def test_cover_contains_start_button():
    p = Path("arcticroute/ui/assets/arctic_ui_cover.html")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    assert "btnStart" in txt, "cover html should contain btnStart for navigation hook"

