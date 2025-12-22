from arcticroute.core.eco.vessel_profiles import get_profile_catalog


def test_profile_catalog_contains_legacy_keys():
    catalog = get_profile_catalog()
    assert len(catalog) >= 10
    assert "handy" in catalog
    assert "panamax" in catalog
    assert "ice_class" in catalog
