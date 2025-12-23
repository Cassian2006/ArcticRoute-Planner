from __future__ import annotations

from arcticroute.core.eco.vessel_profiles import (
    get_default_profiles,
    get_profile_catalog,
)


def test_profile_catalog_larger_than_defaults():
    defaults = get_default_profiles()
    catalog = get_profile_catalog()

    assert isinstance(defaults, dict)
    assert isinstance(catalog, dict)
    assert len(defaults) >= 3
    # 全量目录应明显多于默认配置
    assert len(catalog) > len(defaults)


