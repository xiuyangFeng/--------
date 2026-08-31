from pipeline_wss_min import config as C


def test_yang_yu_qing_uses_audited_fixed_mm_factor():
    unit, reason = C.unit_for_case("ILO/YANG_YU_QING-1", "before")
    assert unit.mode == "fixed"
    assert unit.fixed_factor == 1000.0
    assert reason is not None
    assert "authoritative" in reason


def test_unlisted_case_keeps_base_unit_config():
    base = C.UnitConfig(mode="auto_centerline", fixed_factor=123.0)
    unit, reason = C.unit_for_case("AG/fast", "CHEN_SHI_MING", base)
    assert unit == base
    assert reason is None
