import math

import numpy as np

from wss_v5.views.wall_flowref_v1 import _capfit_shares


def _tree():
    segments = {
        0: {"parent_id": -1, "starts_at_root": True, "ends_at_leaf": False, "outlet_name": ""},
        1: {"parent_id": 0, "starts_at_root": False, "ends_at_leaf": False, "outlet_name": ""},
        2: {"parent_id": 0, "starts_at_root": False, "ends_at_leaf": False, "outlet_name": ""},
        3: {"parent_id": 1, "starts_at_root": False, "ends_at_leaf": True, "outlet_name": "out-le"},
        4: {"parent_id": 1, "starts_at_root": False, "ends_at_leaf": True, "outlet_name": "out-li"},
        5: {"parent_id": 2, "starts_at_root": False, "ends_at_leaf": True, "outlet_name": "out-re"},
        6: {"parent_id": 2, "starts_at_root": False, "ends_at_leaf": True, "outlet_name": "out-ri"},
    }
    children = {0: [1, 2], 1: [3, 4], 2: [5, 6]}
    return segments, children


def test_capfit_shares_follow_the_rule_and_protocol():
    segments, children = _tree()
    rule = {"a": 1.07, "b": 0.16}
    caps = {3: 3.0, 4: 2.0, 5: 2.5, 6: 2.5}
    shares, notes = _capfit_shares(segments, children, caps, rule)
    assert shares[0] == 1.0 and shares[1] == 0.5 and shares[2] == 0.5
    ratio = math.exp(1.07 * math.log((3.0 / 2.0) ** 2) + 0.16)
    assert math.isclose(shares[3] / shares[4], ratio, rel_tol=1e-9)
    assert math.isclose(shares[3] + shares[4], 0.5, rel_tol=1e-12)
    # equal caps: only the intercept separates external from internal
    assert math.isclose(shares[5] / shares[6], math.exp(0.16), rel_tol=1e-9)
    assert notes["fallback_nodes"] == []


def test_capfit_shares_fall_back_when_a_cap_is_missing():
    segments, children = _tree()
    shares, notes = _capfit_shares(segments, children, {3: 3.0, 5: 2.5, 6: 2.5}, {"a": 1.0, "b": 0.0})
    assert notes["fallback_nodes"] == [1]
    # missing cap on the left pair -> equal split of the common-iliac share; right pair still uses the rule
    assert math.isclose(shares[3], 0.25, rel_tol=1e-12) and math.isclose(shares[4], 0.25, rel_tol=1e-12)
    assert math.isclose(shares[5] + shares[6], 0.5, rel_tol=1e-12)
    assert all(np.isfinite(list(shares.values())))
