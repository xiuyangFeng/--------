"""Known-location regression: graph children end before the physical split."""
import numpy as np
from resolve_bifurcation_routes import route_arrays, paths_to_leaves
from wss_v5.bifurcation_locator import _scan


class KnownSplit:
    def __init__(self, split):
        self.split = split

    def contours(self, center, normal):
        assert np.allclose(normal / np.linalg.norm(normal), [0, 0, 1])
        squares = [(-1., .4), (1., .4)] if center[2] >= self.split else [(0., 2.)]
        return [{"polygon_xyz_mm": np.array([[x-r, -r, center[2]], [x+r, -r, center[2]], [x+r, r, center[2]], [x-r, r, center[2]]])} for x, r in squares]


def fixture():
    parts, ids, arcs = [], [], []
    for seg, x, start, end in [(1, -1, 0, 3), (2, 1, 0, 3), (3, 1, 3, 9), (4, 1, 3, 9), (5, -1, 3, 9), (6, -1, 3, 9)]:
        t = np.arange(start, end + .125, .25)
        parts.append(np.column_stack([np.full(len(t), x), np.zeros(len(t)), t]))
        ids.append(np.full(len(t), seg))
        arcs.append(t-start)
    return {"centerline_xyz_mm": np.concatenate(parts), "centerline_segment_id": np.concatenate(ids), "centerline_s_local_mm": np.concatenate(arcs)}


def test_descendant_extension_recovers_only_a_real_persistent_split():
    z = fixture()
    xyz, sid, ss = [z[k] for k in ["centerline_xyz_mm", "centerline_segment_id", "centerline_s_local_mm"]]
    old = _scan(xyz, sid, ss, 0, [1, 2], KnownSplit(5.))
    assert old["candidate"] is None  # physical split is censored by graph endpoints
    children = {1: [5, 6], 2: [3, 4], 3: [], 4: [], 5: [], 6: []}
    for left in paths_to_leaves(children, 1):
        for right in paths_to_leaves(children, 2):
            a, sa = route_arrays(z, left)
            b, sb = route_arrays(z, right)
            args = (np.concatenate([a,b]), np.r_[np.ones(len(a), int), np.full(len(b), 2, int)], np.r_[sa,sb], 0, [1,2])
            recovered = _scan(*args, KnownSplit(5.))
            assert recovered["candidate"]["bracket_mm"] == [4.75, 5.]
            assert sum(r["state"] == "two" for r in recovered["records"]) >= 5
            absent = _scan(*args, KnownSplit(20.))
            assert absent["candidate"] is None  # extension cannot invent a bifurcation


if __name__ == "__main__":
    test_descendant_extension_recovers_only_a_real_persistent_split()
    print("PASS: truncated graph regression and no-split negative control")
