import numpy as np
import pytest

from wss_deploy.streamlines import integrate_streamlines


def test_constant_flow_follows_world_vector_and_stops_at_lumen():
    xyz = np.stack(np.meshgrid(np.arange(-5, 5.1, .5), [-.5, 0, .5], [-.5, 0, .5]), -1).reshape(-1, 3)
    velocity = np.tile([.8, 0, 0], (len(xyz), 1))
    inside = lambda q: (np.abs(q[:, 0]) < 5) & (np.linalg.norm(q[:, 1:], axis=1) < 1)
    lines, info = integrate_streamlines(xyz, velocity, [[0, 0, 0], [0, 2, 0]], inside, step_mm=.25, max_steps=80)
    assert len(lines) == 1
    line = lines[0]
    assert inside(line["points"]).all()
    assert np.allclose(line["points"][:, 1:], 0)
    assert np.allclose(line["speed_m_s"], .8)
    assert line["points"][0, 0] < -4 and line["points"][-1, 0] > 4
    assert info["time_dependent"] is False


def test_streamlines_do_not_bridge_uncovered_gap():
    x = np.r_[np.arange(-5, -.9, .2), np.arange(1, 5, .2)]
    xyz = np.column_stack([x, np.zeros((len(x), 2))])
    lines, _ = integrate_streamlines(xyz, np.tile([1, 0, 0], (len(x), 1)), [[-3, 0, 0]],
                                    lambda q: np.abs(q[:, 0]) < 5,
                                    step_mm=.1, max_steps=100, max_distance_mm=.3)
    assert len(lines) == 1
    assert lines[0]["points"][:, 0].max() < -.6


def test_zero_flow_and_invalid_fields():
    xyz = np.array([[0, 0, 0], [1, 0, 0]])
    lines, _ = integrate_streamlines(xyz, xyz * 0, [[.5, 0, 0]], lambda q: np.ones(len(q), bool))
    assert lines == []
    with pytest.raises(ValueError):
        integrate_streamlines(xyz, np.ones((1, 3)), [[0, 0, 0]], lambda q: True)
