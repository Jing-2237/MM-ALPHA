import numpy as np
import pytest
from aloha2_lab.sim import AlohaSim, JOINTS


def test_model_actuator_mapping_and_simulation_step():
    sim = AlohaSim()
    try:
        assert sim.model.nu == 14
        assert JOINTS[6] == "left/gripper" and JOINTS[13] == "right/gripper"
        before = sim.data.time
        q,v = sim.step(sim.data.ctrl.copy())
        assert q.shape == v.shape == (14,)
        assert np.isfinite(q).all() and np.isfinite(v).all()
        assert sim.data.time == pytest.approx(before+sim.dt)
        with pytest.raises(ValueError,match="limits"):
            sim.step(np.full(14,999.))
        with pytest.raises(ValueError,match="finite"):
            sim.step(np.full(14,np.nan))
    finally:
        sim.close()
