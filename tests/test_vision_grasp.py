import numpy as np
import pytest
from aloha2_lab.sim import AlohaSim
from aloha2_lab.vision import locate_cup

@pytest.mark.parametrize('xy',[(-.19,.02),(-.14,.08),(-.18,.10)])
def test_rgbd_localizes_unseen_spawn_from_pixels(xy):
    sim=AlohaSim(cup_xy=xy)
    try:
        result=locate_cup(sim.rgbd(),sim.data.time)
        np.testing.assert_allclose(result['position_world'][:2],xy,atol=.006)
    finally:sim.close()

def test_missing_cup_stale_frame_and_invalid_depth_refuse():
    sim=AlohaSim()
    try:
        f=sim.rgbd()
        with pytest.raises(ValueError):locate_cup(f,f.timestamp+1.)
        with pytest.raises(ValueError):locate_cup(f,f.timestamp)
        f.depth[:]=np.nan
        with pytest.raises(ValueError):locate_cup(f,f.timestamp)
    finally:sim.close()

def test_vision_grasp_lifts_with_both_physical_fingers(tmp_path):
    from aloha2_lab.grasp import run
    result=run(tmp_path/'grasp',(-.18,.07))
    assert result['lift_success']
    assert len(result['finger_contacts'])==2
    assert not result['ground_truth_used_for_control']
