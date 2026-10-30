import numpy as np
import mujoco
from aloha2_lab.sim import AlohaSim
from aloha2_lab.visual_il import plan,features,load_policy

def test_split_positions_are_spatially_disjoint_and_predeclared(tmp_path):
    m=plan(tmp_path/'benchmark')
    fitting=np.array(m['train']+m['validation'])
    assert len(m['test'])==10
    for p in m['test']:
        assert np.linalg.norm(fitting-p,axis=1).min()>.007

def test_coffee_mug_has_physical_handle():
    s=AlohaSim(cup_xy=(-.18,.07))
    try:
        i=mujoco.mj_name2id(s.model,mujoco.mjtObj.mjOBJ_GEOM,'coffee_handle_0')
        assert i>=0 and s.model.geom_contype[i]!=0
    finally:s.close()

def test_learned_policy_uses_rgb_without_detector_or_ik(monkeypatch):
    from pathlib import Path
    import pytest
    import h5py
    from aloha2_lab import visual_il
    root=visual_il.ROOT/'data/coffee-visual-v1'
    if not (root/'model/encoder.pt').exists():pytest.skip('Requires trained benchmark artifact')
    def forbidden(*args,**kwargs):raise AssertionError('Learned policy called teacher/oracle')
    monkeypatch.setattr(visual_il,'ik',forbidden)
    monkeypatch.setattr(visual_il,'locate_cup',forbidden)
    policy=load_policy(root/'model')
    with h5py.File(sorted((root/'train').glob('*.hdf5'))[0]) as f:rgb=f['initial_rgb'][:]
    actions,_=policy(rgb);blank,_=policy(np.zeros_like(rgb))
    assert actions.shape==(820,14) and np.isfinite(actions).all()
    assert np.max(np.abs(actions-blank))>.01
