import numpy as np
import pytest
from aloha2_lab.rgbd_input import metric_depth,model_input,make_rgbd_encoder


def test_depth_units_invalid_values_and_validity_channel():
    d,v=metric_depth(np.array([[1000,0,65535]],np.uint16),'16UC1')
    np.testing.assert_allclose(d,[[1,0,0]])
    np.testing.assert_array_equal(v,[[True,False,False]])
    rgb=np.zeros((2,3,3),np.uint8)
    depth=np.array([[.5,np.nan,np.inf],[0,1,3]],np.float32)
    x=model_input(rgb,depth)
    assert x.shape==(5,2,3) and np.isfinite(x).all()
    np.testing.assert_array_equal(x[4],[[1,0,0],[0,1,0]])


def test_depth_is_in_model_forward_and_backward():
    import torch
    torch.manual_seed(1)
    model=make_rgbd_encoder()
    x=torch.rand(2,5,8,8);x[:,3]=.5;x[:,4]=1;x.requires_grad_()
    a,_=model(x)
    other=x.detach().clone();other[:,3]=.9
    b,_=model(other)
    assert torch.all(b[:,2]>a[:,2]+.3)
    a.sum().backward()
    assert x.grad[:,3].abs().sum()>0
    empty=x.detach().clone();empty[:,4]=0
    z,_=model(empty)
    assert torch.isfinite(z).all() and torch.all(z==0)


def test_confirmed_paper_cup_geometry_and_depth_localization():
    import json
    from pathlib import Path
    from aloha2_lab.sim import AlohaSim,ROOT
    from aloha2_lab.vision import locate_paper_cup
    p=json.loads((ROOT/'config/paper_cup.json').read_text())
    assert p['geometry_confirmed'] and p['bottom_radius_m']==.025 and p['top_radius_m']==.05 and p['height_m']==.12
    s=AlohaSim(cup_xy=(-.03,.06),cup_profile=p)
    try:
        pose=locate_paper_cup(s.rgbd(),s.data.time,p)
        np.testing.assert_allclose(pose['position_world'][:2],[-.03,.06],atol=.006)
        import mujoco
        assert mujoco.mj_name2id(s.model,mujoco.mjtObj.mjOBJ_GEOM,'paper_rim_0')>=0
        assert mujoco.mj_name2id(s.model,mujoco.mjtObj.mjOBJ_GEOM,'coffee_handle_0')==-1
    finally:s.close()


def test_rgbd_checkpoint_requires_depth_and_does_not_call_teacher(monkeypatch):
    import json,h5py
    from aloha2_lab import visual_il
    root=visual_il.ROOT/'data/paper-rgbd-v2'
    if not (root/'model/training.json').exists():pytest.skip('Requires trained paper RGB-D artifact')
    def forbidden(*args,**kwargs):raise AssertionError('Teacher called during RGB-D inference')
    monkeypatch.setattr(visual_il,'ik',forbidden);monkeypatch.setattr(visual_il,'locate_cup',forbidden);monkeypatch.setattr(visual_il,'locate_paper_cup',forbidden)
    policy=visual_il.load_policy(root/'model')
    with h5py.File(sorted((root/'train').glob('*.hdf5'))[0]) as f:rgb=f['initial_rgb'][:];depth=f['initial_depth_m'][:]
    a,z=policy(rgb,depth)
    assert a.shape==(820,14) and np.isfinite(a).all() and z.shape==(3,)
    with pytest.raises(ValueError,match='requires'):policy(rgb)
    with pytest.raises(ValueError,match='valid depth'):policy(rgb,np.zeros_like(depth))
