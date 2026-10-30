import json
import numpy as np
import mujoco
from aloha2_lab.sim import AlohaSim,ROOT
from aloha2_lab.contact_audit import ContactAudit


def test_collision_revision_keeps_dimensions_and_real_finger_travel():
    profile=json.loads((ROOT/'config/paper_cup_v3.json').read_text())
    assert profile['bottom_radius_m']==.025 and profile['top_radius_m']==.05
    s=AlohaSim(cup_xy=(-.02,.06),cup_profile=profile)
    try:
        assert s.model.opt.timestep==.001
        np.testing.assert_allclose(s.model.actuator_ctrlrange[6],[.002,.037])
        np.testing.assert_allclose(s.model.actuator_forcerange[6],[-3,3])
        for name in ['paper_wall_0','paper_bottom','paper_rim_0']:
            g=s.model.geom(name).id
            assert s.model.geom_contype[g]!=0
            assert s.model.geom_priority[g]==1
            np.testing.assert_allclose(s.model.geom_solref[g],[.003,1])
    finally:s.close()


def test_collision_metric_does_not_treat_lift_as_success():
    profile=json.loads((ROOT/'config/paper_cup_v3.json').read_text())
    s=AlohaSim(cup_xy=(-.02,.06),cup_profile=profile)
    try:
        audit=ContactAudit(s)
        assert audit.report()['collision_pass']
        audit.max_penetration=.008
        assert not audit.report()['collision_pass']
        audit.max_penetration=0;audit.nonfinger_contacts=1
        assert not audit.report()['collision_pass']
    finally:s.close()


def test_retrained_policy_opens_before_side_entry():
    import h5py
    import pytest
    from aloha2_lab.visual_il import load_policy
    root=ROOT/'data/paper-rgbd-v3'
    if not (root/'model/training.json').exists():pytest.skip('Requires retrained v3 artifact')
    p=sorted((root/'train').glob('*.hdf5'))[0]
    with h5py.File(p) as f:
        rgb=f['initial_rgb'][:];depth=f['initial_depth_m'][:]
        assert f['observations/qpos'][59,6]>.0365
    actions,_=load_policy(root/'model')(rgb,depth)
    assert actions.shape==(940,14)
    np.testing.assert_allclose(actions[:480,6],.037,atol=1e-6)
    np.testing.assert_allclose(actions[:60,:6],np.tile([0.,-.6,.6,0.,.4,0.],(60,1)),atol=1e-6)


def test_unverified_contact_aware_dataset_cannot_be_trained(tmp_path):
    import pytest
    from aloha2_lab.visual_il import train
    (tmp_path/'split.json').write_text(json.dumps({'sensor':'rgbd','cup_profile':{'teacher_revision':4}}))
    with pytest.raises(ValueError,match='verify_side_dataset'):
        train(tmp_path,epochs=1)
    assert not (tmp_path/'model').exists()


def test_final_policy_uses_replayed_sources_and_opens_before_approach():
    import h5py
    import pytest
    from aloha2_lab.visual_il import load_policy
    root=ROOT/'data/paper-rgbd-v4'
    if not (root/'model/training.json').exists():pytest.skip('Requires retrained v4 artifact')
    training=json.loads((root/'model/training.json').read_text())
    paths=[root/path for path in training['train_episodes']+training['validation_episodes']]
    assert len(paths)==23 and training['test_episodes_used']==0
    for path in paths:
        with h5py.File(path) as f:
            assert f.attrs['replay_pass']
    with h5py.File(paths[0]) as f:
        rgb=f['initial_rgb'][:];depth=f['initial_depth_m'][:]
        assert f['observations/qpos'][59,6]>.0365
    actions,_=load_policy(root/'model')(rgb,depth)
    assert actions.shape==(960,14)
    np.testing.assert_allclose(actions[:480,6],.037,atol=1e-6)
    np.testing.assert_allclose(actions[:60,:6],np.tile([0.,-.6,.6,0.,.4,0.],(60,1)),atol=1e-6)
