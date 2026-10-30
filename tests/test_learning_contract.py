import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from aloha2_lab.dataset import EpisodeWriter
from aloha2_lab.learning import chunk_arrays, train, load_policy


def make_episodes(root, periods=(.02,.02,.02,.02,.02)):
    for episode,dt in enumerate(periods):
        writer = EpisodeWriter(root/f"episode_{episode:06d}.hdf5","test_fixture",dt=dt)
        for i in range(4):
            q = np.full(14,episode+i*.01,np.float32)
            writer.append(q,np.zeros(14),q+.02,i*dt)
        writer.close()


def test_action_chunks_do_not_cross_episode_boundary_and_mask_padding():
    x = np.ones((2,28),np.float32)
    a = np.stack([np.zeros(14),np.ones(14)])
    xs,ys,mask = chunk_arrays([(x,a),(x*7,a+10)],3)
    np.testing.assert_array_equal(ys[1,:,0],[1,1,1])
    np.testing.assert_array_equal(mask[1],[True,False,False])
    assert ys[2,0,0] == 10
    assert xs[2,0] == 7


def test_training_split_stats_and_checkpoint_are_consistent(tmp_path):
    data = tmp_path/"data"
    make_episodes(data)
    output = tmp_path/"run"
    report = train(data,output,epochs=2,chunk=3,device="cpu")
    assert set(report["train_episodes"]).isdisjoint(report["validation_episodes"])
    train_observations = []
    for path in report["train_episodes"]:
        with h5py.File(path) as f:
            train_observations.append(np.concatenate([f["observations/qpos"][:],f["observations/qvel"][:]],axis=1))
    ckpt = torch.load(output/"policy.pt",weights_only=True)
    np.testing.assert_allclose(ckpt["stats"]["xmean"].numpy(),np.concatenate(train_observations).mean(0))
    policy = load_policy(output/"policy.pt")
    assert policy.control_dt == .02
    assert policy(np.zeros(14),np.zeros(14)).shape == (3,14)
    assert np.isfinite(policy(np.zeros(14),np.zeros(14))).all()
    with pytest.raises(ValueError,match="finite"):
        policy(np.full(14,np.nan),np.zeros(14))


def test_mixed_control_periods_require_explicit_resampling(tmp_path):
    data = tmp_path/"data"
    make_episodes(data,periods=(.02,.02,.04))
    with pytest.raises(ValueError,match="different control periods"):
        train(data,tmp_path/"run",epochs=1,device="cpu")
    assert not (tmp_path/"run").exists()
