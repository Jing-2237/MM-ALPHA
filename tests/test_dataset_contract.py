import json
import h5py
import numpy as np
import pytest
from aloha2_lab.dataset import EpisodeWriter, validate
from aloha2_lab.sim import JOINTS


def write_episode(path, frames=3, dt=.02, offset=0., images=False):
    writer = EpisodeWriter(path, "test_fixture", dt=dt)
    for i in range(frames):
        q = np.full(14,offset+i*.01)
        image = {"overhead_cam":np.zeros((8,12,3),np.uint8)} if images else None
        writer.append(q,np.zeros(14),q+.02,i*dt,image)
    writer.close()
    return writer


def test_observation_action_pairing_and_camera_roundtrip(tmp_path):
    path = tmp_path/"episode_000000.hdf5"
    write_episode(path,images=True)
    assert validate(path)["frames"] == 3
    assert not path.with_suffix(".hdf5.partial").exists()
    with h5py.File(path) as f:
        np.testing.assert_allclose(f["action"][:]-f["observations/qpos"][:],.02,atol=1e-8)
        assert f["observations/images/overhead_cam"].shape == (3,8,12,3)
        assert json.loads(f.attrs["joint_names"]) == JOINTS


def test_incomplete_and_short_episode_not_published(tmp_path):
    path = tmp_path/"episode_000000.hdf5"
    writer = write_episode(path,frames=1)
    assert not path.exists()
    assert writer.partial.exists()
    with pytest.raises(ValueError,match="complete"):
        validate(writer.partial)


def test_existing_episode_never_overwritten(tmp_path):
    path = tmp_path/"episode_000000.hdf5"
    write_episode(path)
    old = path.read_bytes()
    with pytest.raises(FileExistsError):
        EpisodeWriter(path,"another")
    assert old == path.read_bytes()


def test_rejected_camera_frame_does_not_desynchronize_arrays(tmp_path):
    path = tmp_path/"episode_000000.hdf5"
    writer = EpisodeWriter(path,"test")
    q = np.zeros(14)
    good = {"overhead_cam":np.zeros((8,12,3),np.uint8)}
    writer.append(q,q,q,0.,good)
    with pytest.raises(ValueError,match="dimensions"):
        writer.append(q,q,q,.02,{"overhead_cam":np.zeros((12,12,3),np.uint8)})
    assert writer.file["action"].shape[0] == 1
    writer.append(q,q,q,.02,good)
    writer.close()
    assert validate(path)["frames"] == 2


@pytest.mark.parametrize("bad",[np.nan,np.inf])
def test_nonfinite_samples_rejected(tmp_path,bad):
    writer = EpisodeWriter(tmp_path/"episode.hdf5","test")
    q = np.zeros(14)
    action = q.copy()
    action[3] = bad
    try:
        with pytest.raises(ValueError,match="non-finite"):
            writer.append(q,q,action,0.)
    finally:
        writer.close(complete=False)


@pytest.mark.parametrize("attribute,value",[
    ("gripper_unit","motor_radians"),
    ("observation_action_alignment","observation_after_action"),
    ("sim",False),
    ("dt",-1.),
    ("joint_names",json.dumps(JOINTS[::-1])),
])
def test_incompatible_metadata_rejected(tmp_path,attribute,value):
    path = tmp_path/"episode.hdf5"
    write_episode(path)
    with h5py.File(path,"r+") as f:
        f.attrs[attribute] = value
    with pytest.raises(ValueError):
        validate(path)


def test_irregular_simulation_sampling_rejected(tmp_path):
    writer = EpisodeWriter(tmp_path/"episode.hdf5","test")
    q = np.zeros(14)
    try:
        writer.append(q,q,q,0.)
        with pytest.raises(ValueError,match="Irregular"):
            writer.append(q,q,q,.03)
    finally:
        writer.close(complete=False)
