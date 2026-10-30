"""Streaming, atomic episodes; observations are sampled BEFORE their actions."""
import json
import os
from pathlib import Path
import h5py
import numpy as np
from .sim import JOINTS

SCHEMA = "aloha2-lab/sim-v1"


class EpisodeWriter:
    def __init__(self, path, source, dt=.02):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_suffix(".hdf5.partial")
        if self.path.exists():
            raise FileExistsError(self.path)
        self.file = h5py.File(self.partial, "x")
        self.file.attrs.update(schema=SCHEMA, source=source, sim=True, dt=dt,
                               joint_names=json.dumps(JOINTS),
                               gripper_unit="single_finger_displacement_m",
                               observation_action_alignment="observation_before_action", complete=False)
        self.n = 0
        self.last_time = -float("inf")
        self.keys = None
        self.dt = float(dt)

    def append(self, qpos, qvel, action, timestamp, images=None):
        fields = {"observations/qpos": np.asarray(qpos, np.float32),
                  "observations/qvel": np.asarray(qvel, np.float32),
                  "action": np.asarray(action, np.float32), "timestamp": np.asarray(timestamp, np.float64)}
        for key in ("observations/qpos", "observations/qvel", "action"):
            if fields[key].shape != (14,) or not np.isfinite(fields[key]).all():
                raise ValueError(f"Bad shape or non-finite value: {key}")
        if np.ndim(timestamp) != 0 or not np.isfinite(timestamp) or timestamp <= self.last_time:
            raise ValueError("Timestamps must strictly increase")
        if self.n and not np.isclose(timestamp-self.last_time, self.dt, rtol=0, atol=1e-7):
            raise ValueError("Irregular simulation sampling")
        for name, img in (images or {}).items():
            if not isinstance(name, str) or not name or "/" in name:
                raise ValueError("Camera names must be nonempty path components")
            img = np.asarray(img)
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3 or min(img.shape) <= 0:
                raise ValueError("Images must be HxWx3 uint8 RGB")
            fields[f"observations/images/{name}"] = img
        if self.keys is not None and set(fields) != self.keys:
            raise ValueError("Camera set changed during episode")
        # Validate every shape before extending any dataset. A rejected camera
        # frame must not leave state/action datasets one sample ahead of images.
        for key, value in fields.items():
            if key in self.file and self.file[key].shape[1:] != value.shape:
                raise ValueError("Frame dimensions changed")
        for key, value in fields.items():
            if key not in self.file:
                self.file.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                         dtype=value.dtype, chunks=(1, *value.shape), compression="lzf")
            ds = self.file[key]
            if ds.shape[1:] != value.shape:
                raise ValueError("Frame dimensions changed")
            ds.resize(self.n + 1, axis=0)
            ds[self.n] = value
        self.keys = set(fields)
        self.n += 1
        self.last_time = timestamp

    def close(self, complete=True):
        if not self.file.id.valid:
            return
        self.file.attrs["complete"] = bool(complete and self.n >= 2)
        self.file.flush()
        self.file.close()
        if complete and self.n >= 2:
            validate(self.partial)
            # Hard-link creation fails if another writer claimed the final path.
            os.link(self.partial, self.path)
            self.partial.unlink()


def validate(path):
    with h5py.File(path) as f:
        if f.attrs.get("schema") != SCHEMA or not f.attrs.get("complete", False):
            raise ValueError("Not a complete supported simulation episode")
        if not f.attrs.get("sim", False):
            raise ValueError("Simulation schema cannot contain hardware episodes")
        if f.attrs.get("observation_action_alignment") != "observation_before_action":
            raise ValueError("Observation/action alignment mismatch")
        dt = float(f.attrs.get("dt", float("nan")))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid control dt")
        if f.attrs.get("gripper_unit") != "single_finger_displacement_m":
            raise ValueError("Gripper units do not match")
        if json.loads(f.attrs["joint_names"]) != JOINTS:
            raise ValueError("Joint order mismatch")
        n = len(f["action"])
        if n < 2:
            raise ValueError("Episode too short")
        for key in ("observations/qpos", "observations/qvel", "action"):
            if f[key].shape != (n, 14) or not np.isfinite(f[key][:]).all():
                raise ValueError(f"Invalid array: {key}")
        ts = f["timestamp"][:]
        if ts.shape != (n,) or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            raise ValueError("Invalid timestamps")
        if not np.allclose(np.diff(ts), dt, rtol=0, atol=1e-7):
            raise ValueError("Irregular simulation sampling")
        if "observations/images" in f:
            for camera, ds in f["observations/images"].items():
                if not isinstance(ds,h5py.Dataset) or ds.ndim != 4 or ds.shape[0] != n or ds.dtype != np.uint8 or ds.shape[-1] != 3 or min(ds.shape) <= 0:
                    raise ValueError(f"Invalid camera stream: {camera}")
        return {"path": str(path), "frames": n, "source": str(f.attrs["source"]), "dt": float(f.attrs["dt"])}
