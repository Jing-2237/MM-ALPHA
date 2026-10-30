"""Small state-based action-chunk BC baseline (not ACT, not a visual policy)."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from .dataset import validate, SCHEMA


def network(chunk):
    return nn.Sequential(nn.Linear(28, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 14 * chunk))


def read_episode(path):
    validate(path)
    with h5py.File(path) as f:
        return np.concatenate([f["observations/qpos"][:], f["observations/qvel"][:]], axis=1), f["action"][:]


def chunk_arrays(episodes, chunk):
    if chunk < 1:
        raise ValueError("chunk must be positive")
    xs, ys, masks = [], [], []
    for x, a in episodes:
        for t in range(len(a)):
            indices = np.arange(t, t + chunk)
            xs.append(x[t])
            ys.append(a[np.minimum(indices, len(a)-1)])
            masks.append(indices < len(a))
    return np.stack(xs), np.stack(ys), np.stack(masks)


def train(data, output, epochs=80, chunk=10, seed=7, device="auto"):
    if epochs < 1 or chunk < 1:
        raise ValueError("epochs and chunk must be positive")
    paths = sorted(Path(data).glob("episode_*.hdf5"))
    if len(paths) < 3:
        raise ValueError("Need at least three complete episodes for episode-level train/validation split")
    metadata = [validate(p) for p in paths]
    control_dt = metadata[0]["dt"]
    if not all(np.isclose(m["dt"],control_dt,rtol=0,atol=1e-9) for m in metadata):
        raise ValueError("Episodes have different control periods; resample explicitly before training")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    paths = [paths[i] for i in rng.permutation(len(paths))]
    nval = max(1, round(len(paths) * .2))
    val_paths, train_paths = paths[:nval], paths[nval:]
    tr = [read_episode(p) for p in train_paths]
    va = [read_episode(p) for p in val_paths]
    xall, aall = np.concatenate([e[0] for e in tr]), np.concatenate([e[1] for e in tr])
    stats = dict(xmean=xall.mean(0), xstd=xall.std(0).clip(.001),
                 amean=aall.mean(0), astd=aall.std(0).clip(.001))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    def tensors(episodes):
        x, y, m = chunk_arrays(episodes, chunk)
        return (torch.tensor((x-stats["xmean"])/stats["xstd"], device=dev),
                torch.tensor((y-stats["amean"])/stats["astd"], device=dev),
                torch.tensor(m[..., None], device=dev))
    x, y, mask = tensors(tr)
    vx, vy, vm = tensors(va)
    model = network(chunk).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    def loss(px, ty, m):
        pred = model(px).reshape(-1, chunk, 14)
        return (((pred-ty)**2)*m).sum() / (m.sum()*14)
    history, best = [], float("inf")
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x), device=dev)
        total = 0.
        for ids in order.split(128):
            optimizer.zero_grad()
            value = loss(x[ids], y[ids], mask[ids])
            value.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += value.item() * len(ids)
        model.eval()
        with torch.no_grad():
            vl = loss(vx, vy, vm).item()
        history.append(dict(epoch=epoch+1, train_loss=total/len(x), validation_loss=vl))
        if not np.isfinite(vl):
            raise RuntimeError("Non-finite validation loss")
        if vl < best:
            best = vl
            torch.save(dict(schema=SCHEMA, policy="state_chunk_bc", chunk=chunk, control_dt=control_dt,
                            state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                            stats={k:torch.tensor(v) for k,v in stats.items()}), out/"policy.pt")
    report = dict(device=str(dev), best_validation_loss=best, seed=seed, control_dt=control_dt,
                  train_episodes=[str(p) for p in train_paths], validation_episodes=[str(p) for p in val_paths],
                  history=history, observation="qpos+qvel", policy="state action-chunk BC; no camera input",
                  task_success_evaluated=False)
    (out/"metrics.json").write_text(json.dumps(report, indent=2))
    return {k:v for k,v in report.items() if k != "history"}


def load_policy(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["schema"] != SCHEMA or ckpt["policy"] != "state_chunk_bc":
        raise ValueError("Wrong checkpoint schema")
    model = network(ckpt["chunk"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    stats = ckpt["stats"]
    def policy(q, v):
        if np.shape(q) != (14,) or np.shape(v) != (14,) or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("Policy requires finite 14-dimensional position and velocity observations")
        x = torch.tensor(np.concatenate([q,v]), dtype=torch.float32)
        with torch.no_grad():
            actions = model((x-stats["xmean"])/stats["xstd"]).reshape(-1,14)
            return (actions*stats["astd"]+stats["amean"]).numpy()
    policy.control_dt = float(ckpt["control_dt"])
    if not np.isfinite(policy.control_dt) or policy.control_dt <= 0:
        raise ValueError("Invalid checkpoint control dt")
    return policy
