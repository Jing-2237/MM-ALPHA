"""Check published artifacts and load the final policy without commanding hardware."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
    for item in manifest['artifacts']:
        path = ROOT / item['path']
        if path.stat().st_size != item['bytes']:
            raise ValueError(f'Size mismatch: {item["path"]}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'SHA-256 mismatch: {item["path"]}')
    root = ROOT / manifest['experiment_root']
    training = json.loads((root / 'model/training.json').read_text())
    paths = training['train_episodes'] + training['validation_episodes']
    if len(training['train_episodes']) != 17 or len(training['validation_episodes']) != 6:
        raise ValueError('Unexpected final-model dataset')
    for value in paths:
        if Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('Published episode paths must be relative to experiment root')
        with h5py.File(root / value) as episode:
            if not episode.attrs['task_success'] or not episode.attrs['replay_pass']:
                raise ValueError(f'Unverified successful episode: {value}')
    split = json.loads((root / 'split.json').read_text())
    fitting = np.array(split['train'] + split['validation'])
    for xy in split['test']:
        if np.linalg.norm(fitting - xy, axis=1).min() <= .0069:
            raise ValueError('Training/test spatial overlap')
    report = json.loads((root / 'test/report.json').read_text())
    if report['successes'] != 9 or report['trials'] != 10 or training['test_episodes_used'] != 0:
        raise ValueError('Unexpected published evaluation')
    from aloha2_lab.visual_il import load_policy
    with h5py.File(root / paths[0]) as episode:
        rgb = episode['initial_rgb'][:]
        depth = episode['initial_depth_m'][:]
    policy = load_policy(root / 'model')
    actions, _ = policy(rgb, depth)
    if actions.shape != (960, 14) or not np.isfinite(actions).all():
        raise ValueError('Invalid final-model output')
    np.testing.assert_allclose(actions[:480, 6], .037, atol=1e-6)
    print(json.dumps(dict(verified_artifacts=len(manifest['artifacts']),
                         verified_successful_episodes=len(paths),
                         action_shape=list(actions.shape),
                         hardware_motion_commanded=False), indent=2))


if __name__ == '__main__':
    main()
