#!/usr/bin/env python3
"""Validate pinned local upstreams and restore the stationary ROS workspace links.

No network, package installation, hardware launch or repository reset is performed.
"""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / 'config/hardware-sources.lock.json').read_text())
    for name, spec in lock['repositories'].items():
        path = ROOT / 'third_party' / name
        revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != spec['commit']:
            raise RuntimeError(f'{name}: revision changed; review before updating the lock')
    destination = ROOT / 'ros2_ws/src'
    destination.mkdir(parents=True, exist_ok=True)
    for name, relative in lock['packages'].items():
        target = ROOT / relative
        if not (target / 'package.xml').is_file():
            raise FileNotFoundError(target)
        link = destination / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f'Conflicting existing link: {link}')
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(os.path.relpath(target, destination), target_is_directory=True)
        print(f'{name}: {relative}')
    print('Stationary workspace ready. Mobile-only SLATE base is intentionally not linked.')


if __name__ == '__main__':
    main()
