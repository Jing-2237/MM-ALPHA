"""Reproduce the fixed-training-image ablation without changing canonical results."""
import json
from pathlib import Path
import h5py,numpy as np
from aloha2_lab.sim import AlohaSim
from aloha2_lab.visual_il import load_policy,measure
import argparse
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
policy=load_policy(root/'model')
with h5py.File(root/'train/episode_000012.hdf5') as f:reference=f['initial_rgb'][:]
actions,_=policy(reference);manifest=json.loads((root/'split.json').read_text());rows=[]
for i,xy in enumerate(manifest['test']):
 s=AlohaSim(cup_xy=xy)
 try:
  hold=[]
  for t,a in enumerate(actions):
   lo,hi=s.model.actuator_ctrlrange.T;s.step(np.clip(a,lo,hi))
   if t>=720:hold.append(float(s.data.body('cup').xpos[2]))
  row=dict(index=i,spawn_xy=xy,**measure(s,hold));rows.append(row);print(row,flush=True)
 finally:s.close()
(out/'trials.json').write_text(json.dumps(rows,indent=2))
print('Fixed reference success:',sum(x['success'] for x in rows),'/',len(rows))
