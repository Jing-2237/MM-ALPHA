"""Independent saved-action replays. Never calls IK or perception; no hardware."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
from aloha2_lab.visual_il import make_sim,measure,execution_audits,audited_step
from aloha2_lab.contact_audit import side_pinch_report

def verify(root):
 root=Path(root);rows=[]
 for split in ('train','validation'):
  for p in sorted((root/split).glob('*.hdf5')):
   with h5py.File(p) as f:
    if not f.attrs.get('task_success',False):continue
    xy=f.attrs['scene_cup_xy'];actions=f['action'][:]
   sim=make_sim(root,xy);audit,hold_audit=execution_audits(root,sim);hold=[]
   row=dict(split=split,file=p.relative_to(root).as_posix())
   try:
    for t,a in enumerate(actions):
     lo,hi=sim.model.actuator_ctrlrange.T;audited_step(sim,np.clip(a,lo,hi),t,len(actions),audit,hold_audit)
     if t>=len(actions)-100:hold.append(float(sim.data.body('cup').xpos[2]))
    row.update(measure(sim,hold));row.update(audit.report());row.update(hold_audit.report());row.update(side_pinch_report(sim,audit.minimum_side_force))
    row['replay_pass']=row['success'] and row['collision_pass'] and row['stable_hold'] and row['side_wall_pinch']
   except Exception as exc:row.update(replay_pass=False,error=f'{type(exc).__name__}: {exc}')
   finally:sim.close()
   with h5py.File(p,'r+') as f:
    f.attrs['replay_pass']=bool(row['replay_pass']);f.attrs['replay_evaluation']=json.dumps(row)
   rows.append(row);(root/'saved-action-replay.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 return rows

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();rows=verify(a.root)
 print('Replay eligible:',sum(r['replay_pass'] for r in rows),'/',len(rows))
