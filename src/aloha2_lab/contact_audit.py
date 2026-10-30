"""Substep collision audit and independent grasp validation (never a pose controller)."""
import numpy as np
import mujoco

class ContactAudit:
    def __init__(self,sim,limit=.0005):
        self.limit=limit;self.max_penetration=0.;self.max_finger_force=0.;self.nonfinger_contacts=0
        self.min_open_gap=None;self.max_hand_table_penetration=0.;self.phase='open';self.preclose_penetration=0.
        self.cup=sim.model.body('cup').id
        self.fingers={sim.model.body('left/left_finger_link').id,sim.model.body('left/right_finger_link').id}
        self.robot=np.array([b.name.startswith(('left/','right/')) for b in [sim.model.body(i) for i in range(sim.model.nbody)]])
        self.forces={b:0. for b in self.fingers}
        self.hand={sim.model.body(name).id for name in ['left/left_finger_link','left/right_finger_link','left/gripper_base','left/gripper_link','left/wrist_link']}
        self.table={sim.model.geom(name).id for name in ['table','floor']}
    def update(self,sim):
        m,d=sim.model,sim.data;self.forces={b:0. for b in self.fingers}
        contacts=d.contact
        if not len(contacts):return
        pairs=contacts.geom;body=m.geom_bodyid[pairs]
        for idx in np.flatnonzero(((np.isin(pairs[:,0],list(self.table)))&np.isin(body[:,1],list(self.hand)))|((np.isin(pairs[:,1],list(self.table)))&np.isin(body[:,0],list(self.hand)))):
            self.max_hand_table_penetration=max(self.max_hand_table_penetration,max(0.,-float(contacts.dist[idx])))
        mask=(body[:,0]==self.cup)^(body[:,1]==self.cup)
        for idx in np.flatnonzero(mask):
            a,b=body[idx];other=int(b if a==self.cup else a)
            if not self.robot[other]:continue
            pen=max(0.,-float(contacts.dist[idx]));self.max_penetration=max(self.max_penetration,pen)
            if self.phase in ('open','approach'):self.preclose_penetration=max(self.preclose_penetration,pen)
            force=np.zeros(6);mujoco.mj_contactForce(m,d,int(idx),force)
            if other in self.fingers:
                self.forces[other]+=max(0.,float(force[0]))
            elif force[0]>.01:self.nonfinger_contacts+=1
        self.max_finger_force=max(self.max_finger_force,*self.forces.values())
    def step(self,sim,action):
        action=np.asarray(action)
        if action.shape!=(14,) or not np.isfinite(action).all():raise ValueError('Invalid audited action')
        lo,hi=sim.model.actuator_ctrlrange.T
        if np.any(action<lo) or np.any(action>hi):raise ValueError('Audited action exceeds unchanged physical control limits')
        sim.data.ctrl[:]=action
        for _ in range(sim.substeps):
            mujoco.mj_step(sim.model,sim.data);self.update(sim)
    def report(self):
        return dict(max_penetration_m=self.max_penetration,max_preclose_penetration_m=self.preclose_penetration,
                    max_hand_table_penetration_m=self.max_hand_table_penetration,max_finger_force_n=self.max_finger_force,nonfinger_contact_substeps=self.nonfinger_contacts,
                    penetration_limit_m=self.limit,collision_pass=self.max_penetration<=self.limit and self.max_hand_table_penetration<=self.limit and self.preclose_penetration<=.0001 and self.nonfinger_contacts==0)


class HoldAudit:
    def __init__(self):self.positions=[];self.tilts=[]
    def update(self,sim):
        tcp=sim.data.site('left/gripper');cup=sim.data.body('cup')
        self.positions.append(tcp.xmat.reshape(3,3).T@(cup.xpos-tcp.xpos))
        up=cup.xmat.reshape(3,3)[:,2]
        self.tilts.append(float(np.degrees(np.arccos(np.clip(up[2],-1,1)))))
    def report(self):
        p=np.array(self.positions)
        drift=float(np.max(np.linalg.norm(p-p[0],axis=1))) if len(p) else float('inf')
        tilt=max(self.tilts,default=float('inf'))
        return dict(hold_slip_m=drift,max_hold_tilt_degrees=tilt,stable_hold=drift<=.005 and tilt<=20.)


def side_pinch_report(sim,minimum_force=.05):
    """Require opposing lateral forces on CUP WALLS, not support under the bottom."""
    m,d=sim.model,sim.data;cup=m.body('cup').id
    fingers={m.body('left/left_finger_link').id:'left',m.body('left/right_finger_link').id:'right'}
    axis=d.site('left/gripper').xmat.reshape(3,3)[:,1]
    values={'left':[],'right':[]}
    for i,c in enumerate(d.contact):
        a,b=int(c.geom1),int(c.geom2);ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
        if ba==cup and bb in fingers:wall=a;other=bb;normal=-c.frame[:3]
        elif bb==cup and ba in fingers:wall=b;other=ba;normal=c.frame[:3]
        else:continue
        if not m.geom(wall).name.startswith('paper_wall_'):continue
        projection=float(normal@axis)
        if abs(projection)<.7:continue
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        if force[0]>.05:values[fingers[other]].append((projection,float(force[0])))
    forces={key:sum(v[1] for v in val) for key,val in values.items()}
    directions={key:np.sign(sum(v[0]*v[1] for v in val)) for key,val in values.items()}
    valid=all(forces[k]>minimum_force for k in forces) and directions['left']*directions['right']<0
    return dict(side_wall_pinch=bool(valid),side_wall_normal_forces_n=forces)
