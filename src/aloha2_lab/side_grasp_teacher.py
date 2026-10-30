"""Contact-aware side-entry demonstrations within unchanged finger travel."""
import numpy as np
from .grasp import ik
from .contact_audit import ContactAudit,HoldAudit,side_pinch_report


def execute(sim,centre,profile,record):
    audit=ContactAudit(sim,profile['max_penetration_m']);hold_audit=HoldAudit();hold=[]
    angle=np.radians(profile['grasp_pitch_degrees'])
    R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    target=np.array(centre,float);target[0]+=profile['grasp_tcp_x_offset_m'];target[2]=profile['grasp_height_m']
    def step(a):
        a=np.asarray(a,dtype=np.float32).astype(float)
        lo,hi=sim.model.actuator_ctrlrange.T;a=np.clip(a,lo,hi)
        record(a);audit.step(sim,a)
        if audit.max_penetration>profile['max_penetration_m'] or audit.max_hand_table_penetration>profile['max_penetration_m'] or audit.nonfinger_contacts:
            raise RuntimeError('Rejected teacher: collision/penetration budget exceeded')
    for _ in range(profile['initial_open_steps']):
        a=sim.data.ctrl.copy();a[6]=.037;step(a)
    left=sim.data.geom('left/left_g2').xpos;right=sim.data.geom('left/right_g2').xpos
    # Actual fingertip pad geometry, not twice the rail command.
    opening=float(np.linalg.norm(left-right)-.0012)
    if opening<.056:raise RuntimeError('Gripper failed to fully open before approach')
    audit.phase='approach'
    def move(pos,g,n,cartesian=False):
        if cartesian:
            startpos=sim.data.site('left/gripper').xpos.copy()
            for i in range(n):
                u=(i+1)/n;u=u*u*(3-2*u)
                a=sim.data.ctrl.copy();a[:6]=ik(sim,startpos+(pos-startpos)*u,R);a[6]=g;step(a)
        else:
            start=sim.data.ctrl.copy();end=start.copy();end[:6]=ik(sim,pos,R);end[6]=g
            for i in range(n):
                u=(i+1)/n;u=u*u*(3-2*u);step(start+(end-start)*u)
    move(target-R[:,0]*.12,.037,180)
    move(target,.037,240,True)
    audit.phase='close';g=.037
    for _ in range(120):
        if min(audit.forces.values())<.3:g=max(profile['close_target_m'],g-.00006)
        a=sim.data.ctrl.copy();a[6]=g;step(a)
    for _ in range(profile.get('extra_squeeze_steps',0)):
        g=max(profile['close_target_m'],g-profile.get('extra_squeeze_step_m',0))
        a=sim.data.ctrl.copy();a[6]=g;step(a)
    audit.phase='lift';move(target+[0,0,.16],g,240,True)
    audit.phase='hold'
    for _ in range(100):
        step(sim.data.ctrl.copy());hold.append(float(sim.data.body('cup').xpos[2]));hold_audit.update(sim)
    return dict(hold=hold,**side_pinch_report(sim,profile.get('minimum_side_force_n',.05)),**audit.report(),**hold_audit.report(),open_tip_gap_m=opening,closed_target_m=g)
