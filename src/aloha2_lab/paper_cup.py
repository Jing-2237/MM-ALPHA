"""Parametric open frustum paper cup; rigid approximation with decorative branding."""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def build_model(scene,xy,profile):
    rb=float(profile['bottom_radius_m']);rt=float(profile['top_radius_m']);h=float(profile['height_m'])
    thickness=float(profile.get('wall_thickness_m',.0015));mass=float(profile.get('mass_kg',.018))
    if not (0<thickness<rb<=rt and .02<h<.4 and mass>0):raise ValueError('Invalid paper-cup dimensions')
    tree=ET.parse(scene).getroot();tree.find('visual/quality').set('shadowsize','2048')
    for key in ('meshdir','texturedir'):tree.find('compiler').set(key,str(scene.parent/'assets'))
    tree.find('include').set('file',str(scene.parent/'aloha.xml'))
    robust=profile.get('contact_revision',0)>=3
    if robust:
        ET.SubElement(tree,'option',timestep='.001',iterations='100',tolerance='1e-10',integrator='implicitfast')
    body=ET.SubElement(tree.find('worldbody'),'body',name='cup',pos=f'{xy[0]} {xy[1]} {h/2+.001}')
    ET.SubElement(body,'freejoint',name='cup_free')
    ET.SubElement(body,'geom',name='paper_bottom',type='cylinder',size=f'{rb} .001',pos=f'0 0 {-h/2}',mass=str(mass*.2),rgba='.96 .94 .86 1',friction='1.2 .02 .002',condim='4')
    count=48
    for i in range(count):
        a,b=np.array([i,i+1])*2*np.pi/count
        vertices=[]
        for z,r in [(-h/2,rb),(h/2,rt)]:
            for radius in (r-thickness,r):
                for angle in (a,b):vertices.extend([radius*np.cos(angle),radius*np.sin(angle),z])
        name=f'paper_wall_mesh_{i}'
        ET.SubElement(tree.find('asset'),'mesh',name=name,vertex=' '.join(map(str,vertices)))
        ET.SubElement(body,'geom',name=f'paper_wall_{i}',type='mesh',mesh=name,mass=str(mass*.8/count),rgba='.96 .94 .86 1',friction='1.2 .02 .002',condim='4')
        p1=[rt*np.cos(a),rt*np.sin(a),h/2];p2=[rt*np.cos(b),rt*np.sin(b),h/2]
        ET.SubElement(body,'geom',name=f'paper_rim_{i}',type='capsule',size=str(thickness),fromto=' '.join(map(str,p1+p2)),mass='0',rgba='.98 .97 .9 1')
    slope=(rt-rb)/h;angle=np.arctan(slope);r=(rb+rt)/2
    ET.SubElement(body,'geom',name='brand_red',type='box',size='.012 .0003 .016',pos=f'0 {-r-.001} 0',euler=f'{angle} 0 0',mass='0',rgba='.75 .025 .025 1',contype='0',conaffinity='0')
    # Code-native yellow mark, a visual approximation rather than a reconstructed photograph.
    points=[(-.009,-.01),(-.005,.01),(0,-.003),(.005,.01),(.009,-.01)]
    for i,((x1,z1),(x2,z2)) in enumerate(zip(points[:-1],points[1:])):
        p1=[x1,-r-.0015-z1*np.sin(angle),z1*np.cos(angle)]
        p2=[x2,-r-.0015-z2*np.sin(angle),z2*np.cos(angle)]
        ET.SubElement(body,'geom',name=f'brand_yellow_{i}',type='capsule',size='.0013',fromto=' '.join(map(str,p1+p2)),mass='0',rgba='1 .72 .01 1',contype='0',conaffinity='0')
    include=ET.parse(scene.parent/'aloha.xml').getroot();include.find('compiler').set('meshdir',str(scene.parent/'assets'))
    if robust:
        for geom in body.findall('geom'):
            if geom.get('contype','1')!='0':
                geom.set('priority','1');geom.set('solref','.003 1');geom.set('solimp','.99 .999 .0005')
        # Preserve actual travel/geometry. Limit only simulated actuator force.
        actuators=ET.parse(scene.parent/'joint_position_actuators.xml').getroot()
        for actuator in actuators.findall('.//position'):
            if actuator.get('name')=='left/gripper':
                actuator.set('forcelimited','true');actuator.set('forcerange','-3 3')
        return mujoco.MjModel.from_xml_string(ET.tostring(tree).decode(),assets={str(scene.parent/'aloha.xml'):ET.tostring(include),str(scene.parent/'joint_position_actuators.xml'):ET.tostring(actuators)})
    return mujoco.MjModel.from_xml_string(ET.tostring(tree).decode(),assets={str(scene.parent/'aloha.xml'):ET.tostring(include)})
