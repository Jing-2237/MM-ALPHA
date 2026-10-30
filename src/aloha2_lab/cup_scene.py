import xml.etree.ElementTree as ET
import numpy as np
import mujoco

def cup_model(scene, xy):
    if np.shape(xy)!=(2,) or not np.isfinite(xy).all():raise ValueError('Invalid cup spawn')
    tree=ET.parse(scene).getroot()
    tree.find('visual/quality').set('shadowsize','2048')
    tree.find('compiler').set('meshdir',str(scene.parent/'assets'))
    tree.find('compiler').set('texturedir',str(scene.parent/'assets'))
    tree.find('include').set('file',str(scene.parent/'aloha.xml'))
    body=ET.SubElement(tree.find('worldbody'),'body',name='cup',pos=f'{xy[0]} {xy[1]} .043')
    ET.SubElement(body,'freejoint',name='cup_free')
    ET.SubElement(body,'geom',name='cup_bottom',type='cylinder',size='.024 .003',pos='0 0 -.037',mass='.015',rgba='.04 .12 .9 1',friction='1.5 .02 .002',condim='4')
    for i in range(20):
        a=i*2*np.pi/20
        ET.SubElement(body,'geom',name=f'cup_wall_{i}',type='box',size='.002 .004 .04',
                      pos=f'{.022*np.cos(a)} {.022*np.sin(a)} 0',euler=f'0 0 {a}',mass='.003',
                      rgba='.04 .12 .9 1',friction='1.5 .02 .002',condim='4')
    # Open C-shaped ceramic handle in the vertical XZ plane; real collision geometry.
    angles=np.linspace(-np.pi/2,np.pi/2,13)
    for i,(a,b) in enumerate(zip(angles[:-1],angles[1:])):
        p1=[.026+.027*np.cos(a),0,.027*np.sin(a)]
        p2=[.026+.027*np.cos(b),0,.027*np.sin(b)]
        ET.SubElement(body,'geom',name=f'coffee_handle_{i}',type='capsule',size='.004',
                      fromto=' '.join(map(str,p1+p2)),mass='.0015',rgba='.9 .86 .72 1',friction='1.5 .02 .002',condim='4')
    # Coffee surface is visual only; no claim of liquid simulation.
    ET.SubElement(body,'geom',name='coffee_surface',type='cylinder',size='.018 .001',pos='0 0 .02',
                  rgba='.09 .035 .01 1',mass='0',contype='0',conaffinity='0')
    # Included compiler uses relative assets; resolve that include in a virtual FS.
    include=ET.parse(scene.parent/'aloha.xml').getroot()
    include.find('compiler').set('meshdir',str(scene.parent/'assets'))
    return mujoco.MjModel.from_xml_string(ET.tostring(tree).decode(),assets={str(scene.parent/'aloha.xml'):ET.tostring(include)})
