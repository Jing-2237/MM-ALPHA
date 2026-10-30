"""Replay recorded camera frames to a video, without robot or simulation commands."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import h5py
from .dataset import validate


def export_video(episode,output,camera='overhead_cam'):
    meta=validate(episode)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if output.suffix.lower()!='.mp4':raise ValueError('Video output must end in .mp4')
    if not shutil.which('ffmpeg'):raise RuntimeError('Install ffmpeg before exporting recorded frames')
    output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(episode) as f,tempfile.TemporaryDirectory(prefix='.replay-',dir=output.parent) as scratch:
        key=f'observations/images/{camera}'
        if key not in f:raise ValueError(f'No recorded camera {camera}; choose a camera saved during collection')
        images=f[key];_,height,width,_=images.shape
        target=Path(scratch)/'video.mp4'
        process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{width}x{height}','-r',str(1/meta['dt']),'-i','pipe:0','-an',
            '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-threads','2',
            '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],
            stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for frame in images:process.stdin.write(frame.tobytes())
            process.stdin.close()
            error=process.stderr.read().decode(errors='replace')
            if process.wait()!=0:raise RuntimeError(error)
            os.link(target,output)
        finally:
            if process.poll() is None:
                process.terminate();process.wait(timeout=10)
    report={'episode':str(episode),'video':str(output),'camera':camera,'frames':meta['frames'],
            'fps':1/meta['dt'],'mode':'recorded_camera_frames','hardware_motion_commanded':False}
    return report
