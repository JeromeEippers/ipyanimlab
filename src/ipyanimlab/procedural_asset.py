import numpy as np
from pxr import Usd

from .usd.import_asset import read_usd_asset
from . import assets


def create_sphere(viewer, radius):
    """Create a sphere asset

    Args:
        :viewer: (Viewer): the AnimLab viewer instance
        :radius: (float): the radius of the sphere

    Returns:
        :Asset: the create sphere asset
    """

    stage = Usd.Stage.Open(assets.get_asset_path('sphere.usd'))
    vbuffer, ibuffer = read_usd_asset(stage, False)[:2]

    vbuffer = vbuffer.reshape(-1, 10)
    vbuffer[:, :3] *= radius
    vbuffer = vbuffer.flatten()

    return viewer.create_asset(vbuffer, ibuffer, name='sphere')

    
def create_cube(viewer, width, height, depth):
    """Create a cube asset

    Args:
        :viewer: (Viewer): the AnimLab viewer instance
        :width: (float): the width of the cube
        :height: (float): the height of the cube
        :depth: (float): the depth of the cube

    Returns:
        :Asset: the created cube asset
    """
    stage = Usd.Stage.Open(assets.get_asset_path('cube.usd'))
    vbuffer, ibuffer = read_usd_asset(stage, False)[:2]

    vbuffer = vbuffer.reshape(-1, 10)
    vbuffer[:, :3] *= [width, height, depth]
    vbuffer = vbuffer.flatten()

    return viewer.create_asset(vbuffer, ibuffer, name='cube')