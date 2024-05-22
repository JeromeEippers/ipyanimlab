from pxr import Usd, UsdSkel, UsdGeom, Sdf
import numpy as np

from ..animation import Anim
from .. import utils


def read_usd_animation(stage):
    if stage.HasDefaultPrim() == False:
        raise Exception('missing default prim')

    root = stage.GetDefaultPrim()
    cm_per_unit = UsdGeom.GetStageMetersPerUnit(stage) * 100.0
    
    skelCache = UsdSkel.Cache()
    skelPrim = next(prim for prim in Usd.PrimRange(root) if prim.IsA(UsdSkel.Skeleton))

    skel = UsdSkel.Skeleton(skelPrim)
    skelQuery = skelCache.GetSkelQuery(skel)
    
    names = []
    parents = []
    for i, jointToken in enumerate(skelQuery.GetJointOrder()):
        jointPath = Sdf.Path(jointToken)
        names.append(jointPath.name)
        parents.append(skelQuery.GetTopology().GetParent(i))
    
    startFrame = int(stage.GetStartTimeCode())
    endFrame = int(stage.GetEndTimeCode())
    
    xforms = np.zeros([endFrame-startFrame+1, len(names), 4, 4], dtype=np.float32)
    
    for i, frame in enumerate(range(startFrame,endFrame+1)):
        xforms[i, :, :, :] = np.array(skelQuery.ComputeJointLocalTransforms(frame), dtype=np.float32)
    
    # transpose
    xforms = np.einsum('ijkl->ijlk', xforms)
    # scale
    xforms[:, :, :3, 3] *= cm_per_unit
    # convert to quats pos
    q, p = utils.m4x4_to_qp(xforms)
    
    return Anim(q, p, p[0], parents, names)


def import_usd_animation(filepath, anim_mapper=None):
    """Import a usd animation

    Args:
        :filepath: (str): the path of the usd stage
        :anim_mapper: (AnimMapper, optional): the AnimMapper to use when matching to a character. Defaults to None.

    Returns:
        :Anim: the animation
    """
    stage = Usd.Stage.Open('example_animation.usd')
    anim = read_usd_animation(stage)

    if anim_mapper is not None:
        anim = anim_mapper(anim)

    return anim