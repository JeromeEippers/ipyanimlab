from pxr import Usd, UsdGeom, Gf, UsdSkel, Sdf
import numpy as np

from .rigid_asset import read_usd_rigid_asset
from .skinned_asset import read_usd_skinned_asset


def read_usd_asset(stage, allow_partial_mesh_buffer=False):
    if stage.HasDefaultPrim() == False:
        raise Exception('missing default prim')
        
    root = stage.GetDefaultPrim()
    cm_per_unit = UsdGeom.GetStageMetersPerUnit(stage) * 100.0
    
    prim_buf=None
    if root.IsA(UsdSkel.Root):
        prim_buf = read_usd_skinned_asset(stage, root)
    else:
        prim_buf = read_usd_rigid_asset(stage, root)

    materials, vbuffer, ibuffer, sibuffer, swbuffer = prim_buf.create_buffers(allow_partial_mesh_buffer)

    # scale the vertices if needed
    if cm_per_unit < 0.99999 or cm_per_unit > 1.00001:
        # multiply each vertices by the matrix
        vbuffer = vbuffer.reshape(-1, 10)
        vbuffer[:, :3] *= cm_per_unit
        vbuffer = vbuffer.flatten()

    bindXforms = None
    restXforms = None
    names = None
    parents = None
    if prim_buf.skeleton is not None:
        bindXforms = prim_buf.skeleton.bind_transforms
        restXforms = prim_buf.skeleton.rest_transforms
        names = prim_buf.skeleton.names
        parents = prim_buf.skeleton.parents

        # scale skeletons
        if cm_per_unit < 0.99999 or cm_per_unit > 1.00001:
            # multiply each vertices by the matrix
            bindXforms[:, :3, 3] *= cm_per_unit
            restXforms[:, :3, 3] *= cm_per_unit

    return (    vbuffer, 
                ibuffer,
                materials,
                sibuffer,
                swbuffer,
                bindXforms,
                restXforms,
                names,
                parents,
                root.GetName()
            )


def import_usd_asset(viewer, path, allow_partial_mesh_buffer=False):
    stage = Usd.Stage.Open(path)
    vbuffer, ibuffer, materials, sibuffer, swbuffer, bindXforms, restXforms, names, parents, name = read_usd_asset(stage, allow_partial_mesh_buffer)
    return viewer.create_asset(vbuffer, ibuffer, materials, sibuffer, swbuffer, bindXforms, restXforms, names, parents, name)

    