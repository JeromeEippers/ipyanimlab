import numpy as np
from pxr import Usd, UsdGeom

from .reading_buffers import GeoPrimBuffer
from .reading_mesh import read_usd_mesh


def read_usd_rigid_asset(stage, root):
    prim_buf = GeoPrimBuffer(root.GetName())
    read_usd_rigid_prim(prim_buf, stage, root)
    return prim_buf



def read_usd_rigid_prim(prim_buffer, stage, prim):
    if prim.IsA(UsdGeom.Xform):
        casted_prim = UsdGeom.Xform(prim)
        local_matrix = np.array(casted_prim.GetLocalTransformation(), dtype=np.float32).T
        prim_buffer.begin_part(local_matrix)

        for child in prim.GetChildren():
            read_usd_rigid_prim(prim_buffer, stage, child)

        prim_buffer.end_part()

    elif prim.IsA(UsdGeom.Mesh):
        casted_prim = UsdGeom.Mesh(prim)
        local_matrix = np.array(casted_prim.GetLocalTransformation(), dtype=np.float32).T
        prim_buffer.begin_part(local_matrix)
        
        read_usd_mesh(prim_buffer, stage, casted_prim)

        for child in prim.GetChildren():
            read_usd_rigid_prim(prim_buffer, stage, child)

        prim_buffer.end_part()

    else:
        for child in prim.GetChildren():
            read_usd_rigid_prim(prim_buffer, stage, child)
