from pxr import Usd, UsdGeom, Gf, UsdSkel, Sdf
import numpy as np

from .skeleton_prim import SkeletonPrim
from .reading_mesh import read_usd_mesh
from .reading_buffers import GeoPrimBuffer

def read_usd_skinned_asset(stage, root):
    skelCache = UsdSkel.Cache()
    skelRoot = UsdSkel.Root(root)
    skelCache.Populate(skelRoot, Usd.TraverseInstanceProxies())

    bindings = skelCache.ComputeSkelBindings(
        skelRoot, Usd.TraverseInstanceProxies())
    
    if len(bindings) != 1:
        raise Execption('we only support skin asset with one binding')
    binding = bindings[0]
    
    # Get the Skeleton for this binding.
    skelQuery = skelCache.GetSkelQuery(binding.GetSkeleton())
    
    names = []
    parents = []
    for i, jointToken in enumerate(skelQuery.GetJointOrder()):
        jointPath = Sdf.Path(jointToken)
        names.append(jointPath.name)
        parents.append(skelQuery.GetTopology().GetParent(i))

    bind_transforms = np.array(skelQuery.GetJointWorldBindTransforms(), dtype=np.float32).reshape([-1,4,4])
    bind_transforms = np.einsum('ijk->ikj', bind_transforms) # we are column major so we need the transpose of each matrices
    # the binding transforms for us are the inverse matrix ( so we don't need to inverse when computing the final matrices )
    for i in range (bind_transforms.shape[0]):
        bind_transforms[i] = np.linalg.inv(bind_transforms[i])
    
    rest_transforms = np.array(skelQuery.ComputeJointLocalTransforms(), dtype=np.float32).reshape([-1,4,4])
    rest_transforms = np.einsum('ijk->ikj', rest_transforms) # we are column major so we need the transpose of each matrices
    
    skeleton = SkeletonPrim(names, np.array(parents, dtype=np.int32), bind_transforms, rest_transforms)
    prim_buf = GeoPrimBuffer(root.GetName(), skeleton=skeleton)

    # Iterate over the prims that are skinned by this Skeleton.   
    for skinningQuery in binding.GetSkinningTargets():
        primToSkin = skinningQuery.GetPrim()
        
        #find all the parents so we can build the transforms properly
        prim_parents = [primToSkin]
        current_prim = primToSkin
        while current_prim.GetParent().IsValid() :
            current_prim = current_prim.GetParent()
            prim_parents.append(current_prim)
               
        for pp in prim_parents:
            if pp.IsA(UsdGeom.Xform):
                prim_buf.begin_part(np.array(UsdGeom.Xform(pp).GetLocalTransformation(), dtype=np.float32).T)
            elif pp.IsA(UsdGeom.Mesh):
                prim_buf.begin_part(np.array(UsdGeom.Mesh(pp).GetLocalTransformation(), dtype=np.float32).T)
               
        read_usd_mesh(prim_buf, stage, UsdGeom.Mesh(primToSkin), skelQuery, skinningQuery)
               
        for pp in prim_parents:
            if pp.IsA(UsdGeom.Xform) or pp.IsA(UsdGeom.Mesh):
                prim_buf.end_part()
        
    return prim_buf