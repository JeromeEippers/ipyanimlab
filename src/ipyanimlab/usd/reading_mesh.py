from pxr import Usd, UsdGeom, Gf, UsdSkel, Sdf
import numpy as np

from .reading_accessors import ConstantAccessor, FaceVaryingAccessor, VertexAccessor, VertexIndexedAccessor, FaceUniformAccessor
from .material_prim import read_usd_mesh_material_part


def read_usd_mesh(prim_buffer, stage, prim, skel_query=None, skin_query=None):

    face_vertex_count = np.array(prim.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    face_vertex_indices = np.array(prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    face_vertex_first_index_indirection = np.concatenate([[0], np.cumsum(face_vertex_count)[:-1]])
    
    # get the vertex
    points = VertexAccessor(face_vertex_indices, face_vertex_first_index_indirection, np.array(prim.GetPointsAttr().Get(), dtype=np.float32))
    
    # get the normals
    normals = None
    if prim.GetNormalsAttr().GetMetadata('interpolation') == 'faceVarying':
        normals = FaceVaryingAccessor(face_vertex_first_index_indirection, np.array(prim.GetNormalsAttr().Get(), dtype=np.float32))
    elif prim.GetNormalsAttr().GetMetadata('interpolation') == 'vertex':
        normals = VertexAccessor(face_vertex_indices, face_vertex_first_index_indirection, np.array(prim.GetNormalsAttr().Get(), dtype=np.float32))
    else:
        nrms = np.zeros([face_vertex_indices.shape[0], 3], dtype=np.float32)
        for face_id in range(face_vertex_count.shape[0]):
            first_index = face_vertex_first_index_indirection[face_id]
            nrms[first_index:first_index+face_vertex_count[face_id]] = np.cross(points.get(face_id,2) - points.get(face_id,0), points.get(face_id,1) - points.get(face_id,0))[np.newaxis,:].repeat(face_vertex_count[face_id], axis=0)
        normals = FaceVaryingAccessor(face_vertex_first_index_indirection, ardlab.utils.normalize(nrms))
        
    # get vertex color
    colors = None
    if prim.GetDisplayColorAttr().GetMetadata('interpolation') == 'faceVarying':
        colors = FaceVaryingAccessor(face_vertex_first_index_indirection, np.array(prim.GetDisplayColorPrimvar().ComputeFlattened(), dtype=np.float32))
    elif prim.GetDisplayColorAttr().GetMetadata('interpolation') == 'vertex':
        colors = VertexAccessor(face_vertex_indices, face_vertex_first_index_indirection, np.array(prim.GetDisplayColorPrimvar().ComputeFlattened(), dtype=np.float32))
    elif prim.GetDisplayColorAttr().GetMetadata('interpolation') == 'uniform':
        colors = FaceUniformAccessor(np.array(prim.GetDisplayColorPrimvar().ComputeFlattened(), dtype=np.float32))
    else:
        colors = ConstantAccessor(np.ones(3, dtype=np.float32))

    # get ao
    aos = None
    if prim.GetDisplayOpacityAttr().GetMetadata('interpolation') == 'faceVarying':
        colors = FaceVaryingAccessor(face_vertex_first_index_indirection, np.array(prim.GetDisplayOpacityPrimvar().ComputeFlattened(), dtype=np.float32))
    elif prim.GetDisplayOpacityAttr().GetMetadata('interpolation') == 'vertex':
        aos = VertexAccessor(face_vertex_indices, face_vertex_first_index_indirection, np.array(prim.GetDisplayOpacityPrimvar().ComputeFlattened(), dtype=np.float32))
    elif prim.GetDisplayOpacityAttr().GetMetadata('interpolation') == 'uniform':
        aos= FaceUniformAccessor(np.array(prim.GetDisplayOpacityPrimvar().ComputeFlattened(), dtype=np.float32))
    else:
        aos = ConstantAccessor(1.0)
    
    # get skinning
    skinning_indices = ConstantAccessor(None)
    skinning_weights = ConstantAccessor(None)
    
    if skin_query is not None and skel_query is not None:
        influences = skin_query.ComputeJointInfluences()
        if influences:
            jointIndices, jointWeights = influences
            skeleton_joint_order = list(skel_query.GetJointOrder())
            remap = {i:skeleton_joint_order.index(b) for i,b in enumerate(skin_query.GetJointOrder())}
            
            numInfluencesPerComponent = skin_query.GetNumInfluencesPerComponent()
            if numInfluencesPerComponent > 4:
                UsdSkel.ResizeInfluences(jointIndices, numInfluencesPerComponent, 8)
                UsdSkel.ResizeInfluences(jointWeights, numInfluencesPerComponent, 8)
                numInfluencesPerComponent = 8
            else:
                UsdSkel.ResizeInfluences(jointIndices, numInfluencesPerComponent, 4)
                UsdSkel.ResizeInfluences(jointWeights, numInfluencesPerComponent, 4)
                numInfluencesPerComponent = 4
            skinning_indices = VertexAccessor(
                face_vertex_indices, 
                face_vertex_first_index_indirection, 
                np.array([remap[ji] for ji in jointIndices], dtype=np.int32).reshape(-1, numInfluencesPerComponent))
            skinning_weights = VertexAccessor(
                face_vertex_indices, 
                face_vertex_first_index_indirection, 
                np.array(jointWeights, dtype=np.float32).reshape(-1, numInfluencesPerComponent))
    
    # read mesh per material (as we regroup the indices for each materials)
    found_material = False
    if prim.GetPrim().HasRelationship('material:binding'):
        found_material = True
        face_mat_indices = range(face_vertex_count.shape[0])
        material_path = prim.GetPrim().GetRelationship('material:binding').GetTargets()[0]
        read_usd_mesh_material_part(stage, face_vertex_count, face_vertex_first_index_indirection, face_mat_indices, material_path, prim_buffer, points, normals, colors, aos, skinning_indices, skinning_weights)
    else:
        for geoSubset in prim.GetPrim().GetChildren():
            if geoSubset.GetTypeName() == 'GeomSubset' and geoSubset.GetAttribute('familyName').Get() == 'materialBind':
                found_material = True
                face_mat_indices = geoSubset.GetAttribute('indices').Get()
                material_path = geoSubset.GetRelationship('material:binding').GetTargets()[0]
                read_usd_mesh_material_part(stage, face_vertex_count, face_vertex_first_index_indirection, face_mat_indices, material_path, prim_buffer, points, normals, colors, aos, skinning_indices, skinning_weights)
    if not found_material:
        face_mat_indices = range(face_vertex_count.shape[0])
        material_path = 'DefaultMaterial'
        read_usd_mesh_material_part(stage, face_vertex_count, face_vertex_first_index_indirection, face_mat_indices, material_path, prim_buffer, points, normals, colors, aos, skinning_indices, skinning_weights)
                
