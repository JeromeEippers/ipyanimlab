import numpy as np

from .reading_buffers import VertexPrimBuffer

def read_usd_mesh_material_part(stage, face_vertex_count, face_vertex_first_index_indirection, face_mat_indices, material_path, prim_buffer, points, normals, colors, aos, skinning_indices, skinning_weights):
    # read material
    mat_albedo = np.ones([3], dtype=np.float32)
    mat_metallic = 0.0
    mat_roughness = 0.0
    mat_reflectance = 0.0
    mat_emissive = 0

    if stage.GetPrimAtPath(material_path) and stage.GetPrimAtPath(material_path).GetChildren():
        mat = stage.GetPrimAtPath(material_path).GetChildren()[0]

        if mat.HasAttribute('inputs:diffuseColor'):
            mat_albedo = np.array(mat.GetAttribute('inputs:diffuseColor').Get(), dtype=np.float32)
        if mat.HasAttribute('inputs:metallic'):
            mat_metallic = mat.GetAttribute('inputs:metallic').Get()
        if mat.HasAttribute('inputs:roughness'):
            mat_roughness = mat.GetAttribute('inputs:roughness').Get()
        if mat.HasAttribute('inputs:clearcoat'):
            mat_reflectance = mat.GetAttribute('inputs:clearcoat').Get()
        if mat.HasAttribute('inputs:emissiveColor'):
            mat_emissive = mat.GetAttribute('inputs:emissiveColor').Get()[0]

    # create a new buffer for this material
    prim_buffer.begin_material(
        str(material_path), 
        mat_albedo,
        mat_roughness, 
        mat_metallic, 
        mat_reflectance, 
        mat_emissive)

    # get all the faces for this material
    for face_id in face_mat_indices:
        num_of_vertices = face_vertex_count[face_id]
        vertices = [
            VertexPrimBuffer(
                points.get(face_id, i),
                normals.get(face_id, i),
                colors.get(face_id, i),
                aos.get(face_id, i),
                skinning_indices.get(face_id, i),
                skinning_weights.get(face_id, i)
            )
            for i in range(num_of_vertices)
        ] 

        # retriangulate and add in the buffer
        for j in range(2, num_of_vertices):
            prim_buffer.add_vertex(vertices[0])
            prim_buffer.add_vertex(vertices[j-1])
            prim_buffer.add_vertex(vertices[j])

    # end material
    prim_buffer.end_material()
