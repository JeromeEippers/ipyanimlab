import ipywebgl
import numpy as np
from .material import Material
from .mesh import Mesh

class MeshRender:
    MAX_MATRIX_COUNT = 256

    def __init__(self, viewer:ipywebgl.GLViewer):
        self.static_prog = viewer.create_program_ext(
            self.static_vertex_shader,
            self.fragment_shader,
            {
                'in_vert' : 0,
                'in_normal' : 1,
                'in_vertColor' :2,
                'in_world' :7,
            },
            auto_execute=False
        )
        self.skinned_prog = viewer.create_program_ext(
            self.skinned_vertex_shader,
            self.fragment_shader,
            {
                'in_vert' : 0,
                'in_normal' : 1,
                'in_vertColor' :2,
                'in_skinIndices' : 3,
                'in_skinWeights' : 5
            },
            auto_execute=False
        )
        self.skinned2_prog = viewer.create_program_ext(
            self.skinned2_vertex_shader,
            self.fragment_shader,
            {
                'in_vert' : 0,
                'in_normal' : 1,
                'in_vertColor' :2,
                'in_skinIndices' : 3,
                'in_skinWeights' : 5,
                'in_skinIndicesB' : 4,
                'in_skinWeightsB' : 6
            },
            auto_execute=False
        )
        self.matrix_buffer = viewer.create_buffer_ext(
            'ARRAY_BUFFER',
            np.eye(4, dtype=np.float32)[np.newaxis,...].repeat(self.MAX_MATRIX_COUNT, axis=0),
            'DYNAMIC_DRAW',
            auto_execute=True
        )

    def render_mesh(self, viewer:ipywebgl.GLViewer, mesh:Mesh, xfroms, materials):
        
        if mesh.skin_type == 0:
            viewer.bind_buffer('ARRAY_BUFFER', self.matrix_buffer)
            viewer.buffer_sub_data('ARRAY_BUFFER', 0, xfroms, 'DYNAMIC_DRAW')
            viewer.bind_buffer('ARRAY_BUFFER', None)

            viewer.use_program(self.static_prog)
            viewer.bind_vertex_array(mesh.vao)
            
            for material in materials:
                viewer.uniform('u_color', material.albedo)
                viewer.uniform('u_material', material.material_uniform)
                viewer.draw_elements_instanced('TRIANGLES', material.index_count, 'UNSIGNED_SHORT', material.first_index*2, xfroms.shape[0])
        else:
            if mesh.skin_type == 1:
                viewer.use_program(self.skinned_prog)
            else:
                viewer.use_program(self.skinned2_prog)
            viewer.bind_vertex_array(mesh.vao)
            viewer.uniform_matrix('u_bones[0]', xfroms)

            for material in materials:
                viewer.uniform('u_color', material.albedo)
                viewer.uniform('u_material', material.material_uniform)
                viewer.draw_elements('TRIANGLES', material.index_count, 'UNSIGNED_SHORT', material.first_index*2)

        



MeshRender.static_vertex_shader = """#version 300 es
//the ViewBlock that is automatically filled by ipywebgl
layout(std140) uniform ViewBlock
{
    mat4 u_cameraMatrix;          //the camera matrix in world space
    mat4 u_viewMatrix;            //the inverse of the camera matrix
    mat4 u_projectionMatrix;      //the projection matrix
    mat4 u_viewProjectionMatrix;  //the projection * view matrix
};

in mat4 in_world;

in vec3 in_vert;
in vec3 in_normal;
in vec4 in_vertColor;

out vec4 v_viewposition;
out vec4 v_viewnormal;
out vec4 v_vertcolor;

void main() {
    vec4 pos = vec4(in_vert, 1.0) * in_world;
    vec4 normal = vec4(in_normal, 0.0) * in_world;
    
    v_viewposition = pos * u_viewMatrix;
    v_viewnormal = normal * u_viewMatrix;
    gl_Position = v_viewposition * u_projectionMatrix;

    v_vertcolor = in_vertColor;
  }
"""

MeshRender.skinned_vertex_base = """#version 300 es
//the ViewBlock that is automatically filled by ipywebgl
layout(std140) uniform ViewBlock
{
    mat4 u_cameraMatrix;          //the camera matrix in world space
    mat4 u_viewMatrix;            //the inverse of the camera matrix
    mat4 u_projectionMatrix;      //the projection matrix
    mat4 u_viewProjectionMatrix;  //the projection * view matrix
};

uniform mat4 u_bones["""+str(MeshRender.MAX_MATRIX_COUNT)+"""];

in vec3 in_vert;
in vec3 in_normal;
in vec4 in_vertColor;

out vec4 v_viewposition;
out vec4 v_viewnormal;
out vec4 v_vertcolor;

"""

MeshRender.skinned_vertex_end = """

void main() {
    vec4 vert = vec4(in_vert , 1.0);
    vec4 norm = vec4(in_normal , 0.0);
    vec4 accumulated = skinned(vert);
    vec4 accumulatedN = skinned(norm);

    accumulated.w = 1.0;
    accumulatedN.w = 0.0;
    
    v_viewposition = accumulated * u_viewMatrix;
    v_viewnormal = accumulatedN * u_viewMatrix;
    gl_Position = v_viewposition * u_projectionMatrix;

    v_vertcolor = in_vertColor;
  }
"""

MeshRender.skinned_vertex_shader = MeshRender.skinned_vertex_base + """
in vec4 in_skinIndices;
in vec4 in_skinWeights;

vec4 skinned (vec4 vector){
    vec4 accumulated = vec4(0,0,0,0);
    accumulated += (vector * u_bones[int(in_skinIndices.x)]) * in_skinWeights.x;
    accumulated += (vector * u_bones[int(in_skinIndices.y)]) * in_skinWeights.y;
    accumulated += (vector * u_bones[int(in_skinIndices.z)]) * in_skinWeights.z;
    accumulated += (vector * u_bones[int(in_skinIndices.w)]) * in_skinWeights.w;
    return accumulated;
}
""" + MeshRender.skinned_vertex_end

MeshRender.skinned2_vertex_shader = MeshRender.skinned_vertex_base + """
in vec4 in_skinIndices;
in vec4 in_skinWeights;
in vec4 in_skinIndicesB;
in vec4 in_skinWeightsB;

vec4 skinned (vec4 vector){
    vec4 accumulated = vec4(0,0,0,0);
    accumulated += (vector * u_bones[int(in_skinIndices.x)]) * in_skinWeights.x;
    accumulated += (vector * u_bones[int(in_skinIndices.y)]) * in_skinWeights.y;
    accumulated += (vector * u_bones[int(in_skinIndices.z)]) * in_skinWeights.z;
    accumulated += (vector * u_bones[int(in_skinIndices.w)]) * in_skinWeights.w;
    accumulated += (vector * u_bones[int(in_skinIndicesB.x)]) * in_skinWeightsB.x;
    accumulated += (vector * u_bones[int(in_skinIndicesB.y)]) * in_skinWeightsB.y;
    accumulated += (vector * u_bones[int(in_skinIndicesB.z)]) * in_skinWeightsB.z;
    accumulated += (vector * u_bones[int(in_skinIndicesB.w)]) * in_skinWeightsB.w;
    return accumulated;
}
""" + MeshRender.skinned_vertex_end

MeshRender.fragment_shader = """#version 300 es
precision highp float;

uniform vec3 u_color;
uniform vec4 u_material;
in vec4 v_viewposition;
in vec4 v_viewnormal;
in vec4 v_vertcolor;


layout(location=0) out vec4 color;
layout(location=1) out vec4 material;
layout(location=2) out vec4 viewPosition;
layout(location=3) out vec4 viewNormal;

void main() {
    color = vec4(u_color * v_vertcolor.rgb, v_vertcolor.a);
    material = u_material;
    viewPosition = vec4(v_viewposition.xyz, 1.0);
    viewNormal = vec4(normalize(v_viewnormal.xyz), 0.0);
}
"""