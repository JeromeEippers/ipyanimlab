import ipywebgl
import numpy as np
from .mesh import Mesh
from .mesh_render import MeshRender

class Shadow:
    def __init__(self, viewer:ipywebgl.GLViewer, resolution=1024):
        self._resolution = resolution
        self.static_shadow_prog = viewer.create_program_ext(
            self.static_vertex_shader,
            self.fragment_shader,
            {'in_vert' : 0,
            'in_world' :7,},
            auto_execute=False
            )
        self.skinned_shadow_prog = viewer.create_program_ext(
            self.skinned_vertex_shader,
            self.fragment_shader,
            {'in_vert' : 0,
            'in_skinIndices' : 3,
            'in_skinWeights' : 5},
            auto_execute=False
            )
        self.skinned2_shadow_prog = viewer.create_program_ext(
            self.skinned2_vertex_shader,
            self.fragment_shader,
            {'in_vert' : 0,
            'in_skinIndices' : 3,
            'in_skinIndicesB' : 4,
            'in_skinWeights' : 5,
            'in_skinWeightsB' : 6},
            auto_execute=False
            )

        self.shadow_buffer = viewer.create_framebuffer()
        viewer.bind_framebuffer('FRAMEBUFFER', self.shadow_buffer)

        self.shadow_texture = viewer.create_texture()
        viewer.active_texture(11)
        viewer.bind_texture('TEXTURE_2D', self.shadow_texture)
        viewer.tex_image_2d('TEXTURE_2D', 0, 'DEPTH_COMPONENT32F', resolution, resolution, 0, 'DEPTH_COMPONENT', 'FLOAT', None)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')

        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'DEPTH_ATTACHMENT', 'TEXTURE_2D', self.shadow_texture, 0)
        viewer.bind_framebuffer('FRAMEBUFFER', None)

        self.frustum_prog = viewer.create_program_ext(
            self.frustum_vertex_shader,
            self.frustum_fragment_shader,
            auto_execute=False
            )

        frustum_vbo = viewer.create_buffer_ext(
            src_data=np.array(
            [-1, -1, -1,
            1, -1, -1,
            -1,  1, -1,
            1,  1, -1,
            -1, -1,  1,
            1, -1,  1,
            -1,  1,  1,
            1,  1,  1,], dtype=np.float32)
        )
        frustum_indices = np.array(
            [
            0, 1,
            1, 3,
            3, 2,
            2, 0,

            4, 5,
            5, 7,
            7, 6,
            6, 4,

            0, 4,
            1, 5,
            3, 7,
            2, 6,
            ], dtype=np.uint8).flatten()

        self.frustum_vao = viewer.create_vertex_array_ext(
            self.frustum_prog,
            [
                (frustum_vbo, '3f32', 'in_vert'),
            ],
            frustum_indices,
            auto_execute=False
        )

        viewer.execute_commands(execute_once=True)


    def activate_shadow(self, viewer:ipywebgl.GLViewer):
        viewer.bind_framebuffer('FRAMEBUFFER', self.shadow_buffer)
        viewer.enable(depth_test=True)
        viewer.viewport(0,0,self._resolution,self._resolution)
        viewer.clear()
        

    def render_mesh(self, viewer:ipywebgl.GLViewer, mesh:Mesh, light_ortho_projection, worlds):
        if mesh.skin_type == 0:
            viewer.bind_buffer('ARRAY_BUFFER', viewer._mesh_render.matrix_buffer)
            viewer.buffer_sub_data('ARRAY_BUFFER', 0, worlds, 'DYNAMIC_DRAW')
            viewer.bind_buffer('ARRAY_BUFFER', None)

            viewer.use_program(self.static_shadow_prog)
            viewer.uniform_matrix('u_lightProjection', light_ortho_projection)
            viewer.bind_vertex_array(mesh.vao)
            viewer.draw_elements_instanced('TRIANGLES', mesh.indices.shape[0], 'UNSIGNED_SHORT', 0, worlds.shape[0])
        else:
            if mesh.skin_type == 1:
                viewer.use_program(self.skinned_shadow_prog)
            else:
                viewer.use_program(self.skinned2_shadow_prog)
            viewer.uniform_matrix('u_lightProjection', light_ortho_projection)
            viewer.bind_vertex_array(mesh.vao)
            viewer.uniform_matrix('u_bones[0]', worlds)

            viewer.draw_elements('TRIANGLES', mesh.indices.shape[0], 'UNSIGNED_SHORT', 0)


    def render_frustum(self, viewer:ipywebgl.GLViewer, light_ortho_projection):
        viewer.use_program(self.frustum_prog)
        viewer.uniform_matrix('u_lightProjection', light_ortho_projection)
        viewer.bind_vertex_array(self.frustum_vao)
        viewer.draw_elements('LINES', 24, 'UNSIGNED_BYTE', 0)


Shadow.static_vertex_shader = """#version 300 es
uniform mat4 u_lightProjection;
in mat4 in_world;
in vec3 in_vert;

void main() {
    gl_Position = vec4(in_vert, 1.0) * in_world * u_lightProjection;
}
"""


Shadow.skinned_vertex_shader = """#version 300 es
uniform mat4 u_lightProjection;
uniform mat4 u_bones["""+str(MeshRender.MAX_MATRIX_COUNT)+"""];

in vec3 in_vert;
in vec4 in_skinIndices;
in vec4 in_skinWeights;


void main() {
    vec4 vert = vec4(in_vert , 1.0);
    vec4 accumulated = vec4(0,0,0,0);

    accumulated += (vert * u_bones[int(in_skinIndices.x)]) * in_skinWeights.x;
    accumulated += (vert * u_bones[int(in_skinIndices.y)]) * in_skinWeights.y;
    accumulated += (vert * u_bones[int(in_skinIndices.z)]) * in_skinWeights.z;
    accumulated += (vert * u_bones[int(in_skinIndices.w)]) * in_skinWeights.w;
    accumulated.w = 1.0;

    gl_Position = accumulated * u_lightProjection;
  }
"""

Shadow.skinned2_vertex_shader = """#version 300 es
uniform mat4 u_lightProjection;
uniform mat4 u_bones["""+str(MeshRender.MAX_MATRIX_COUNT)+"""];

in vec3 in_vert;
in vec4 in_skinIndices;
in vec4 in_skinWeights;
in vec4 in_skinIndicesB;
in vec4 in_skinWeightsB;


void main() {
    vec4 vert = vec4(in_vert , 1.0);
    vec4 accumulated = vec4(0,0,0,0);

    accumulated += (vert * u_bones[int(in_skinIndices.x)]) * in_skinWeights.x;
    accumulated += (vert * u_bones[int(in_skinIndices.y)]) * in_skinWeights.y;
    accumulated += (vert * u_bones[int(in_skinIndices.z)]) * in_skinWeights.z;
    accumulated += (vert * u_bones[int(in_skinIndices.w)]) * in_skinWeights.w;
    accumulated += (vert * u_bones[int(in_skinIndicesB.x)]) * in_skinWeightsB.x;
    accumulated += (vert * u_bones[int(in_skinIndicesB.y)]) * in_skinWeightsB.y;
    accumulated += (vert * u_bones[int(in_skinIndicesB.z)]) * in_skinWeightsB.z;
    accumulated += (vert * u_bones[int(in_skinIndicesB.w)]) * in_skinWeightsB.w;
    accumulated.w = 1.0;

    gl_Position = accumulated * u_lightProjection;
  }
"""

Shadow.fragment_shader = """#version 300 es
precision highp float;
out vec4 f_color;
void main() {
    f_color = vec4(1, 0.1, 0.1, 1.0);
}
"""

Shadow.frustum_vertex_shader = """#version 300 es

//the ViewBlock that is automatically filled by ipywebgl
layout(std140) uniform ViewBlock
{
    mat4 u_cameraMatrix;          //the camera matrix in world space
    mat4 u_viewMatrix;            //the inverse of the camera matrix
    mat4 u_projectionMatrix;      //the projection matrix
    mat4 u_viewProjectionMatrix;  //the projection * view matrix
};

uniform mat4 u_lightProjection;

in vec3 in_vert;
void main() {
    vec4 world = vec4(in_vert, 1.0) * inverse(u_lightProjection);
    world = world/world.w;
    gl_Position = world * u_viewProjectionMatrix;
}
"""

Shadow.frustum_fragment_shader = """#version 300 es
precision highp float;
out vec4 f_color;
void main() {
    f_color = vec4(0,0,0, 1);
}
"""
