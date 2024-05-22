import ipywebgl
import numpy as np

class ShowTexture:

    def __init__(self, viewer:ipywebgl.GLViewer):
        self.show_texture_progs = [
            viewer.create_program_ext(
                self.vertex_shader,
                ShowTexture.fragment_default,
                {
                    'in_vert' : 0,
                    'in_coord' : 1,
                },
                auto_execute=False),
            viewer.create_program_ext(
                self.vertex_shader,
                ShowTexture.fragment_r,
                {
                    'in_vert' : 0,
                    'in_coord' : 1,
                },
                auto_execute=False),
            viewer.create_program_ext(
                self.vertex_shader,
                ShowTexture.fragment_a,
                {
                    'in_vert' : 0,
                    'in_coord' : 1,
                },
                auto_execute=True)
            ]

    def show(self, viewer:ipywebgl.GLViewer, screen_vao, texture_id:int, lod_id:int=0, shader_type:int=0):
        viewer.bind_framebuffer('FRAMEBUFFER', None)
        viewer.clear()
        viewer.disable(depth_test=True)
        viewer.use_program(self.show_texture_progs[shader_type])
        viewer.uniform('u_texture', np.array([texture_id], dtype=np.int32))
        viewer.uniform('u_miplevel', np.array([lod_id], dtype=np.float32))
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)



ShowTexture.vertex_shader = """#version 300 es
in vec2 in_vert;
in vec2 in_coord;
out vec2 v_coord;

void main() {
    gl_Position = vec4(in_vert, 0, 1);
    v_coord = in_coord;
}
"""

ShowTexture.fragment_default = """#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform float u_miplevel;
in vec2 v_coord;
out vec4 color;

void main() {
    color = vec4(textureLod(u_texture, v_coord, u_miplevel).rgb, 1);
}
"""

ShowTexture.fragment_r = """#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform float u_miplevel;
in vec2 v_coord;
out vec4 color;

void main() {
    color = vec4(textureLod(u_texture, v_coord, u_miplevel).rrr, 1);
}
"""

ShowTexture.fragment_a = """#version 300 es
precision highp float;

uniform sampler2D u_texture;
uniform float u_miplevel;
in vec2 v_coord;
out vec4 color;

void main() {
    color = vec4(textureLod(u_texture, v_coord, u_miplevel).aaa, 1);
}
"""