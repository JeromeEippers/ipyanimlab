import ipywebgl
import numpy as np
from .material import Material
from .mesh import Mesh

class Background:
    def __init__(self, viewer:ipywebgl.GLViewer):
        self.background_prog = viewer.create_program_ext(
            self.backbround_vertex_shader,
            self.background_pixel_shader,
            {
                'in_vert' : 0,
            },
            auto_execute=True
        )

    def render(self, viewer:ipywebgl.GLViewer, screen_vao):
        viewer.use_program(self.background_prog)
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)


Background.backbround_vertex_shader = """#version 300 es

//the ViewBlock that is automatically filled by ipywebgl
layout(std140) uniform ViewBlock
{
    mat4 u_cameraMatrix;          //the camera matrix in world space
    mat4 u_viewMatrix;            //the inverse of the camera matrix
    mat4 u_projectionMatrix;      //the projection matrix
    mat4 u_viewProjectionMatrix;  //the projection * view matrix
};


in vec2 in_vert;

out vec4 v_viewposition;
out vec4 v_viewnormal;

void main() {
    vec4 clip_pos = vec4(in_vert, .9999, 1);

    vec4 viewPos =  clip_pos * inverse(u_projectionMatrix);
    viewPos /= viewPos.w;
    
    v_viewposition = viewPos;
    v_viewnormal = vec4(normalize(-viewPos.xyz), 0);
    gl_Position = clip_pos;
  }
"""

Background.background_pixel_shader = """#version 300 es
precision highp float;

uniform vec3 u_color;
in vec4 v_viewposition;
in vec4 v_viewnormal;

layout(location=0) out vec4 color;
layout(location=1) out vec4 material;
layout(location=2) out vec4 viewPosition;
layout(location=3) out vec4 viewNormal;

void main() {
    color = vec4(1);
    material = vec4(.10, .08, .4, 0);
    viewPosition = vec4(-v_viewposition.xyz, 0.);
    viewNormal = -vec4(normalize(v_viewnormal.xyz), 0);
}
"""