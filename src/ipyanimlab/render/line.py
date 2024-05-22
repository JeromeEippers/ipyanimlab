import ipywebgl
import numpy as np

class Line:

    MAX_POINT_COUNT = 512

    def __init__(self, viewer:ipywebgl.GLViewer):
        self.prog = viewer.create_program_ext(
        '''#version 300 es
            //the ViewBlock that is automatically filled by ipywebgl
            layout(std140) uniform ViewBlock
            {
                mat4 u_cameraMatrix;          //the camera matrix in world space
                mat4 u_viewMatrix;            //the inverse of the camera matrix
                mat4 u_projectionMatrix;      //the projection matrix
                mat4 u_viewProjectionMatrix;  //the projection * view matrix
            };

            uniform float u_scale;
            in vec3 in_a;
            in vec3 in_b;  
            in float in_vert;   
            void main() {
                gl_Position = vec4(mix(in_a, in_b, vec3(in_vert)), 1.0) * u_viewProjectionMatrix;
            }
        ''',
        '''#version 300 es
            precision highp float;
            uniform vec3 u_color;
            out vec4 f_color;
            void main() {
                f_color = vec4(u_color, 1.0);
            }
        ''',
            auto_execute=False
        )

        self.vbo = viewer.create_buffer_ext(
            src_data= np.array([
                    0, 1
                ], dtype=np.float32),
            auto_execute=False
        )

        self.pos_vbo = viewer.create_buffer_ext(
            'ARRAY_BUFFER',
            np.zeros([self.MAX_POINT_COUNT, 3], dtype=np.float32),
            'DYNAMIC_DRAW',
            auto_execute=False
        )

        self.vao = viewer.create_vertex_array_ext(
            self.prog,
            [
                (self.vbo, '1f32', 'in_vert'),
                (self.pos_vbo, '3f32:1 3f32:1', 'in_a', 'in_b'),
            ],
            auto_execute=False
        )

        viewer.execute_commands(execute_once=True)


    def show(self, viewer:ipywebgl.GLViewer, positions, color):

        viewer.bind_buffer('ARRAY_BUFFER', self.pos_vbo)
        viewer.buffer_sub_data('ARRAY_BUFFER', 0, positions, 'DYNAMIC_DRAW')
        viewer.bind_buffer('ARRAY_BUFFER', None)

        viewer.use_program(self.prog)
        viewer.uniform('u_color', color)
        viewer.bind_vertex_array(self.vao)
        viewer.draw_arrays_instanced('LINES', 0, 2, positions.shape[0]/2)

