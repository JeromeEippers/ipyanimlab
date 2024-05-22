import ipywebgl
import numpy as np

class Axis:

    MAX_AXIS_COUNT = 256

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
            in vec3 in_vert;
            in vec3 in_color;
            in mat4 in_world;
            out vec3 v_color;    
            void main() {
                gl_Position = vec4(in_vert * u_scale, 1.0) * in_world * u_viewProjectionMatrix;
                v_color = in_color;
            }
        ''',
        '''#version 300 es
            precision highp float;
            in vec3 v_color;
            out vec4 f_color;
            void main() {
                f_color = vec4(v_color, 1.0);
            }
        ''',
            {
                'in_world' : 0,
                'in_vert' : 4,
                'in_color' : 5,
            },
            auto_execute=False
        )

        self.vbo = viewer.create_buffer_ext(
            src_data= np.array([
                    # x, y ,z red, green, blue
                    0, 0, 0, 1, 0, 0,
                    1, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 1, 0,
                    0, 1, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 1,
                    0, 0, 1, 0, 0, 1,
                ], dtype=np.float32),
            auto_execute=False
        )

        self.mat_vbo = viewer.create_buffer_ext(
            'ARRAY_BUFFER',
            np.eye(4, dtype=np.float32)[np.newaxis,...].repeat(self.MAX_AXIS_COUNT, axis=0),
            'DYNAMIC_DRAW',
            auto_execute=False
        )

        self.vao = viewer.create_vertex_array_ext(
            None,
            [
                (self.vbo, '3f32 3f32', 4, 5),
                (self.mat_vbo, '1mat4:1', 0),
            ],
            auto_execute=False
        )

        viewer.execute_commands(execute_once=True)


    def show(self, viewer:ipywebgl.GLViewer, matrices, scale):

        viewer.bind_buffer('ARRAY_BUFFER', self.mat_vbo)
        viewer.buffer_sub_data('ARRAY_BUFFER', 0, matrices, 'DYNAMIC_DRAW')
        viewer.bind_buffer('ARRAY_BUFFER', None)

        viewer.use_program(self.prog)
        viewer.uniform('u_scale', np.array([scale], dtype=np.float32))
        viewer.bind_vertex_array(self.vao)
        viewer.draw_arrays_instanced('LINES', 0, 6, matrices.shape[0])

