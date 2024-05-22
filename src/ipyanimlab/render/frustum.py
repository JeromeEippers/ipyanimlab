import ipywebgl
import numpy as np

class Frustum:

    def __init__(self, viewer:ipywebgl.GLViewer):
        self.frustum_prog = viewer.create_program_ext(
            """#version 300 es

            //the ViewBlock that is automatically filled by ipywebgl
            layout(std140) uniform ViewBlock
            {
                mat4 u_cameraMatrix;          //the camera matrix in world space
                mat4 u_viewMatrix;            //the inverse of the camera matrix
                mat4 u_projectionMatrix;      //the projection matrix
                mat4 u_viewProjectionMatrix;  //the projection * view matrix
            };

            uniform mat4 u_viewproj_to_draw;

            in vec3 in_vert;
            void main() {
                vec4 world = vec4(in_vert, 1.0) * inverse(u_viewproj_to_draw);
                world = world/world.w;
                gl_Position = world * u_viewProjectionMatrix;
            }
            """
            ,
            """#version 300 es
            precision highp float;
            out vec4 f_color;
            void main() {
                f_color = vec4(0,0,0, 1);
            }
            """,
            auto_execute=False
        )

        self.frustum_vbo = viewer.create_buffer_ext(
            src_data=np.array(
            [-1, -1, -1,
            1, -1, -1,
            -1,  1, -1,
            1,  1, -1,
            -1, -1,  1,
            1, -1,  1,
            -1,  1,  1,
            1,  1,  1,], dtype=np.float32),
            auto_execute=False
        )
        self.frustum_indices = np.array(
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
                (self.frustum_vbo, '3f32', 'in_vert'),
            ],
            self.frustum_indices,
            auto_execute=False
        )

        viewer.execute_commands(execute_once=True)


    def show(self, viewer:ipywebgl.GLViewer, viewproj_matrix):

        viewer.disable(depth_test=True)
        viewer.use_program(self.frustum_prog)
        viewer.uniform_matrix('u_viewproj_to_draw', viewproj_matrix)
        viewer.bind_vertex_array(self.frustum_vao)
        viewer.draw_elements('LINES', self.frustum_indices.shape[0], 'UNSIGNED_BYTE', 0)
