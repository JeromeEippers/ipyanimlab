import ipywebgl
import numpy as np

class Ground:

    def __init__(self, viewer:ipywebgl.GLViewer):
        self.ground_prog = viewer.create_program_ext(
            """#version 300 es
            //the ViewBlock that is automatically filled by ipywebgl
            layout(std140) uniform ViewBlock
            {
                mat4 u_cameraMatrix;          //the camera matrix in world space
                mat4 u_viewMatrix;            //the inverse of the camera matrix
                mat4 u_projectionMatrix;      //the projection matrix
                mat4 u_viewProjectionMatrix;  //the projection * view matrix
            };


            in vec3 in_vert;
            in vec2 in_texcoord;
            out vec2 v_texcoord;

            out vec4 v_viewposition;
            out vec4 v_viewnormal;

            void main() {
                v_texcoord = in_texcoord;
                vec4 pos = vec4(in_vert * 1000.0, 1.0) ;
                vec4 normal = vec4(.0, 1., .0, 0.0) ;
                
                v_viewposition = pos * u_viewMatrix;
                v_viewnormal = normal * u_viewMatrix;
                gl_Position = v_viewposition * u_projectionMatrix;
            }
            """
            ,
            """#version 300 es
            precision highp float;
            uniform sampler2D u_texture;
            in vec2 v_texcoord;

            in vec4 v_viewposition;
            in vec4 v_viewnormal;

            layout(location=0) out vec4 color;
            layout(location=1) out vec4 material;
            layout(location=2) out vec4 viewPosition;
            layout(location=3) out vec4 viewNormal;

            void main() {
                float t = texture(u_texture, v_texcoord).r;

                color = mix(vec4(.4, .4, .4, 1.), vec4(.5, .5, .5, 1.), vec4(t));
                material = mix(vec4(.55, 0, .25, 0.), vec4(.5, 0, .3, 0.), vec4(t));
                viewPosition = v_viewposition;
                viewNormal = vec4(normalize(v_viewnormal.xyz), 0);
            }
            """,
            {'in_vert' : 0, 'in_texcoord':1},
            auto_execute=False
        )

        self.ground_vbo = viewer.create_buffer_ext(
            #x y z u v
            src_data=np.array([
                -1, 0, -1, 0, 10,
                1, 0, -1, 10, 10,
                1, 0, 1, 10, 0,
                -1, 0,1, 0, 0,
            ], dtype=np.float32),
            auto_execute=False
        )
        self.ground_vao = viewer.create_vertex_array_ext(
            None,
            [
                (self.ground_vbo, '3f32 2f32', 0, 1),
            ],
            auto_execute=False
            )

        self.ground_texture = viewer.create_texture()
        viewer.active_texture(15)
        viewer.bind_texture('TEXTURE_2D', self.ground_texture)
        viewer.tex_image_2d('TEXTURE_2D', 0, 'RGBA', 2, 2, 0, 'RGBA', 'UNSIGNED_BYTE',
                    np.array([255,255,255,255, 0,0,0,255, 0,0,0,255, 255,255,255,255], dtype=np.uint8))
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'REPEAT')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'REPEAT')

        viewer.execute_commands(execute_once=True)


    def show(self, viewer:ipywebgl.GLViewer):
        viewer.use_program(self.ground_prog)
        viewer.uniform('u_texture', np.array([15], dtype=np.int32))
        self.draw_arrays(viewer)


    def draw_arrays(self, viewer:ipywebgl.GLViewer):
        viewer.bind_vertex_array(self.ground_vao)
        viewer.draw_arrays('TRIANGLE_FAN', 0, 4)