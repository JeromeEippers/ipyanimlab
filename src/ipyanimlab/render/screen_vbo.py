import ipywebgl
import numpy as np

def create_screen_vao(viewer:ipywebgl.GLViewer):

    screen_vbo = viewer.create_buffer_ext(
        #x y u v
        src_data=np.array(
        [-1, 1, 0, 1,
            -1, -1, 0, 0,
            1, -1, 1, 0,
            -1, 1, 0, 1,
            1, -1, 1, 0,
            1, 1, 1, 1], dtype=np.float32).flatten(),
        auto_execute=False    
    )

    screen_vao = viewer.create_vertex_array_ext(
        None,
        [
            (screen_vbo, '2f32 2f32', 0, 1),
        ],
        auto_execute=False    
    )

    viewer.execute_commands(execute_once=True)
    return screen_vao