import ipywebgl
import os.path
import pickle
import numpy as np
import itertools
from .material import Material

from .. import utils
class Mesh:
    MAX_MATRIX_COUNT = 256 # should match the renderer constant

    def __init__(self, viewer:ipywebgl.GLViewer, vertices, indices, boneids=None, weights=None, bindpose=None, initialpose=None, bone_names=None, bone_parents=None, name=None):

        self.bindpose = None
        self.initialpose = None
        self.bone_names = []
        self.bone_parents = None
        self.vertices = vertices
        self.indices = indices
        self.boneids = boneids
        self.weights = weights
        self.skin_type = 0
        self.name = name or 'NoName'

        if bindpose is not None:
            self.bindpose = bindpose
            self.initialpose = initialpose
            self.bone_names = bone_names
            self.bone_parents = bone_parents

        self.vbo = viewer.create_buffer_ext(
            src_data=vertices.astype(np.float32).flatten(),
            auto_execute=False)

        self.bone_id_vbo = None
        self.bone_weights_vbo = None

        if bindpose is not None:
            self.bone_id_vbo = viewer.create_buffer_ext(
                src_data=boneids.astype(np.int32).flatten(),
                auto_execute=False)
            self.bone_weights_vbo = viewer.create_buffer_ext(
                src_data=weights.astype(np.float32).flatten(),
                auto_execute=False)

        self.indices = indices.astype(np.uint16).flatten()

        mapping = [(self.vbo, '3f32 3f32 4f32', 0, 1, 2)]
        if bindpose is not None:
            if vertices.shape[0]//10 < boneids.shape[0]//4:
                self.skin_type = 2
                mapping += [(self.bone_id_vbo, '4i32 4i32', 3, 4)]
                mapping += [(self.bone_weights_vbo, '4f32 4f32', 5, 6)]
            else:
                self.skin_type = 1
                mapping += [(self.bone_id_vbo, '4i32', 3)]
                mapping += [(self.bone_weights_vbo, '4f32', 5)]
        else :
            mapping += [(viewer._mesh_render.matrix_buffer, '1mat4:1', 7),]

        self.vao = viewer.create_vertex_array_ext(
            None,
            mapping,
            self.indices,
            auto_execute=False
        )

        viewer.execute_commands(execute_once=True)


    def bone_count(self):
        return len(self.bone_names)


    def bone_index(self, name):
        return self.bone_names.index(name)


    def add_bone(self, name, q, p, parent_name=None, global_space=True):
        if name in self.bone_names:
            raise Exception(name + " already in skeleton")

        m = utils.quat_to_mat(q, p)
        parent_id = -1
        if parent_name in self.bone_names:
            parent_id = self.bone_names.index(parent_name)
            
        if parent_id > -1 and global_space:
            self.global_bones()
            m = np.dot(np.linalg.inv(self.scratch_buffer[parent_id, :, :]), m)

        self.bone_names.append(name)
        self.bone_names.append(name)
        self.bone_parents = np.append(self.bone_parents, parent_id)
        initial = np.zeros([self.bone_count(), 4, 4], dtype=np.float32)
        initial[:self.bone_count()-1,:, :] = self.initialpose
        initial[-1, :, :] = m
        self.initialpose = initial
    
