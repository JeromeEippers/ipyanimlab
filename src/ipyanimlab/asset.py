import numpy as np
from copy import deepcopy
import itertools

from .render.material import Material
from . import utils

class Asset:
    """An asset in the viewer that can be rendered.
    An asset represent a mesh in webgl.
    Several asset can share the same mesh for a smaller memory footprint on the GPU
    """

    MAX_MATRIX_COUNT = 256 # should match the renderer constant
    _scratch_skeleton_ = np.zeros([256, 4, 4], dtype=np.float32) # 256
    _scratch_position_ = np.zeros([256*2, 3], dtype=np.float32) # 256

    def __init__(self, mesh, materials):
        """Initialize an asset

        Args:
            :mesh: (Mesh): the mesh to use when rendering
            :materials: (list of Material): the materials to use to do the rendering
        """
        self._mesh = mesh
        self._materials = materials
        self._xform = np.eye(4, dtype=np.float32)
        self._skeleton_xforms = None

        if materials is None:
            self._materials = [Material('full', np.ones((3), dtype=np.float32), index_count=self._mesh.indices.shape[0])]

        if self._mesh.skin_type > 0:
            self._skeleton_xforms = np.zeros_like(self._mesh.initialpose, dtype=np.float32)
            self._skeleton_xforms[:,:,:] = self._compute_skeleton_xforms()

    def duplicate(self):
        """Duplicate the asset
        This allows to have several asset using the same GPU buffers

        Returns:
            :Asset: the duplicated asset
        """
        return Asset(self._mesh, deepcopy(self._materials))

    def name(self):
        """get the name of the asset/mesh

        Returns:
            :str: the name
        """
        return self._mesh.name

    def is_rigid(self):
        """is this a rigid mesh or a skinned mesh with skeleton

        Returns:
            :bool: rigid or not
        """
        return self._mesh.skin_type == 0

    def reset(self):
        """Reset will reset the internal xform to the identity and the skeleton to the bindpose
        """
        self._xform = np.eye(4, dtype=np.float32)
        if self._mesh.skin_type > 0:
            self._skeleton_xforms[:,:,:] = self._compute_skeleton_xforms()

    def materials(self):
        """get the list of materials

        Returns:
            :list(Material): the list of materials
        """
        return self._materials

    def material_names(self):
        """get the list of material names

        Returns:
            :list(str): the list of names of the materials
        """
        return [m.name for m in self._materials]

    def material(self, name):
        """get one material by name

        Args:
            :name: (str): the name of the material

        Returns:
            :Material: the material
        """
        return next((m for m in self._materials if m.name == name),None)

    def replace_materials(self, materials=None):
        """replace the list of materials

        Args:
            :materials: (List(Material), optional): the list of material to use, or None to get a default material. Defaults to None.
        """
        self._materials = materials
        if materials is None:
            self._materials = [Material('full', np.ones((3), dtype=np.float32), index_count=self._mesh.indices.shape[0])]

    def xform(self):
        """get the transform of the asset

        Returns:
            :np.array([4,4], dtype=np.float32): the xform matrix
        """
        return self._xform

    def set_xform(self, xform):
        """set the xform

        Args:
            :xform: (np.array([4,4], dtype=np.float32)): the matrix of the xform
        """
        self._xform[:,:] = xform

    def set_skeleton(self, local_matrices=None, names=None):
        """set the internal skeleton pose

        Args:
            :local_matrices: (np.array([...,4,4], dtype=np.float32), optional): the list of matrices or none to get back to the bindpose. Defaults to None.
            :names: (list(str), optional): the list of names of each bone we sent, if none we assume that the matrices are matching the skeleton order and size. Defaults to None.
        """
        self._skeleton_xforms[:,:,:] = self._compute_skeleton_xforms(local_matrices, names)

    def bone_count(self):
        """get the bone count

        Returns:
            :int: the number of bones
        """
        return self._mesh.bone_count()

    def bone_names(self):
        """get the bone names

        Returns:
            :list(str): the bone names
        """
        return self._mesh.bone_names

    def bone_index(self, name):
        """get the index of a bone

        Args:
            :name: (str): the name of the bone

        Returns:
            :int: the index of that bone
        """
        return self._mesh.bone_index(name)

    def world_skeleton_xforms(self, local_matrices=None, names=None):
        """get the world skeleton transforms

        Args:
            :local_matrices: (np.array([...,4,4], dtype=np.float32), optional): the list of matrices or none to get back to the bindpose. Defaults to None.
            :names: (list(str), optional): the list of names of each bone we sent, if none we assume that the matrices are matching the skeleton order and size. Defaults to None.
        
        Returns:
            :np.array([bone_count,4,4], dtype=np.float32): the world transforms of the skeleton
        """
        if local_matrices is None:
            self._scratch_skeleton_[:self._skeleton_xforms.shape[0], :, :] = self._skeleton_xforms
        else:
            self._compute_skeleton_xforms(local_matrices, names)
        for i in range(self._skeleton_xforms.shape[0]):
            self._scratch_skeleton_[i, :, :] = np.dot(self._xform, self._scratch_skeleton_[i, :, :])
        return self._scratch_skeleton_[:self._skeleton_xforms.shape[0], :, :]

    def skinning_bones(self, local_matrices=None, names=None):
        """get the skinning transforms for the shader to render

        Args:
            :local_matrices: (np.array([...,4,4], dtype=np.float32), optional): the list of matrices or none to get back to the bindpose. Defaults to None.
            :names: (list(str), optional): the list of names of each bone we sent, if none we assume that the matrices are matching the skeleton order and size. Defaults to None.
        
        Returns:
            :np.array([bone_count,4,4], dtype=np.float32): the skin transforms of the skeleton
        """
        self.world_skeleton_xforms(local_matrices, names)
        for i in range(self._skeleton_xforms.shape[0]):
            self._scratch_skeleton_[i, :, :] = np.dot(self._scratch_skeleton_[i], self._mesh.bindpose[i])
            
        return self._scratch_skeleton_

    def world_skeleton_lines(self, local_matrices=None, names=None):
        """get the skeleton as a couple of points in world space, to be passed to the line renderer

        Args:
            :local_matrices: (np.array([...,4,4], dtype=np.float32), optional): the list of matrices or none to get back to the bindpose. Defaults to None.
            :names: (list(str), optional): the list of names of each bone we sent, if none we assume that the matrices are matching the skeleton order and size. Defaults to None.
        
        Returns:
            :np.array([2*bone_count,3], dtype=np.float32): the list of points to render the lines
        """
        self.world_skeleton_xforms(local_matrices, names)
        counter = itertools.count()
        for i in range(self._skeleton_xforms.shape[0]):
            parent = self._mesh.bone_parents[i]
            if parent >= 0:
                self._scratch_position_[next(counter), :] = self._scratch_skeleton_[i, :3, 3]
                self._scratch_position_[next(counter), :] = self._scratch_skeleton_[parent, :3, 3]
        return  self._scratch_position_[:next(counter),:]

    def add_bone(self, name, q, p, parent_name=None, global_space=True):
        """add a bone in the skeleton

        Args:
            :name: (str): the name of the skeleton
            :q: (np.array([4], dtype=np.float32)): the quaternion of the bone to add
            :p: (np.array([3], dtype=np.float32)): the position of the bone to add
            :parent_name: (str, optional): the name of the parent bone. Defaults to None.
            :global_space: (bool, optional): is the quaternion and position in localspace or worldspace. Defaults to True.
        """
        matrix = utils.quat_to_mat(q, p)
        parent_id = -1
        if parent_name in self.bone_names():
            parent_id = self.bone_names().index(parent_name)
            
        if parent_id > -1 and global_space:
            self.world_skeleton_xforms()
            matrix = np.dot(np.linalg.inv(self._scratch_skeleton_[parent_id, :, :]), matrix)

        self._mesh.add_bone(name, parent_id, matrix)
        self._skeleton_xforms = np.zeros_like(self._mesh.initialpose, dtype=np.float32)
        self._skeleton_xforms[:,:,:] = self._compute_skeleton_xforms()

    def _compute_skeleton_xforms(self, local_matrices=None, names=None):
        if local_matrices is not None:
            if names:
                self._scratch_skeleton_[:self._skeleton_xforms.shape[0], :, :] = self._mesh.initialpose[:, :, :]
                for i, name in enumerate(names):
                    if name in self._mesh.bone_names:
                        j = self._mesh.bone_names.index(name)
                        self._scratch_skeleton_[j, :, :] = local_matrices[i, :, :]
            elif local_matrices.shape[0] == self._skeleton_xforms.shape[0]:
                self._scratch_skeleton_[:self._skeleton_xforms.shape[0], :, :] = local_matrices
            else:
                self._scratch_skeleton_[:self._skeleton_xforms.shape[0], :, :] = self._mesh.initialpose[:, :, :]
        else:
            self._scratch_skeleton_[:self._skeleton_xforms.shape[0], :, :] = self._mesh.initialpose[:, :, :]

        for i in range(self._skeleton_xforms.shape[0]):
            parent = self._mesh.bone_parents[i]
            if parent >= 0:
                self._scratch_skeleton_[i, :, :] = np.dot(self._scratch_skeleton_[parent, :, :], self._scratch_skeleton_[i, :, :])

        return self._scratch_skeleton_[:self._skeleton_xforms.shape[0], :, :]

    

    