from pxr import Usd, UsdGeom, Gf, UsdSkel, Sdf
import numpy as np
from ..render.material import Material

class NumpyBuffer:
    def __init__(self, buffer):
        self.buffer = buffer
        self.last_index = 0
        
    def append(self, buffer):
        self.buffer[self.last_index: self.last_index+buffer.shape[0]] = buffer
        self.last_index += buffer.shape[0]
        
    def can_append(self, buffer):
        return (self.buffer.shape[0] - self.last_index) > buffer.shape[0]
        
    def get(self):
        return self.buffer[:self.last_index]

    
class VertexPrimBuffer:
    def __init__(self, pt, normal, color, ao, skin_indices, skin_weights):
        self.pt = pt
        self.normal = normal
        self.color = color
        self.ao = ao
        self.skin_indices = skin_indices
        self.skin_weights = skin_weights
        self.hash = hash((pt[0], pt[1], pt[2], normal[0], normal[1], normal[2], color[0], color[1], color[2], ao))
    
    
class MaterialPrimBuffer:
    def __init__(self, material):
        self.material = material
        self.hashmap = {}
        self.vertices = []
        self.indices = []
        
    def add_vertex(self, vertex):
        if vertex.hash not in self.hashmap:
            id = len(self.vertices)
            self.vertices.append(vertex)
            self.hashmap[vertex.hash]=id
            self.indices.append(id)
        else:
            self.indices.append(self.hashmap[vertex.hash])
            
    def compute_buffers(self, vertex_buffer, index_buffer, skin_indices_buffer=None, skin_weights_buffer=None, global_matrix=None):
        start_vert_index = vertex_buffer.last_index//10
        
        v_buff = np.array([np.concatenate([v.pt, v.normal, v.color, [v.ao]]) for v in self.vertices], dtype=np.float32).flatten()
        if not vertex_buffer.can_append(v_buff):
            raise Exception("mesh is too big to fit the buffer")
            
        if global_matrix is not None:
            # multiply each vertices by the matrix
            v_buff = v_buff.reshape(-1, 10)
            vertices = np.ones([v_buff.shape[0], 4], dtype=np.float32)
            vertices[:, :3] = v_buff[:, :3]
            v_buff[:, :3] = np.einsum('ij,kj->ki', global_matrix, vertices)[:, :3]
            # multiply each normal by the matrix
            vertices = np.zeros([v_buff.shape[0], 4], dtype=np.float32)
            vertices[:, :3] = v_buff[:, 3:6]
            v_buff[:, 3:6] = np.einsum('ij,kj->ki', global_matrix, vertices)[:, :3]
            v_buff = v_buff.flatten()

        i_buffer = np.array(self.indices, dtype=np.int16) + start_vert_index

        if not index_buffer.can_append(i_buffer):
            raise Exception("mesh is too big to fit the buffer")
            
        self.material.first_index = index_buffer.last_index
        self.material.index_count = i_buffer.shape[0]
        vertex_buffer.append(v_buff)
        index_buffer.append(i_buffer)

        if skin_indices_buffer is not None:
            si_buffer = np.array([v.skin_indices for v in self.vertices], dtype=np.float32).flatten()
            skin_indices_buffer.append(si_buffer)

        if skin_weights_buffer is not None:
            sw_buffer = np.array([v.skin_weights for v in self.vertices], dtype=np.float32).flatten()
            skin_weights_buffer.append(sw_buffer)
            

class GeoPartPrimBuffer:
    def __init__(self, local_matrix, parent=None):
        self.materials = []
        self.children = []
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        self.local_matrix = local_matrix
        
    def begin_material(self, name, albedo, roughness, metallic, reflectance, emissive):
        mat = Material(name, albedo, roughness=roughness, metallic=metallic, reflectance=reflectance, emissive=emissive)
        self.materials.append(MaterialPrimBuffer(mat))
    
    def end_material(self):
        pass
        
    def add_vertex(self, vertex):
        self.materials[-1].add_vertex(vertex)
        
    def compute_buffers(self, materials, vertex_buffer, index_buffer, skin_indices_buffer=None, skin_weights_buffer=None, parent_matrix=None):
        global_matrix = None
        if parent_matrix is not None:
            global_matrix = np.dot(parent_matrix, self.local_matrix)
            
        for child in self.children:
            child.compute_buffers(materials, vertex_buffer, index_buffer, skin_indices_buffer, skin_weights_buffer, global_matrix)
        if self.materials:
            for mat in self.materials:
                mat.compute_buffers(vertex_buffer, index_buffer, skin_indices_buffer, skin_weights_buffer, global_matrix)
            materials.extend([mat.material for mat in self.materials])
            
    def rendered_part_count(self):
        count = 0
        for child in self.children:
            count += child.rendered_part_count()
        for mat in self.materials:
            if len(mat.vertices) > 0:
                count += 1
        return count
    

class GeoPrimBuffer:
    def __init__(self, name, skeleton=None):
        self.name = name
        self.skeleton = skeleton
        self.root_part = GeoPartPrimBuffer(np.eye(4, dtype=np.float32))
        self.current_part = self.root_part
        
    def begin_part(self, local_matrix):
        self.current_part = GeoPartPrimBuffer(local_matrix, self.current_part)
        
    def end_part(self):
        self.current_part = self.current_part.parent
        
    def begin_material(self, name, albedo, roughness, metallic, reflectance, emissive):
        self.current_part.begin_material(name, albedo, roughness, metallic, reflectance, emissive)
    
    def end_material(self):
        self.current_part.end_material()
        
    def add_vertex(self, vertex):
        self.current_part.add_vertex(vertex)
        
    def create_buffers(self, allow_partial_mesh_buffer=False):
        materials = []
        count = self.root_part.rendered_part_count()
        if count == 0:
            raise Exception('no mesh in this asset')

        
        vertex_buffer = NumpyBuffer(np.zeros([327670], dtype=np.float32)) #32767 indexable vertices
        index_buffer = NumpyBuffer(np.zeros([4194304], dtype=np.int16))
        skin_indices_buffer = None
        skin_weights_buffer = None
        parent_matrix=np.eye(4, dtype=np.float32)
        if self.skeleton is not None:
            skin_indices_buffer = NumpyBuffer(np.zeros([327670], dtype=np.int32)) 
            skin_weights_buffer = NumpyBuffer(np.zeros([327670], dtype=np.float32)) 
        try:
            self.root_part.compute_buffers(materials, vertex_buffer, index_buffer, skin_indices_buffer, skin_weights_buffer, parent_matrix)
        except Exception as e:
            if not allow_partial_mesh_buffer:
                raise e
            
        # return all the buffers
        vertex_buffer = vertex_buffer.get()
        index_buffer = index_buffer.get()
        if skin_indices_buffer :
            skin_indices_buffer = skin_indices_buffer.get()
        if skin_weights_buffer :
            skin_weights_buffer = skin_weights_buffer.get()
            
        return materials, vertex_buffer, index_buffer, skin_indices_buffer, skin_weights_buffer