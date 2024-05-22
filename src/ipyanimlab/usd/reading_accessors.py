class ConstantAccessor:
    def __init__(self, data):
        self.data = data
        
    def get(self, face_id, id_offset):
        return self.data
    
    
class FaceVaryingAccessor:
    def __init__(self, face_vertex_first_index_indirection, data):
        self.face_vertex_first_index_indirection = face_vertex_first_index_indirection
        self.data = data
        
    def get(self, face_id, id_offset):
        vertex_id = self.face_vertex_first_index_indirection[face_id] + id_offset
        return self.data[vertex_id]
    

class FaceUniformAccessor:
    def __init__(self, data):
        self.data = data
        
    def get(self, face_id, id_offset):
        return self.data[face_id]
    

class VertexAccessor:
    def __init__(self, face_vertex_indices, face_vertex_first_index_indirection, data):
        self.face_vertex_indices = face_vertex_indices
        self.face_vertex_first_index_indirection = face_vertex_first_index_indirection
        self.data = data
        
    def get(self, face_id, id_offset):
        face_offset_id = self.face_vertex_first_index_indirection[face_id] + id_offset
        vertex_id = self.face_vertex_indices[face_offset_id]
        return self.data[vertex_id]
    

class VertexIndexedAccessor:
    def __init__(self, face_vertex_indices, face_vertex_first_index_indirection, indices, data):
        self.face_vertex_indices = face_vertex_indices
        self.face_vertex_first_index_indirection = face_vertex_first_index_indirection
        self.indices = indices
        self.data = data
        
    def get(self, face_id, id_offset):
        face_offset_id = self.face_vertex_first_index_indirection[face_id] + id_offset
        vertex_id = self.face_vertex_indices[face_offset_id]
        return self.data[self.indices[vertex_id]]