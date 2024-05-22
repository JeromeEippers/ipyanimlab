import numpy as np

class Material():
    def __init__(self, name, albedo, roughness=0.1, metallic=0.0, reflectance=.5, emissive=0.0, first_index=0, index_count=0):

        self.name = name
        self.albedo = albedo
        self._roughness = roughness
        self._metallic = metallic
        self._reflectance = reflectance
        self._emissive = emissive
        self.material_uniform = None
        self.first_index = first_index
        self.index_count = index_count
        self.compute()

    def compute(self):
        self.material_uniform = np.array([self._roughness, self._metallic, self._reflectance, self._emissive], dtype=np.float32)

    def set_roughness(self, value):
        self._roughness = value
        self.compute()

    def set_metallic(self, value):
        self._metallic = value
        self.compute()

    def set_reflectance(self, value):
        self._reflectance = value
        self.compute()

    def set_emissive(self, value):
        self._emissive = value
        self.compute()

    def set_albedo(self, value):
        self.albedo = value