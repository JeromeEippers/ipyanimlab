from enum import Enum
import os.path

import ipywebgl
import numpy as np

from .render import screen_vbo
from .render import sky
from .render import show_texture
from .render import frustum
from .render import gbuffer
from .render import mesh_render
from .render import mesh as mesh_module
from .render import background
from .render import shadow
from .render import ssao
from .render import ground
from .render import axis
from .render import line
from . import procedural_asset
from .usd.import_asset import import_usd_asset
from .asset import Asset
from . import assets


class ShadowQuality(Enum):
    """The shadow quality

    .LOW
    .MID
    .HIGH
    .ULTRA
    """
    LOW=1
    MID=2
    HIGH=4
    ULTRA=8


class ShadowSize(Enum):
    """Shadow size

    .NARROW
    .WIDE
    """

    NARROW=1
    WIDE=4


class Viewer(ipywebgl.GLViewer):
    """The Viewer for the ipyAnimLab
    
    This inherits from the ipywebgl viewer, so all the function from there are available.

    The ipyAnimLab is using the column_major shader convention.

    Args:
        :shadow_quality: (ShadowQuality enum, optional): the quality of the shadow. Defaults to ShadowQuality.MID. 
        :shadow_size: (ShadowSize enum, optional): the type of shadow (around one character or around the scene). Defaults to ShadowSize.WIDE.
        :width: (int): the width of the canvas. Defaults to 1000.
        :height: (int): the height of the canvas. Defaults to 600.
        :camera_pos: ([float, float, float]): the camera position in the scene. Defaults to [-370, 280, 350].
        :camera_yaw: (float): the camera yaw angle in degree. Defaults to -45.
        :camera_pitch: (float): the camera pitch angle in degree. Defaults to -18.
        :camera_fov: (float): the camera field of view in degree. Defaults to 50.
        :camera_near: (float): the camera near plane distance. Defaults to 20.
        :camera_far: (float): the camera far plane distance. Defaults to 2800.
        :mouse_speed: (float): mouse speed (camera rotation speed). Defaults to 1.
        :move_speed: (float): move speed (camera translation speed). Defaults to 1.
        :move_keys: (str): the move keys as a string. Forward, Left, Back, Right. Defaults to 'wasd'.
        :sync_image_data: (bool): do we store the rendered imaged in python. This will significantly slow the rendering. Defaults to False.
        :image_data: (bytes): the stored image as bytes. If the sync_image_data is set to True.
        :verbose: (int): with verbose set to 1, all the commands executed by the frontend will be logged in the console. Defaults to 0.
    """

    def __init__(self, shadow_quality=ShadowQuality.MID, shadow_size=ShadowSize.WIDE, **kwargs):
        default = {'shader_matrix_major':'column_major', 'width':1000, 'height':600, 'camera_near':20, 'camera_far':2800, 'camera_pos':[-370, 280, 350], 'camera_pitch':-18, 'camera_yaw':-45}
        default.update(kwargs)
        super().__init__(**default)

        self._screen_vao = screen_vbo.create_screen_vao(self)
        self._time = 0
        self._sky = sky.Sky(self, self._screen_vao)
        
        self.light_matrix = np.eye(4, dtype=np.float32)
        self.light_poi = np.zeros((3), dtype=np.float32)
        self.light_ortho_projection = np.eye(4, dtype=np.float32)
        self.light_ortho_reprojection = np.eye(4, dtype=np.float32)
        self._eye = np.eye(4, dtype=np.float32)

        self.set_shadow_projection(shadow_size.value * 256.0, shadow_size.value * 256.0, 1.0, 4096.0)
        self.set_time_of_day(20)

        self._gbuffer = gbuffer.GBuffer(self)
        self._mesh_render = mesh_render.MeshRender(self)
        self._background = background.Background(self)
        self._shadow = shadow.Shadow(self, resolution=shadow_quality.value * 1024.0)
        self._rendering_shadow = False
        self._ssao = ssao.Ssao(self)
        self._ground = ground.Ground(self)
        self._axis = axis.Axis(self)
        self._line = line.Line(self)

        self._frustum = frustum.Frustum(self)
        self._debug_buffers = show_texture.ShowTexture(self)

    def set_time_of_day(self, time:int):
        """Change the time of day

        Args:
            :time: (int): the time between dawn (0) to noon (60)
        """
        self._time = time
        self._sky.update_time(self, self._screen_vao, time)

        def getSunAltitude(time):
            periodSec = 120.0
            halfPeriod = periodSec / 2.0
            sunriseShift = 0.1
            cyclePoint = (1.0 - np.abs(((time%periodSec)-halfPeriod)/halfPeriod))
            cyclePoint = (cyclePoint*(1.0+sunriseShift))-sunriseShift
            return (0.5*np.pi)*cyclePoint

        altitude = getSunAltitude(time)
        self.light_matrix[...] = np.eye(4, dtype=np.float32)
        self.light_matrix[:3, 3] = np.array([0.0, -np.sin(altitude), -np.cos(altitude)])
        self.light_matrix[:3, 1] = np.array([0.0, np.cos(altitude), -np.sin(altitude)])
        self.light_matrix[:3, 2] = np.array([0.0, np.sin(altitude), np.cos(altitude)])
        self.__update_light_matrices()


    def set_shadow_projection(self, width, height, near, far):
        """update the shadow projection

        Args:
            :width: (float): width of the projection
            :height: (float): height of the projection
            :near: (float): near plane distance
            :far: (float): far plane distance
        """
        A = 1. / width
        B = 1. / height
        C = -(far + near) / (far - near)
        D = -2. / (far - near)
        self.light_ortho = np.array([
            [A, 0, 0, 0],
            [0, B, 0, 0],
            [0, 0, D, C],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.__update_light_matrices()


    def set_shadow_poi(self, position):
        """update the shadow point of interest (the center of the projection)

        Args:
            :position: (np.array([0,0,0], dtype=np.float32)): the position as a float32 numpy array
        """
        self.light_poi = position
        self.__update_light_matrices()


    def create_sphere(self, radius=1.0):
        """Create a sphere asset

        Args:
            :radius: (float, optional): radius of the sphere. Defaults to 1.0.

        Returns:
            :asset: the created asset
        """
        return procedural_asset.create_sphere(self, radius)


    def create_cube(self, width=1.0, height=1.0, depth=1.0):
        """create a cube asset

        Args:
            :width: (float, optional): the with of the cube. Defaults to 1.0.
            :height: (float, optional): the height of the cube. Defaults to 1.0.
            :depth: (float, optional): the depth of the cube. Defaults to 1.0.

        Returns:
            :asset: the created asset
        """
        return procedural_asset.create_cube(self, width, height, depth)


    def create_asset(self, vertices=None, indices=None, materials=None, boneids=None, weights=None, bindpose=None, initialpose=None, bone_names=None, bone_parents=None, name=None, mesh=None):
        """Create an asset from raw data

            For a skinned mesh we only support 256 bones maximum
        Args:
            :vertices: (np float32 array, optional): the flatten vertex buffer with [pos.x, pos.y, pos.z, normal.x, normal.y, normal.z, color.r, color.g, color.b, ambientocclusion] * number_of_vertices. Defaults to None.
            :indices: (np int16 array, optional): the indices to draw the mesh as triangle list. Defaults to None.
            :materials: (list of Material, optional): the material list to render the mesh. If none it renders the entire mesh with one default material. Defaults to None.
            :boneids: (np int 32 array, optional): if skinned mesh, the flatten list of boneids per vertex, can be 4 ids per vertex or 8 ids per vertex. Defaults to None.
            :weights: (np float 32 array, optional): if skinned mesh, the fatten list of bone weights per vertex, can be 4 or 8 weights per vertex, but it must match the boneids. Defaults to None.
            :bindpose: (np float 32 [bone_count,4,4], optional): if skinned mesh this is the bindpose of the skeleton (the inverse of the world matrices of each bone). Defaults to None.
            :initialpose: (np float 32 [bone_count,4,4], optional): if skinned mesh this is the localpose of the skeleton. Defaults to None.
            :bone_names: (list of strings, optional): the list of bone names. It must match the number of bone. Defaults to None.
            :bone_parents: (np int32 array, optional): the list of parent ids for each bone. It must match the number of bone. Defaults to None.
            :name: (string, optional): the name of the asset. Defaults to None.
            :mesh: (Mesh, optional): if a mesh is already created we will discard any given buffers and use the provided mesh instead. Defaults to None.

        Returns:
            :Asset: the created asset
        """
        if mesh is None:
            mesh = mesh_module.Mesh(self, vertices, indices, boneids, weights, bindpose, initialpose, bone_names, bone_parents, name)
        return Asset(mesh, materials)


    def import_usd_asset(self, path):
        """Import a usd file as an asset.
        The asset can be a rigid asset or a skeletal asset

        if the path is not found, we will try to load the asset from the internal 'assets' folder

        Args:
            :path: (string): the path to the usd file

        Returns:
            :Asset: the loaded asset
        """
        if not os.path.exists(path):
            path = assets.get_asset_path(path)
        if not os.path.exists(path):
            raise Exception(f"{path} not found")
        return import_usd_asset(self, path)


    def begin_shadow(self):
        """Before rendering any asset in the shadow rendered texture we need to activate the shadow rendering mode.
        """
        self._shadow.activate_shadow(self)
        self._rendering_shadow = True


    def end_shadow(self):
        """Once we have finished rendering all the asset we wanted in the shadow rendered texture  we deactivate the shadow rendering mode.
        This will make the texture available for the lighting after.
        """
        self.bind_framebuffer('FRAMEBUFFER', None)
        self.viewport(0, 0, self.width, self.height)
        self._rendering_shadow = False
    

    def begin_display(self):
        """Before rendering any asset to screen (and after ending the shadow rendering) we need to activate the display rendering mode.
        This will activate all the needed gbuffers for the deferred rendering.
        """
        self._gbuffer.activate_framebuffer(self)
        

    def draw(self, asset, worlds=None, names=None, materials=None):
        """Draw an asset
        This funciton is only available bewteen begin_shadow and end_shadow, and begin_display and end_display

        if no worlds are given, we use the internal values from the asset.
        if we have one world and a rigid asset, we use the given matrix instead of the xform from the asset.
        if we have multiple worlds and a rigid asset, we do a instancied rendering of the mesh.
        if we have multiple worlds and a skinned asset, we assume them to be the skeleton matrices in local space.
        > if we have names, we do the mapping between the names and the skeleton. names and worlds must have the same length
        > if we do not have names, we assume that the list of worlds match the skeleton exactly.

        Args:
            :asset: (Asset): the asset to render.
            :worlds: (np float32 [...,4,4], optional): a 4x4 matrix of an array of 4x4 matrices. Defaults to None.
            :names: (list of string, optional): the list of names of the bones given as worlds. Defaults to None.
            :materials: (list of Material, optional): the list of materials we want to use instead of the one found in the asset. Defaults to None.

        """
        if asset.is_rigid():
            if worlds is None:
                worlds = asset.xform()

            if len(worlds.shape) == 2:
                worlds = worlds[np.newaxis, ...]
        else:
            worlds = asset.skinning_bones(worlds, names)
            
        if self._rendering_shadow:
            self._shadow.render_mesh(self, asset._mesh, self.light_ortho_projection, worlds)
        else:
            if materials is None:
                materials = asset._materials
            self._mesh_render.render_mesh(self, asset._mesh, worlds, materials)


    def draw_ground(self):
        if self._rendering_shadow:
            self._ground.draw_arrays(self)
        else:
            self._ground.show(self)


    def end_display(self):
        """Once all the asset are draw in the gbuffers we call end_display.
        This will compute the SSAO, lighting and shadow and display the result on framebuffer
        """
        self._background.render(self, self._screen_vao)
        self._ssao.compute_ssao(self, self._screen_vao)
        self.bind_framebuffer('FRAMEBUFFER', None)
        self._gbuffer.final_render(self, self._screen_vao, self.light_matrix[:3, 2], self.light_ortho_reprojection)


    def draw_axis(self, matrices, scale=5.0):
        """Draw one or multiple XYZ axis, this must be called after end_display
        You can use a disable(depth_test=True) to be sure to display on top of the zbuffer

        Args:
            :matrices: (np float32 array [..., 4, 4]): the matrix of list of matrices to render
            :scale: (float, optional): the size of the axis to render. Defaults to 5.0.
        """
        if len(matrices.shape) == 2:
            matrices = matrices[np.newaxis, ...]
        self._axis.show(self, matrices, scale)


    def draw_lines(self, positions, color = np.zeros([3], dtype=np.float32)):
        """Draw lines, this must be called after end_display
        You can use a disable(depth_test=True) to be sure to display on top of the zbuffer

        Args:
            :positions: (np float 32 array [2*line_count, 3]): a list of points to draw line. A line will be rendered between each pair of points
            :color: (np float 32 array, optional): the rgb color between 0 and 1. Defaults to np.zeros([3], dtype=np.float32).
        """
        self._line.show(self, positions, color)
            

    def debug_display_buffer(self, texture_id:int, lod_id:int=0):
        """debug display one of the internal render texture used by the viewer, this must be after the end_display

        Args:
            :texture_id: (int): the id of the texture to render
            :lod_id: (int, optional): the lod of the texture to render. Defaults to 0.
        """
        self._debug_buffers.show(self, self._screen_vao, texture_id, lod_id)


    def debug_display_color(self):
        """Debug display the color buffer, this must be after the end_display
        """
        self._debug_buffers.show(self, self._screen_vao, 6, 0, 0)


    def debug_display_ao(self):
        """debug display the ambient occlusion buffer (the value coming the ao in the mesh), this must be after the end_display
        """
        self._debug_buffers.show(self, self._screen_vao, 6, 0, 2)


    def debug_display_normal(self):
        """debug display the normal buffer, this must be after the end_display
        """
        self._debug_buffers.show(self, self._screen_vao, 9, 0, 0)


    def debug_display_ssao(self):
        """debug display the screen space ambien occlusion buffer, this must be after the end_display
        """
        self._debug_buffers.show(self, self._screen_vao, 14, 0, 1)


    def debug_draw_shadow_frustum(self):
        """display the shadow frustrum, this must be after the end_display
        """
        self._shadow.render_frustum(self, self.light_ortho_projection)


    def debug_draw_frustum(self, viewproj_matrix):
        """display a frustum

        Args:
            :viewproj_matrix: (np float 32 array [4,4]): the matrix to display
        """
        self._frustum.show(self, viewproj_matrix)

    def __update_light_matrices(self):
        light_matrix = np.eye(4, dtype=np.float32)
        light_matrix[...] = self.light_matrix
        light_matrix[:3, 3] = (light_matrix[:3, 2] * 2048) + self.light_poi

        light_matrix = np.linalg.inv(light_matrix)
    
        bias = np.array(
            [[0.5, 0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        self.light_ortho_projection = np.dot(self.light_ortho, light_matrix)
        self.light_ortho_reprojection = np.dot(bias, self.light_ortho_projection)