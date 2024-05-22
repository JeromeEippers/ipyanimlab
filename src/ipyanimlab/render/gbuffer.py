import ipywebgl
import numpy as np
from .mesh import Mesh
from . import sky

class GBuffer:

    def __init__(self, viewer:ipywebgl.GLViewer):
        self.color_geo_buffer = viewer.create_framebuffer()
        viewer.bind_framebuffer('FRAMEBUFFER', self.color_geo_buffer)

        self.color_target = viewer.create_texture()
        viewer.active_texture(6)
        viewer.bind_texture('TEXTURE_2D', self.color_target)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'RGBA8', viewer.width, viewer.height)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT0', 'TEXTURE_2D', self.color_target, 0)

        self.material_target = viewer.create_texture()
        viewer.active_texture(7)
        viewer.bind_texture('TEXTURE_2D', self.material_target)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'RGBA8', viewer.width, viewer.height)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT1', 'TEXTURE_2D', self.material_target, 0)

        self.position_target = viewer.create_texture()
        viewer.active_texture(8)
        viewer.bind_texture('TEXTURE_2D', self.position_target)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'RGBA32F', viewer.width, viewer.height)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT2', 'TEXTURE_2D', self.position_target, 0)

        self.normal_target = viewer.create_texture()
        viewer.active_texture(9)
        viewer.bind_texture('TEXTURE_2D', self.normal_target)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'RGBA16F', viewer.width, viewer.height)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT3', 'TEXTURE_2D', self.normal_target, 0)

        self.depth_target = viewer.create_texture()
        viewer.active_texture(10)
        viewer.bind_texture('TEXTURE_2D', self.depth_target)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_image_2d('TEXTURE_2D', 0, 'DEPTH_COMPONENT32F', viewer.width, viewer.height, 0, 'DEPTH_COMPONENT', 'FLOAT', None)
        #viewer.tex_storage_2d('TEXTURE_2D', 1, 'DEPTH_COMPONENT16', viewer.width, viewer.height)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'DEPTH_ATTACHMENT', 'TEXTURE_2D', self.depth_target, 0)
        

        viewer.draw_buffers(['COLOR_ATTACHMENT0', 'COLOR_ATTACHMENT1', 'COLOR_ATTACHMENT2', 'COLOR_ATTACHMENT3'])
        viewer.bind_framebuffer('FRAMEBUFFER', None)

        self.render_prog = viewer.create_program_ext(
            self.screen_vertex_shader,
            self.final_fragment_shader,
            {
                'in_vert' : 0,
            },
            auto_execute=False
        )

        viewer.execute_commands(execute_once=True)


    def activate_framebuffer(self, viewer:ipywebgl.GLViewer):
        viewer.bind_framebuffer('FRAMEBUFFER', self.color_geo_buffer)
        viewer.clear()
        viewer.enable(depth_test=True)

    def final_render(self, viewer:ipywebgl.GLViewer, screen_vao, light_dir, light_ortho_reprojection):
        viewer.bind_framebuffer('FRAMEBUFFER', None)
        viewer.clear()
        viewer.use_program(self.render_prog)
        viewer.uniform('u_resolution', np.array([viewer.width, viewer.height], dtype=np.float32))
        viewer.uniform('u_mipCount', np.array([6], dtype=np.int32))
        viewer.uniform('u_lightdir', light_dir)
        
        viewer.uniform('u_transmittance', np.array([0], dtype=np.int32))
        viewer.uniform('diffuseIBLmap', np.array([3], dtype=np.int32))
        viewer.uniform('brdfIntegrationmap', np.array([4], dtype=np.int32))
        viewer.uniform('specularIBLmap', np.array([5], dtype=np.int32))
        viewer.uniform('u_color', np.array([6], dtype=np.int32))
        viewer.uniform('u_material', np.array([7], dtype=np.int32))
        viewer.uniform('u_normal', np.array([9], dtype=np.int32))
        viewer.uniform('u_position', np.array([8], dtype=np.int32))
        viewer.uniform('u_shadowmap', np.array([11], dtype=np.int32))
        viewer.uniform('u_ssao', np.array([14], dtype=np.int32))
        viewer.uniform('u_bias', np.array([-0.002], dtype=np.float32))
        viewer.uniform('u_depthmap', np.array([10], dtype=np.int32))
        viewer.uniform_matrix('u_lightProjection', light_ortho_reprojection)
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)

GBuffer.screen_vertex_shader = """#version 300 es
in vec2 in_vert;

void main() {
    gl_Position = vec4(in_vert, 0, 1);
}
"""

GBuffer.final_fragment_shader = sky.Sky.common_sky_render_fragment_shader + """
// material parameters
uniform int u_mipCount;

uniform vec3 u_lightdir;

uniform sampler2D u_color;
uniform sampler2D u_material;
uniform sampler2D u_normal;
uniform sampler2D u_position;

uniform sampler2D specularIBLmap;
uniform sampler2D brdfIntegrationmap;
uniform sampler2D diffuseIBLmap;

uniform sampler2D u_shadowmap;
uniform mat4 u_lightProjection;
uniform float u_bias;

uniform sampler2D u_depthmap;

uniform sampler2D u_ssao;

out vec4 fragColor;

//the ViewBlock that is automatically filled by ipywebgl
layout(std140) uniform ViewBlock
{
    mat4 u_cameraMatrix;          //the camera matrix in world space
    mat4 u_viewMatrix;            //the inverse of the camera matrix
    mat4 u_projectionMatrix;      //the projection matrix
    mat4 u_viewProjectionMatrix;  //the projection * view matrix
};

vec2 directionToSphericalEnvmap(vec3 dir) {
  float s = 1.0 - mod(1.0 / (2.0*PI) * atan(dir.x, dir.z), 1.0);
  float t = 1.0 / (PI) * acos(-dir.y);
  return vec2(s, t);
}

// adapted from "Real Shading in Unreal Engine 4", Brian Karis, Epic Games
// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
vec3 specularIBL(vec3 F0 , float roughness, vec3 N, vec3 V) {
  float NoV = clamp(dot(N, V), 0.0, 1.0);
  vec3 R = reflect(-V, N);
  vec2 uv = directionToSphericalEnvmap(R);
  vec3 prefilteredColor = textureLod(specularIBLmap, uv, roughness*float(u_mipCount)).rgb;
  vec4 brdfIntegration = texture(brdfIntegrationmap, vec2(NoV, roughness));
  return prefilteredColor * ( F0 * brdfIntegration.x + brdfIntegration.y );
}

vec3 diffuseIBL(vec3 normal) {
  vec2 uv = directionToSphericalEnvmap(normal);
  return texture(diffuseIBLmap, uv).rgb;
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;
	
    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
	
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;
	
    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
    return ggx1 * ggx2;
}
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
} 

void main() {

    vec3 position = texelFetch(u_position, ivec2(gl_FragCoord.xy), 0).rgb;
    vec3 viewDir = normalize(position);
    viewDir = -(vec4(viewDir,0) * u_cameraMatrix).xyz;
    
    float ao = pow(texelFetch(u_ssao, ivec2(gl_FragCoord.xy), 0).r, 2.);
   
    vec4 colorbuffer = texelFetch(u_color, ivec2(gl_FragCoord.xy), 0).rgba;
    vec3 baseColor = colorbuffer.xyz;
    ao *= colorbuffer.a;
    vec3 normal = texelFetch(u_normal, ivec2(gl_FragCoord.xy), 0).rgb;
    vec4 material = texelFetch(u_material, ivec2(gl_FragCoord.xy), 0).rgba;
    normal = (vec4(normal,0) * u_cameraMatrix).xyz;

    baseColor = pow(baseColor, vec3(2.2));
    
    float roughness = material.x;
    float metallic = material.y;
    float reflectance = material.z;
    vec3 emission = baseColor * vec3(material.w);

    // F0 for dielectics in range [0.0, 0.16] 
    // default FO is (0.16 * 0.5^2) = 0.04
    vec3 f0 = vec3(0.16 * (reflectance * reflectance)); 
    // in case of metals, baseColor contains F0
    f0 = mix(f0, baseColor, metallic);

    // compute diffuse and specular factors
    vec3 F = fresnelSchlick(max(dot(normal, viewDir), 0.0), f0);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic; 
    
    vec3 specular = specularIBL(f0, roughness, normal, viewDir); 
    vec3 diffuse = diffuseIBL(normal);

    // shading front-facing
    vec3 color = emission + (kD * baseColor * diffuse + specular) * ao ;
    
    // calculate sun radiance
    {
        vec4 world_pos = vec4(position,1) * u_cameraMatrix;
    
        vec4 shadowcoord = world_pos * u_lightProjection;
        vec3 shadow = shadowcoord.xyz / shadowcoord.w;
        float currentDepth = shadow.z + u_bias;

        bool inRange =
          shadow.x >= 0.0 &&
          shadow.x <= 1.0 &&
          shadow.y >= 0.0 &&
          shadow.y <= 1.0 &&
          shadow.z >= 0.0 &&
          shadow.z <= 1.0;

        float projectedDepth = texture(u_shadowmap, shadow.xy).r;
        float shadowLight = (inRange && projectedDepth < currentDepth) ? 0.0 : 1.0;
    
        vec3 L = u_lightdir.xyz;
        vec3 H = normalize(viewDir + L);
        vec3 radiance = vec3(2.2) * shadowLight;  
        
        if (rayIntersectSphere(viewPos, L, groundRadiusMM) >= 0.0) {
            radiance *= 0.0;
        } else {
            // If the sun value is applied to this pixel, we need to calculate the transmittance to obscure it.
            radiance *= getValFromTLUT(u_transmittance, tLUTRes.xy, viewPos, L);
        }

        // cook-torrance brdf
        float NDF = DistributionGGX(normal, H, roughness);        
        float G   = GeometrySmith(normal, viewDir, L, roughness);      
        vec3 F    = fresnelSchlick(max(dot(H, viewDir), 0.0), f0);       

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;	  

        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, L), 0.0) + 0.0001;
        vec3 specular     = numerator / denominator;  

        // add to outgoing radiance Lo
        float NdotL = max(dot(normal, L), 0.0);                
        color +=  (kD * baseColor * ao / PI + specular) * radiance * NdotL;
    }

    gl_FragDepth = texelFetch(u_depthmap, ivec2(gl_FragCoord.xy), 0).r;
    fragColor.rgb = pow(color, vec3(1.0/2.2));
    fragColor.a = 1.0;
    
}
"""
