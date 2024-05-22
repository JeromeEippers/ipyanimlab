import ipywebgl
import numpy as np


class Ssao:
    def __init__(self, viewer:ipywebgl.GLViewer):
        kernels_count = 32
        self.kernels = np.random.random([kernels_count,3]).astype(np.float32)
        self.kernels *= np.array([2,2,1], dtype=np.float32)
        self.kernels -= np.array([1,1,0], dtype=np.float32)
        self.kernels /= np.linalg.norm(self.kernels, axis=1).reshape([kernels_count,1])
        self.kernels *= (np.random.random([kernels_count,1]) *0.9 + .1)

        self.ssao_prog = viewer.create_program_ext(
            self.screen_vertex_shader,
            self.ssao_fragment_shader,
            {
                'in_vert' : 0,
            },
            auto_execute=False
        )

        self.blur_prog = viewer.create_program_ext(
            self.screen_vertex_shader,
            self.blur_fragment_shader,
            {
                'in_vert' : 0,
            },
            auto_execute=False
        )

        noise = np.zeros([4,4,4], dtype=np.uint8)
        noise[:,:,:2] = np.random.random([4,4,2]) * 255

        noise_tex = viewer.create_texture()
        viewer.active_texture(12)
        viewer.bind_texture('TEXTURE_2D', noise_tex)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'REPEAT')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'REPEAT')
        viewer.tex_image_2d('TEXTURE_2D', 0, 'RGBA', 4, 4, 0, 'RGBA', 'UNSIGNED_BYTE', noise.flatten())

        # ssao render target
        self.occlusion_buffer = viewer.create_framebuffer()
        viewer.bind_framebuffer('FRAMEBUFFER', self.occlusion_buffer)

        # create the ssao texture and bind it to the texture_4
        self.occlusion_target = viewer.create_texture()
        viewer.active_texture(13)
        viewer.bind_texture('TEXTURE_2D', self.occlusion_target)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'R8', viewer.width, viewer.height)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT0', 'TEXTURE_2D', self.occlusion_target, 0)

        viewer.bind_framebuffer('FRAMEBUFFER', None)

        # blured ssao render target
        self.occlusion_buffer_blured = viewer.create_framebuffer()
        viewer.bind_framebuffer('FRAMEBUFFER', self.occlusion_buffer_blured)

        # create the blured ssao texture and bind it to the texture_5
        self.occlusion_target_blured = viewer.create_texture()
        viewer.active_texture(14)
        viewer.bind_texture('TEXTURE_2D', self.occlusion_target_blured)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'NEAREST')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'R8', viewer.width, viewer.height)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT0', 'TEXTURE_2D', self.occlusion_target_blured, 0)

        viewer.bind_framebuffer('FRAMEBUFFER', None)

        viewer.execute_commands(execute_once=True)


    def compute_ssao(self, viewer:ipywebgl.GLViewer, screen_vao):
        # render ssao
        viewer.bind_framebuffer('FRAMEBUFFER', self.occlusion_buffer)
        viewer.clear()
        viewer.use_program(self.ssao_prog)
        viewer.uniform('u_kernels[0]', self.kernels)
        viewer.uniform('u_radius', np.array([6], dtype=np.float32))
        viewer.uniform('u_bias', np.array([1.5], dtype=np.float32))
        viewer.uniform('u_position', np.array([8], dtype=np.int32))
        viewer.uniform('u_normal', np.array([9], dtype=np.int32))
        viewer.uniform('u_noise', np.array([12], dtype=np.int32))
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)
        
        # blur
        viewer.bind_framebuffer('FRAMEBUFFER', self.occlusion_buffer_blured)
        viewer.clear()
        viewer.use_program(self.blur_prog)
        viewer.uniform('u_texture', np.array([13], dtype=np.int32))
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)
    
        viewer.bind_framebuffer('FRAMEBUFFER', None)


    
Ssao.screen_vertex_shader = """#version 300 es
in vec2 in_vert;

void main() {
    gl_Position = vec4(in_vert, 0, 1);
}
"""

Ssao.ssao_fragment_shader = """#version 300 es
precision highp float;

//the ViewBlock that is automatically filled by ipywebgl
layout(std140) uniform ViewBlock
{
    mat4 u_cameraMatrix;          //the camera matrix in world space
    mat4 u_viewMatrix;            //the inverse of the camera matrix
    mat4 u_projectionMatrix;      //the projection matrix
    mat4 u_viewProjectionMatrix;  //the projection * view matrix
};

uniform sampler2D u_position;
uniform sampler2D u_normal;
uniform sampler2D u_noise;
uniform vec3 u_kernels[32];

uniform float u_radius;
uniform float u_bias;

out vec4 color;

void main() {

    ivec2 noiseSize = textureSize(u_noise, 0); 
    vec2 uv = gl_FragCoord.xy / vec2(1000.0, 600.0);
    vec4 posBuffer = texelFetch(u_position, ivec2(gl_FragCoord.xy), 0).rgba;
    vec3 fragPos   = posBuffer.rgb;
    vec3 normal    = texelFetch(u_normal, ivec2(gl_FragCoord.xy), 0).rgb;
    vec3 randomVec = texelFetch(u_noise, ivec2(gl_FragCoord.xy) % noiseSize, 0).rgb;
    randomVec.xy = randomVec.xy * 2.0 - 1.0;
    randomVec = vec3(1.,0.,0.);

    vec3 tangent   = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normal);  

    
    
    float occlusion = 0.0;
    float divisor = 0.00001;
    for(int i = 0; i < 32; ++i)
    {
        // get sample position
        vec3 samplePos = TBN * u_kernels[i]; // from tangent to view-space
        //vec3 samplePos = u_kernels[i].xyz; // from tangent to view-space
        samplePos = fragPos + samplePos * u_radius; 

        float depth = samplePos.z;

        vec4 offset = vec4(samplePos, 1.0);
        offset      = offset * u_projectionMatrix;    // from view to clip-space
        offset.xyz /= offset.w;               // perspective divide
        offset.xy  = offset.xy * 0.5 + 0.5; // transform to range 0.0 - 1.0

        vec2 sampleDepth = texture(u_position, offset.xy).zw ;
        float rangeCheck = smoothstep(0.0, 1.0, u_radius / abs(depth - sampleDepth.x));
        occlusion += (sampleDepth.x > depth + u_bias ? 1.0*sampleDepth.y  : 0.0) * rangeCheck;    
        divisor += sampleDepth.y;
    }  
    occlusion *= posBuffer.w;
    color = vec4((1.0 - occlusion/divisor) , 1, 1, 1);

    //turn off ssao for now
    //color = vec4(1.);
}
"""


Ssao.blur_fragment_shader = """#version 300 es
precision highp float;

uniform sampler2D u_texture;

out vec4 color;

void main() {
    float result = 0.0;
    for (int x = -2; x < 3; ++x) 
    {
        for (int y = -2; y < 3; ++y) 
        {
            float src = texelFetch(u_texture, ivec2(gl_FragCoord.xy) + ivec2(x,y), 0).r;
            result += src;
        }
    }
    float fragcolor = result / (5.0 * 5.0);
    color = vec4(fragcolor,fragcolor,fragcolor, 1);
}
"""