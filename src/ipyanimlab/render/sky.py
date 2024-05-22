import ipywebgl
import numpy as np

class Sky:

    def __init__(self, viewer:ipywebgl.GLViewer, screen_vao:ipywebgl.glresource.GLResourceWidget):
        self.transmittance_prog = viewer.create_program_ext(
            self.screen_vs,
            self.transmittance_fragment_shader,
            {'in_vert' : 0},
            auto_execute=False)
        self.scattering_prog = viewer.create_program_ext(
            self.screen_vs,
            self.scattering_fragment_shader,
            {'in_vert' : 0},
            auto_execute=False)
        self.skyview_prog = viewer.create_program_ext(
            self.screen_vs,
            self.skyview_fragment_shader,
            {'in_vert' : 0},
            auto_execute=False)
        self.diffuse_importance_sampling_prog = viewer.create_program_ext(
            self.screen_vs,
            self.diffuse_important_sampling_fragment_shader,
            {'in_vert' : 0},
            auto_execute=False)
        self.brdf_integration_prog = viewer.create_program_ext(
            self.screen_vs,
            self.brdf_integration_fragment_shader,
            {'in_vert' : 0},
            auto_execute=False)
        self.brdf_importance_sampling_prog = viewer.create_program_ext(
            self.screen_vs,
            self.brdf_importance_sampling_fragment_shader,
            {'in_vert' : 0},
            auto_execute=False)

        self.transmittance_buffer = viewer.create_framebuffer()
        viewer.bind_framebuffer('FRAMEBUFFER', self.transmittance_buffer)
        self.transmittance_lut = viewer.create_texture()
        viewer.active_texture(0)
        viewer.bind_texture('TEXTURE_2D', self.transmittance_lut)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'RGBA16F', 256, 64)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT0', 'TEXTURE_2D', self.transmittance_lut, 0)

        self.scattering_buffer = viewer.create_framebuffer()
        viewer.bind_framebuffer('FRAMEBUFFER', self.scattering_buffer)
        self.scattering_lut = viewer.create_texture()
        viewer.active_texture(1)
        viewer.bind_texture('TEXTURE_2D', self.scattering_lut)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'RGBA16F', 32, 32)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT0', 'TEXTURE_2D', self.scattering_lut, 0)

        self.skyview_buffer = viewer.create_framebuffer()
        viewer.bind_framebuffer('FRAMEBUFFER', self.skyview_buffer)
        self.skyview_tex = viewer.create_texture()
        viewer.active_texture(2)
        viewer.bind_texture('TEXTURE_2D', self.skyview_tex)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'RGBA16F', 200, 200)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT0', 'TEXTURE_2D', self.skyview_tex, 0)

        self.diffuse_ibl_buffer = viewer.create_framebuffer()
        viewer.bind_framebuffer('FRAMEBUFFER', self.diffuse_ibl_buffer)
        self.diffuse_ibl_tex = viewer.create_texture()
        viewer.active_texture(3)
        viewer.bind_texture('TEXTURE_2D', self.diffuse_ibl_tex)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'RGBA16F', 128, 64)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT0', 'TEXTURE_2D', self.diffuse_ibl_tex, 0)

        self.brdf_integration_buffer = viewer.create_framebuffer()
        viewer.bind_framebuffer('FRAMEBUFFER', self.brdf_integration_buffer)
        self.brdf_integration_tex = viewer.create_texture()
        viewer.active_texture(4)
        viewer.bind_texture('TEXTURE_2D', self.brdf_integration_tex)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 1, 'RGBA16F', 128, 128)
        viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT0', 'TEXTURE_2D', self.brdf_integration_tex, 0)

        self.brdf_tex = viewer.create_texture()
        viewer.active_texture(5)
        viewer.bind_texture('TEXTURE_2D', self.brdf_tex)
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MAG_FILTER', 'LINEAR_MIPMAP_LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_MIN_FILTER', 'LINEAR_MIPMAP_LINEAR')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_S', 'CLAMP_TO_EDGE')
        viewer.tex_parameter('TEXTURE_2D', 'TEXTURE_WRAP_T', 'CLAMP_TO_EDGE')
        viewer.tex_storage_2d('TEXTURE_2D', 6, 'RGBA16F', 512, 256)

        self.brdf_buffers = [0]*6
        for i in range(6):
            self.brdf_buffers[i] = viewer.create_framebuffer()
            viewer.bind_framebuffer('FRAMEBUFFER', self.brdf_buffers[i])
            viewer.framebuffer_texture_2d('FRAMEBUFFER', 'COLOR_ATTACHMENT0', 'TEXTURE_2D', self.brdf_tex, i)

        viewer.bind_framebuffer('FRAMEBUFFER', None)

        # this texture is created once only
        viewer.bind_framebuffer('FRAMEBUFFER', self.brdf_integration_buffer)
        viewer.viewport(0, 0, 128, 128)
        viewer.clear()
        viewer.use_program(self.brdf_integration_prog)
        viewer.uniform('u_resolution', np.array([viewer.width, viewer.height], dtype=np.float32))
        viewer.uniform('u_samples', np.array([128], dtype=np.int32))
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)
        viewer.bind_framebuffer('FRAMEBUFFER', None)
        viewer.viewport(0, 0, viewer.width, viewer.height)

        viewer.execute_commands(execute_once=True)

    def update_time(self, viewer:ipywebgl.GLViewer, screen_vao:ipywebgl.glresource.GLResourceWidget, time:float):
        viewer.bind_framebuffer('FRAMEBUFFER', self.transmittance_buffer)
        viewer.viewport(0, 0, 256, 64)
        viewer.clear()
        viewer.use_program(self.transmittance_prog)
        viewer.uniform('u_resolution', np.array([viewer.width, viewer.height], dtype=np.float32))
        viewer.uniform('u_time', np.array([time], dtype=np.float32))
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)
        viewer.bind_framebuffer('FRAMEBUFFER', None)
        viewer.viewport(0, 0, viewer.width, viewer.height)

        viewer.bind_framebuffer('FRAMEBUFFER', self.scattering_buffer)
        viewer.viewport(0, 0, 32, 32)
        viewer.clear()
        viewer.use_program(self.scattering_prog)
        viewer.uniform('u_resolution', np.array([viewer.width, viewer.height], dtype=np.float32))
        viewer.uniform('u_time', np.array([time], dtype=np.float32))
        viewer.uniform('u_transmittance', np.array([0], dtype=np.int32))
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)
        viewer.bind_framebuffer('FRAMEBUFFER', None)
        viewer.viewport(0, 0, viewer.width, viewer.height)

        viewer.bind_framebuffer('FRAMEBUFFER', self.skyview_buffer)
        viewer.viewport(0, 0, 200, 200)
        viewer.clear()
        viewer.use_program(self.skyview_prog)
        viewer.uniform('u_resolution', np.array([viewer.width, viewer.height], dtype=np.float32))
        viewer.uniform('u_time', np.array([time], dtype=np.float32))
        viewer.uniform('u_transmittance', np.array([0], dtype=np.int32))
        viewer.uniform('u_scattering', np.array([1], dtype=np.int32))
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)
        viewer.bind_framebuffer('FRAMEBUFFER', None)
        viewer.viewport(0, 0, viewer.width, viewer.height)
        
        viewer.bind_framebuffer('FRAMEBUFFER', self.diffuse_ibl_buffer)
        viewer.viewport(0, 0, 128, 64)
        viewer.clear()
        viewer.use_program(self.diffuse_importance_sampling_prog)
        viewer.uniform('u_resolution', np.array([viewer.width, viewer.height], dtype=np.float32))
        viewer.uniform('u_time', np.array([time], dtype=np.float32))
        viewer.uniform('u_samples', np.array([2048], dtype=np.int32))
        viewer.uniform('u_transmittance', np.array([0], dtype=np.int32))
        viewer.uniform('u_skyview', np.array([2], dtype=np.int32))
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)
        viewer.bind_framebuffer('FRAMEBUFFER', None)
        viewer.viewport(0, 0, viewer.width, viewer.height)
        
        viewer.bind_framebuffer('FRAMEBUFFER', self.brdf_integration_buffer)
        viewer.viewport(0, 0, 128, 128)
        viewer.clear()
        viewer.use_program(self.brdf_integration_prog)
        viewer.uniform('u_resolution', np.array([viewer.width, viewer.height], dtype=np.float32))
        viewer.uniform('u_samples', np.array([128], dtype=np.int32))
        viewer.bind_vertex_array(screen_vao)
        viewer.draw_arrays('TRIANGLES',0, 6)
        viewer.bind_framebuffer('FRAMEBUFFER', None)
        viewer.viewport(0, 0, viewer.width, viewer.height)
        
        for i in range(6):
            viewer.bind_framebuffer('FRAMEBUFFER', self.brdf_buffers[i])
            viewer.viewport(0, 0, 1024//(2**i), 512//(2**i))
            viewer.clear()
            viewer.use_program(self.brdf_importance_sampling_prog)
            viewer.uniform('u_resolution', np.array([viewer.width, viewer.height], dtype=np.float32))
            viewer.uniform('u_time', np.array([time], dtype=np.float32))
            viewer.uniform('u_transmittance', np.array([0], dtype=np.int32))
            viewer.uniform('u_skyview', np.array([2], dtype=np.int32))
            viewer.uniform('u_texresolution', np.array([512//(2**i), 256//(2**i)], dtype=np.float32))
            viewer.uniform('u_samples', np.array([512], dtype=np.int32))
            viewer.uniform('u_roughness', np.array([0.2*i], dtype=np.float32))
            
            viewer.bind_vertex_array(screen_vao)
            viewer.draw_arrays('TRIANGLES',0, 6)
        viewer.bind_framebuffer('FRAMEBUFFER', None)
        viewer.viewport(0, 0, viewer.width, viewer.height)
        
        viewer.execute_commands(execute_once=True)

Sky.screen_vs = """#version 300 es
in vec2 in_vert;

void main() {
    gl_Position = vec4(in_vert, 0, 1);
}
"""

Sky.common_fragment_shader = """#version 300 es
precision highp float;

const float PI = 3.14159265358;

uniform float u_time;
uniform vec2 u_resolution;


// Units are in megameters.
const float groundRadiusMM = 6.360;
const float atmosphereRadiusMM = 6.460;

// 200M above the ground.
const vec3 viewPos = vec3(0.0, groundRadiusMM + 0.0002, 0.0);

const vec2 tLUTRes = vec2(256.0, 64.0);
const vec2 msLUTRes = vec2(32.0, 32.0);
// Doubled the vertical skyLUT res from the paper, looks way
// better for sunrise.
const vec2 skyLUTRes = vec2(200.0, 200.0);

const vec3 groundAlbedo = vec3(0.2);

// These are per megameter.
const vec3 rayleighScatteringBase = vec3(5.802, 13.558, 33.1);
const float rayleighAbsorptionBase = 0.0;

const float mieScatteringBase = 3.996;
const float mieAbsorptionBase = 4.4;

const vec3 ozoneAbsorptionBase = vec3(0.650, 1.881, .085);

/*
 * Animates the sun movement.
 */
float getSunAltitude(float time)
{
    const float periodSec = 120.0;
    const float halfPeriod = periodSec / 2.0;
    const float sunriseShift = 0.1;
    float cyclePoint = (1.0 - abs((mod(time,periodSec)-halfPeriod)/halfPeriod));
    cyclePoint = (cyclePoint*(1.0+sunriseShift))-sunriseShift;
    return (0.5*PI)*cyclePoint;
}
vec3 getSunDir(float time)
{
    float altitude = getSunAltitude(time);
    return normalize(vec3(0.0, sin(altitude), cos(altitude)));
}

float getMiePhase(float cosTheta) {
    const float g = 0.8;
    const float scale = 3.0/(8.0*PI);
    
    float num = (1.0-g*g)*(1.0+cosTheta*cosTheta);
    float denom = (2.0+g*g)*pow((1.0 + g*g - 2.0*g*cosTheta), 1.5);
    
    return scale*num/denom;
}

float getRayleighPhase(float cosTheta) {
    const float k = 3.0/(16.0*PI);
    return k*(1.0+cosTheta*cosTheta);
}

void getScatteringValues(vec3 pos, 
                         out vec3 rayleighScattering, 
                         out float mieScattering,
                         out vec3 extinction) {
    float altitudeKM = (length(pos)-groundRadiusMM)*1000.0;
    // Note: Paper gets these switched up.
    float rayleighDensity = exp(-altitudeKM/8.0);
    float mieDensity = exp(-altitudeKM/1.2);
    
    rayleighScattering = rayleighScatteringBase*rayleighDensity;
    float rayleighAbsorption = rayleighAbsorptionBase*rayleighDensity;
    
    mieScattering = mieScatteringBase*mieDensity;
    float mieAbsorption = mieAbsorptionBase*mieDensity;
    
    vec3 ozoneAbsorption = ozoneAbsorptionBase*max(0.0, 1.0 - abs(altitudeKM-25.0)/15.0);
    
    extinction = rayleighScattering + rayleighAbsorption + mieScattering + mieAbsorption + ozoneAbsorption;
}

float safeacos(const float x) {
    return acos(clamp(x, -1.0, 1.0));
}

// From https://gamedev.stackexchange.com/questions/96459/fast-ray-sphere-collision-code.
float rayIntersectSphere(vec3 ro, vec3 rd, float rad) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - rad*rad;
    if (c > 0.0f && b > 0.0) return -1.0;
    float discr = b*b - c;
    if (discr < 0.0) return -1.0;
    // Special case: inside sphere, use far discriminant
    if (discr > b*b) return (-b + sqrt(discr));
    return -b - sqrt(discr);
}

/*
 * Same parameterization here.
 */
vec3 getValFromTLUT(sampler2D tex, vec2 bufferRes, vec3 pos, vec3 sunDir) {
    float height = length(pos);
    vec3 up = pos / height;
	float sunCosZenithAngle = dot(sunDir, up);
    vec2 uv = vec2(tLUTRes.x*clamp(0.5 + 0.5*sunCosZenithAngle, 0.0, 1.0),
                   tLUTRes.y*max(0.0, min(1.0, (height - groundRadiusMM)/(atmosphereRadiusMM - groundRadiusMM))));
    uv /= bufferRes;
    return texture(tex, uv).rgb;
}
vec3 getValFromMultiScattLUT(sampler2D tex, vec2 bufferRes, vec3 pos, vec3 sunDir) {
    float height = length(pos);
    vec3 up = pos / height;
	float sunCosZenithAngle = dot(sunDir, up);
    vec2 uv = vec2(msLUTRes.x*clamp(0.5 + 0.5*sunCosZenithAngle, 0.0, 1.0),
                   msLUTRes.y*max(0.0, min(1.0, (height - groundRadiusMM)/(atmosphereRadiusMM - groundRadiusMM))));
    uv /= bufferRes;
    return texture(tex, uv).rgb;
}
"""

Sky.transmittance_fragment_shader = Sky.common_fragment_shader + """
// Buffer A generates the Transmittance LUT. Each pixel coordinate corresponds to a height and sun zenith angle, and
// the value is the transmittance from that point to sun, through the atmosphere.
const float sunTransmittanceSteps = 40.0;

vec3 getSunTransmittance(vec3 pos, vec3 sunDir) {
    if (rayIntersectSphere(pos, sunDir, groundRadiusMM) > 0.0) {
        return vec3(0.0);
    }
    
    float atmoDist = rayIntersectSphere(pos, sunDir, atmosphereRadiusMM);
    float t = 0.0;
    
    vec3 transmittance = vec3(1.0);
    for (float i = 0.0; i < sunTransmittanceSteps; i += 1.0) {
        float newT = ((i + 0.3)/sunTransmittanceSteps)*atmoDist;
        float dt = newT - t;
        t = newT;
        
        vec3 newPos = pos + t*sunDir;
        
        vec3 rayleighScattering, extinction;
        float mieScattering;
        getScatteringValues(newPos, rayleighScattering, mieScattering, extinction);
        
        transmittance *= exp(-dt*extinction);
    }
    return transmittance;
}

out vec4 fragColor;

void main()
{
    float u = gl_FragCoord.x/tLUTRes.x;
    float v = gl_FragCoord.y/tLUTRes.y;
    
    float sunCosTheta = 2.0*u - 1.0;
    float sunTheta = safeacos(sunCosTheta);
    float height = mix(groundRadiusMM, atmosphereRadiusMM, v);
    
    vec3 pos = vec3(0.0, height, 0.0); 
    vec3 sunDir = normalize(vec3(0.0, sunCosTheta, -sin(sunTheta)));
    
    fragColor = vec4(getSunTransmittance(pos, sunDir), 1.0);
}
"""

Sky.scattering_fragment_shader = Sky.common_fragment_shader + """
// Buffer B is the multiple-scattering LUT. Each pixel coordinate corresponds to a height and sun zenith angle, and
// the value is the multiple scattering approximation (Psi_ms from the paper, Eq. 10).

uniform sampler2D u_transmittance;

const float mulScattSteps = 20.0;
const int sqrtSamples = 8;

vec3 getSphericalDir(float theta, float phi) {
     float cosPhi = cos(phi);
     float sinPhi = sin(phi);
     float cosTheta = cos(theta);
     float sinTheta = sin(theta);
     return vec3(sinPhi*sinTheta, cosPhi, sinPhi*cosTheta);
}

// Calculates Equation (5) and (7) from the paper.
void getMulScattValues(vec3 pos, vec3 sunDir, out vec3 lumTotal, out vec3 fms) {
    lumTotal = vec3(0.0);
    fms = vec3(0.0);
    
    float invSamples = 1.0/float(sqrtSamples*sqrtSamples);
    for (int i = 0; i < sqrtSamples; i++) {
        for (int j = 0; j < sqrtSamples; j++) {
            // This integral is symmetric about theta = 0 (or theta = PI), so we
            // only need to integrate from zero to PI, not zero to 2*PI.
            float theta = PI * (float(i) + 0.5) / float(sqrtSamples);
            float phi = safeacos(1.0 - 2.0*(float(j) + 0.5) / float(sqrtSamples));
            vec3 rayDir = getSphericalDir(theta, phi);
            
            float atmoDist = rayIntersectSphere(pos, rayDir, atmosphereRadiusMM);
            float groundDist = rayIntersectSphere(pos, rayDir, groundRadiusMM);
            float tMax = atmoDist;
            if (groundDist > 0.0) {
                tMax = groundDist;
            }
            
            float cosTheta = dot(rayDir, sunDir);
    
            float miePhaseValue = getMiePhase(cosTheta);
            float rayleighPhaseValue = getRayleighPhase(-cosTheta);
            
            vec3 lum = vec3(0.0), lumFactor = vec3(0.0), transmittance = vec3(1.0);
            float t = 0.0;
            for (float stepI = 0.0; stepI < mulScattSteps; stepI += 1.0) {
                float newT = ((stepI + 0.3)/mulScattSteps)*tMax;
                float dt = newT - t;
                t = newT;

                vec3 newPos = pos + t*rayDir;

                vec3 rayleighScattering, extinction;
                float mieScattering;
                getScatteringValues(newPos, rayleighScattering, mieScattering, extinction);

                vec3 sampleTransmittance = exp(-dt*extinction);
                
                // Integrate within each segment.
                vec3 scatteringNoPhase = rayleighScattering + mieScattering;
                vec3 scatteringF = (scatteringNoPhase - scatteringNoPhase * sampleTransmittance) / extinction;
                lumFactor += transmittance*scatteringF;
                
                // This is slightly different from the paper, but I think the paper has a mistake?
                // In equation (6), I think S(x,w_s) should be S(x-tv,w_s).
                vec3 sunTransmittance = getValFromTLUT(u_transmittance, tLUTRes, newPos, sunDir);

                vec3 rayleighInScattering = rayleighScattering*rayleighPhaseValue;
                float mieInScattering = mieScattering*miePhaseValue;
                vec3 inScattering = (rayleighInScattering + mieInScattering)*sunTransmittance;

                // Integrated scattering within path segment.
                vec3 scatteringIntegral = (inScattering - inScattering * sampleTransmittance) / extinction;

                lum += scatteringIntegral*transmittance;
                transmittance *= sampleTransmittance;
            }
            
            if (groundDist > 0.0) {
                vec3 hitPos = pos + groundDist*rayDir;
                if (dot(pos, sunDir) > 0.0) {
                    hitPos = normalize(hitPos)*groundRadiusMM;
                    lum += transmittance*groundAlbedo*getValFromTLUT(u_transmittance, tLUTRes, hitPos, sunDir);
                }
            }
            
            fms += lumFactor*invSamples;
            lumTotal += lum*invSamples;
        }
    }
}

out vec4 fragColor;

void main()
{
    float u = clamp(gl_FragCoord.x, 0.0, msLUTRes.x-1.0)/msLUTRes.x;
    float v = clamp(gl_FragCoord.y, 0.0, msLUTRes.y-1.0)/msLUTRes.y;
    
    float sunCosTheta = 2.0*u - 1.0;
    float sunTheta = safeacos(sunCosTheta);
    float height = mix(groundRadiusMM, atmosphereRadiusMM, v);
    
    vec3 pos = vec3(0.0, height, 0.0); 
    vec3 sunDir = normalize(vec3(0.0, sunCosTheta, -sin(sunTheta)));
    
    vec3 lum, f_ms;
    getMulScattValues(pos, sunDir, lum, f_ms);
    
    // Equation 10 from the paper.
    vec3 psi = lum  / (1.0 - f_ms); 
    fragColor = vec4(psi, 1.0);
}
"""

Sky.skyview_fragment_shader = Sky.common_fragment_shader + """
// Buffer C calculates the actual sky-view! It's a lat-long map (or maybe altitude-azimuth is the better term),
// but the latitude/altitude is non-linear to get more resolution near the horizon.

uniform sampler2D u_transmittance;
uniform sampler2D u_scattering;

const int numScatteringSteps = 32;
vec3 raymarchScattering(vec3 pos, 
                              vec3 rayDir, 
                              vec3 sunDir,
                              float tMax,
                              float numSteps) {
    float cosTheta = dot(rayDir, sunDir);
    
	float miePhaseValue = getMiePhase(cosTheta);
	float rayleighPhaseValue = getRayleighPhase(-cosTheta);
    
    vec3 lum = vec3(0.0);
    vec3 transmittance = vec3(1.0);
    float t = 0.0;
    for (float i = 0.0; i < numSteps; i += 1.0) {
        float newT = ((i + 0.3)/numSteps)*tMax;
        float dt = newT - t;
        t = newT;
        
        vec3 newPos = pos + t*rayDir;
        
        vec3 rayleighScattering, extinction;
        float mieScattering;
        getScatteringValues(newPos, rayleighScattering, mieScattering, extinction);
        
        vec3 sampleTransmittance = exp(-dt*extinction);

        vec3 sunTransmittance = getValFromTLUT(u_transmittance, tLUTRes.xy, newPos, sunDir);
        vec3 psiMS = getValFromMultiScattLUT(u_scattering, msLUTRes.xy, newPos, sunDir);
        
        vec3 rayleighInScattering = rayleighScattering*(rayleighPhaseValue*sunTransmittance + psiMS);
        vec3 mieInScattering = mieScattering*(miePhaseValue*sunTransmittance + psiMS);
        vec3 inScattering = (rayleighInScattering + mieInScattering);

        // Integrated scattering within path segment.
        vec3 scatteringIntegral = (inScattering - inScattering * sampleTransmittance) / extinction;

        lum += scatteringIntegral*transmittance;
        
        transmittance *= sampleTransmittance;
    }
    return lum;
}
out vec4 fragColor;
void main()
{
    float u = clamp(gl_FragCoord.x, 0.0, skyLUTRes.x-1.0)/skyLUTRes.x;
    float v = clamp(gl_FragCoord.y, 0.0, skyLUTRes.y-1.0)/skyLUTRes.y;
    
    float azimuthAngle = (u - 0.5)*2.0*PI;
    // Non-linear mapping of altitude. See Section 5.3 of the paper.
    float adjV;
    if (v < 0.5) {
		float coord = 1.0 - 2.0*v;
		adjV = -coord*coord;
	} else {
		float coord = v*2.0 - 1.0;
		adjV = coord*coord;
	}
    
    float height = length(viewPos);
    vec3 up = viewPos / height;
    float horizonAngle = safeacos(sqrt(height * height - groundRadiusMM * groundRadiusMM) / height) - 0.5*PI;
    float altitudeAngle = adjV*0.5*PI - horizonAngle;
    
    float cosAltitude = cos(altitudeAngle);
    vec3 rayDir = vec3(cosAltitude*sin(azimuthAngle), sin(altitudeAngle), -cosAltitude*cos(azimuthAngle));
    
    float sunAltitude = (0.5*PI) - acos(dot(getSunDir(u_time), up));
    vec3 sunDir = vec3(0.0, sin(sunAltitude), -cos(sunAltitude));
    
    float atmoDist = rayIntersectSphere(viewPos, rayDir, atmosphereRadiusMM);
    float groundDist = rayIntersectSphere(viewPos, rayDir, groundRadiusMM);
    float tMax = (groundDist < 0.0) ? atmoDist : groundDist;
    vec3 lum = raymarchScattering(viewPos, rayDir, sunDir, tMax, float(numScatteringSteps));
    fragColor = vec4(lum, 1.0);
}
"""

Sky.common_sky_render_fragment_shader = Sky.common_fragment_shader + """
/*
 * Partial implementation of
 *    "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"
 *    by Sébastien Hillaire (2020).
 * Very much referenced and copied Sébastien's provided code: 
 *    https://github.com/sebh/UnrealEngineSkyAtmosphere
 *
 * This basically implements the generation of a sky-view LUT, so it doesn't
 * include aerial perspective. It only works for views inside the atmosphere,
 * because the code assumes that the ray-marching starts at the camera position.
 * For a planetary view you'd want to check that and you might march from, e.g.
 * the edge of the atmosphere to the ground (rather than the camera position
 * to either the ground or edge of the atmosphere).
 *
 * Also want to cite: 
 *    https://wwviewer.shadertoy.com/view/tdSXzD
 * Used the jodieReinhardTonemap from there, but that also made
 * me realize that the paper switched the Mie and Rayleigh height densities
 * (which was confirmed after reading Sébastien's code more closely).
 */

/*
 * Final output basically looks up the value from the skyLUT, and then adds a sun on top,
 * does some tonemapping.
 */
 
uniform sampler2D u_transmittance;
uniform sampler2D u_skyview;
 
vec3 getValFromSkyLUT(vec3 rayDir, vec3 sunDir) {
    float height = length(viewPos);
    vec3 up = viewPos / height;
    
    float horizonAngle = safeacos(sqrt(height * height - groundRadiusMM * groundRadiusMM) / height);
    float altitudeAngle = horizonAngle - acos(dot(rayDir, up)); // Between -PI/2 and PI/2
    float azimuthAngle; // Between 0 and 2*PI
    if (abs(altitudeAngle) > (0.5*PI - .0001)) {
        // Looking nearly straight up or down.
        azimuthAngle = 0.0;
    } else {
        vec3 right = cross(sunDir, up);
        vec3 forward = cross(up, right);
        
        vec3 projectedDir = normalize(rayDir - up*(dot(rayDir, up)));
        float sinTheta = dot(projectedDir, right);
        float cosTheta = dot(projectedDir, forward);
        azimuthAngle = atan(sinTheta, cosTheta) + PI;
    }
    
    // Non-linear mapping of altitude angle. See Section 5.3 of the paper.
    float v = 0.5 + 0.5*sign(altitudeAngle)*sqrt(abs(altitudeAngle)*2.0/PI);
    vec2 uv = vec2(azimuthAngle / (2.0*PI), v);
   // uv *= skyLUTRes;
   // uv /= iChannelResolution[1].xy;
    
    return texture(u_skyview, uv).rgb;
}

vec3 jodieReinhardTonemap(vec3 c){
    // From: https://wwviewer.shadertoy.com/view/tdSXzD
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 tc = c / (c + 1.0);
    return mix(c / (l + 1.0), tc, tc);
}

vec3 sunWithBloom(vec3 rayDir, vec3 sunDir) {
    const float sunSolidAngle = 0.53*PI/180.0;
    const float minSunCosTheta = cos(sunSolidAngle);

    float cosTheta = dot(rayDir, sunDir);
    if (cosTheta >= minSunCosTheta) return vec3(1.0);
    
    float offset = minSunCosTheta - cosTheta;
    float gaussianBloom = exp(-offset*50000.0)*0.5;
    float invBloom = 1.0/(0.02 + offset*300.0)*0.01;
    return vec3(gaussianBloom+invBloom);
}

vec3 createRay(vec2 px, mat4 PInv, mat4 VInv)
{
  
    // convert pixel to NDS
    // [0,1] -> [-1,1]
    vec2 pxNDS = px*2. - 1.;

    // choose an arbitrary point in the viewing volume
    // z = -1 equals a point on the near plane, i.e. the screen
    vec3 pointNDS = vec3(pxNDS, -1.);

    // as this is in homogenous space, add the last homogenous coordinate
    vec4 pointNDSH = vec4(pointNDS, 1.0);
    // transform by inverse projection to get the point in view space
    vec4 dirEye = pointNDSH * PInv;

    // since the camera is at the origin in view space by definition,
    // the current point is already the correct direction 
    // (dir(0,P) = P - 0 = P as a direction, an infinite point,
    // the homogenous component becomes 0 the scaling done by the 
    // w-division is not of interest, as the direction in xyz will 
    // stay the same and we can just normalize it later
    dirEye.w = 0.;

    // compute world ray direction by multiplying the inverse view matrix
    vec3 dirWorld = (dirEye * VInv).xyz;

    // now normalize direction
    return normalize(dirWorld); 
}
"""

Sky.diffuse_important_sampling_fragment_shader = Sky.common_sky_render_fragment_shader + """
out vec4 outColor;
uniform int u_samples;

mat3 getNormalSpace(in vec3 normal) {
   vec3 someVec = vec3(1.0, 0.0, 0.0);
   float dd = dot(someVec, normal);
   vec3 tangent = vec3(0.0, 1.0, 0.0);
   if(abs(dd) > 1e-8) {
     tangent = normalize(cross(someVec, normal));
   }
   vec3 bitangent = cross(normal, tangent);
   return mat3(tangent, bitangent, normal);
}

// from http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// Hacker's Delight, Henry S. Warren, 2001
float radicalInverse(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint n, uint N) {
  return vec2(float(n) / float(N), radicalInverse(n));
}

// The origin of the random2 function is probably the paper:
// 'On generating random numbers, with help of y= [(a+x)sin(bx)] mod 1'
// viewer.J.J. Rey, 22nd European Meeting of Statisticians and the
// 7th Vilnius Conference on Probability Theory and Mathematical Statistics, August 1998
// as discussed here:
// https://stackoverfloviewer.com/questions/12964279/whats-the-origin-of-this-glsl-rand-one-liner
float random2(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

vec3 sun(vec3 rayDir, vec3 sunDir) {
    const float sunSolidAngle = 5.0*PI/180.0;
    const float minSunCosTheta = cos(sunSolidAngle);

    float cosTheta = dot(rayDir, sunDir);
    if (cosTheta >= minSunCosTheta) return vec3(1.0);
    
    return vec3(0);
}

vec3 luminosity(vec3 rayDir, vec3 sunDir) {
    return getValFromSkyLUT(rayDir, sunDir);
}

void main() {
    vec3 sunDir = getSunDir(u_time);
  
  vec2 coord = (gl_FragCoord.xy / vec2(128.0, 64.0));
  float thetaN = PI * (1.0-coord.y);
  float phiN = 2.0 * PI * (coord.x) + (PI/2.0);
  vec3 normal = vec3(sin(thetaN) * cos(phiN), cos(thetaN), sin(thetaN) * sin(phiN));
  mat3 normalSpace = getNormalSpace(normal);

  vec3 result = vec3(0.0);

  uint N = uint(u_samples);
  
  float r = random2(coord);
  
  for(uint n = 1u; n <= N; n++) {
      //vec2 p = hammersley(n, N);
      vec2 p = mod(hammersley(n, N) + r, 1.0);
      float theta = asin(sqrt(p.y));
      float phi = 2.0 * PI * p.x;
      vec3 pos = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
      vec3 posGlob = normalSpace * pos;
      vec3 radiance = luminosity(posGlob,sunDir).rgb;
      result += radiance;
  }
  result = result / float(u_samples);
  
   result *= 20.0;
    result = pow(result, vec3(1.3));
  
  outColor.rgb = result;
  outColor.a = 1.0;
}
"""

Sky.brdf_integration_fragment_shader = """#version 300 es
precision highp float;

out vec4 outColor;
uniform int u_samples; // description="number of samples" 
const float PI = 3.1415926535897932384626433832795;

// from http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// Hacker's Delight, Henry S. Warren, 2001
float radicalInverse(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint n, uint N) {
  return vec2(float(n) / float(N), radicalInverse(n));
}

float G1_GGX_Schlick(float NdotV, float roughness) {
  float r = roughness; // original
  //float r = 0.5 + 0.5 * roughness; // Disney remapping
  float k = (r * r) / 2.0;
  float denom = NdotV * (1.0 - k) + k;
  return NdotV / denom;
}

float G_Smith(float NoV, float NoL, float roughness) {
  float g1_l = G1_GGX_Schlick(NoL, roughness);
  float g1_v = G1_GGX_Schlick(NoV, roughness);
  return g1_l * g1_v;
}

// adapted from "Real Shading in Unreal Engine 4", Brian Karis, Epic Games
// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
vec2 integrateBRDF(float roughness, float NoV) {
  vec3 V;
  V.x = sqrt(1.0 - NoV * NoV); // sin
  V.y = 0.0;
  V.z = NoV; // cos
  vec2 result = vec2(0.0);
  uint sampleCount = uint(u_samples);
  for(uint n = 1u; n <= sampleCount; n++) {
    vec2 p = hammersley(n, sampleCount);
    float a = roughness * roughness;
    float theta = acos(sqrt((1.0 - p.y) / (1.0 + (a * a - 1.0) * p.y)));
    float phi = 2.0 * PI * p.x;
    // sampled h direction in normal space
    vec3 H = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    vec3 L = 2.0 * dot(V, H) * H - V;

    // because N = vec3(0.0, 0.0, 1.0) follows
    float NoL = clamp(L.z, 0.0, 1.0);
    float NoH = clamp(H.z, 0.0, 1.0);
    float VoH = clamp(dot(V, H), 0.0, 1.0);
    if(NoL > 0.0) {
      float G = G_Smith(NoV, NoL, roughness);
      float G_Vis = G * VoH / (NoH * NoV);
      float Fc = pow(1.0 - VoH, 5.0);
      result.x += (1.0 - Fc) * G_Vis;
      result.y += Fc * G_Vis;
    }
  }
  result = result / float(sampleCount);
  return result;
}

void main() {
  vec2 r = integrateBRDF(gl_FragCoord.y/128.0, gl_FragCoord.x/128.0);
  outColor = vec4(r, 0.0, 1.0);
}

"""

Sky.brdf_importance_sampling_fragment_shader = Sky.common_sky_render_fragment_shader + """
out vec4 outColor;
uniform int u_samples;
uniform float u_roughness; // description="roughness in range [0.0, 1.0]" 
uniform vec2 u_texresolution;

mat3 getNormalSpace(in vec3 normal) {
   vec3 someVec = vec3(1.0, 0.0, 0.0);
   float dd = dot(someVec, normal);
   vec3 tangent = vec3(0.0, 1.0, 0.0);
   if(abs(dd) > 1e-8) {
     tangent = normalize(cross(someVec, normal));
   }
   vec3 bitangent = cross(normal, tangent);
   return mat3(tangent, bitangent, normal);
}

// from http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// Hacker's Delight, Henry S. Warren, 2001
float radicalInverse(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint n, uint N) {
  return vec2(float(n) / float(N), radicalInverse(n));
}

// The origin of the random2 function is probably the paper:
// 'On generating random numbers, with help of y= [(a+x)sin(bx)] mod 1'
// viewer.J.J. Rey, 22nd European Meeting of Statisticians and the
// 7th Vilnius Conference on Probability Theory and Mathematical Statistics, August 1998
// as discussed here:
// https://stackoverfloviewer.com/questions/12964279/whats-the-origin-of-this-glsl-rand-one-liner
float random2(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

vec3 sun(vec3 rayDir, vec3 sunDir) {
    const float sunSolidAngle = 5.0*PI/180.0;
    const float minSunCosTheta = cos(sunSolidAngle);

    float cosTheta = dot(rayDir, sunDir);
    if (cosTheta >= minSunCosTheta) return vec3(1.0);
    
    return vec3(0);
}

vec3 luminosity(vec3 rayDir, vec3 sunDir) {
    return getValFromSkyLUT(rayDir, sunDir);
}

void main() {
  
  vec2 coord = (gl_FragCoord.xy / u_texresolution);
  float thetaN = PI * (1.0-coord.y);
  float phiN = 2.0 * PI * (coord.x) + (PI/2.0);
  vec3 R = vec3(sin(thetaN) * cos(phiN), cos(thetaN), sin(thetaN) * sin(phiN));
  
  vec3 sunDir = getSunDir(u_time);

  vec3 N = R;
  vec3 V = R;
  uint sampleCount = uint(u_samples);
  float r = random2(coord);
  float r2 = random2(coord*10.33f);
  mat3 normalSpace = getNormalSpace(N);
  
  float totalWeight = 0.0;
  vec3 result = vec3(0.0);
  
  for(uint n = 1u; n <= sampleCount; n++) {
    //vec2 p = hammersley(n, sampleCount);
    vec2 p = mod(hammersley(n, sampleCount) + r, 1.0);
    float a = u_roughness * u_roughness;
    float theta = acos(sqrt((1.0 - p.y) / (1.0 + (a * a - 1.0) * p.y)));
    float phi = 2.0 * PI * (p.x + r2*0.1);
    // sampled h direction in normal space
    vec3 Hn = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    // sampled h direction in world space
    vec3 H = normalSpace * Hn;
    vec3 L = 2.0 * dot(V, H) * H - V;
    
    float NoL = max(dot(N, L), 0.0);
    if( NoL > 0.0 ) {
      vec3 radiance = luminosity(L,sunDir).rgb;
      result += radiance * NoL;
      totalWeight +=NoL;
    }
  }
  result = result / totalWeight;
  result *= 20.0;
  result = pow(result, vec3(1.3));
  
  outColor.rgb = result;
  outColor.a = 1.0;
}
"""