#include <metal_stdlib>
using namespace metal;

struct VSIn {
    float2 pos [[attribute(0)]];
};

struct VSOut {
    float4 position [[position]];
    float2 uv;
    float2 l;
    float2 r;
    float2 t;
    float2 b;
};

struct Uniforms {
    float2 texelSize;
};

vertex VSOut fullscreenVS(VSIn in [[stage_in]], constant Uniforms& u [[buffer(1)]]) {
    VSOut out;
    out.position = float4(in.pos, 0.0, 1.0);
    out.uv = in.pos * 0.5 + 0.5;
    out.l = out.uv - float2(u.texelSize.x, 0.0);
    out.r = out.uv + float2(u.texelSize.x, 0.0);
    out.t = out.uv + float2(0.0, u.texelSize.y);
    out.b = out.uv - float2(0.0, u.texelSize.y);
    return out;
}

fragment float4 copyFS(VSOut in [[stage_in]],
                       texture2d<float> src [[texture(0)]],
                       sampler s [[sampler(0)]]) {
    return float4(src.sample(s, in.uv));
}

// Simple color fill
struct ColorUniform { float4 color; };
fragment float4 colorFS(VSOut in [[stage_in]], constant ColorUniform& cu [[buffer(0)]]) {
    return cu.color;
}

// Advection fragment shader: advects uSource by uVelocity into the render target
struct AdvectionUniforms {
    float2 texelSize;   // of velocity/source textures in UV space (unused when sampling with normalized UVs)
    float dt;
    float dissipation;
};

fragment float4 advectionFS(VSOut in [[stage_in]],
                            constant AdvectionUniforms& au [[buffer(0)]],
                            texture2d<float> velocityTex [[texture(0)]],
                            texture2d<float> sourceTex [[texture(1)]],
                            sampler linearSamp [[sampler(0)]]) {
    // Backtrace in UV space
    float2 v = velocityTex.sample(linearSamp, in.uv).xy;
    float2 coord = in.uv - au.dt * v * au.texelSize; // texelSize here encodes 1/size in pixels for scale
    float4 result = sourceTex.sample(linearSamp, coord);
    float decay = 1.0 + au.dissipation * au.dt;
    return result / decay;
}

// Splat: add a gaussian blob to the source texture (read) and output the result
struct SplatUniforms {
    float aspectRatio;
    float2 point; // uv
    float radius;
    float3 color; // for velocity: (dx,dy,0), for dye: (r,g,b)
};

fragment float4 splatFS(VSOut in [[stage_in]],
                        constant SplatUniforms& su [[buffer(0)]],
                        texture2d<float> src [[texture(0)]],
                        sampler s [[sampler(0)]]) {
    float2 p = in.uv - su.point;
    p.x *= su.aspectRatio;
    float3 splat = exp(-dot(p,p) / su.radius) * su.color;
    float3 base = src.sample(s, in.uv).xyz;
    return float4(base + splat, 1.0);
}

// Divergence of velocity (R-L + T-B)/2 with boundary handling
fragment float4 divergenceFS(VSOut in [[stage_in]],
                             texture2d<float> vel [[texture(0)]],
                             sampler s [[sampler(0)]]) {
    float L = vel.sample(s, in.l).x;
    float R = vel.sample(s, in.r).x;
    float T = vel.sample(s, in.t).y;
    float B = vel.sample(s, in.b).y;
    float2 C = vel.sample(s, in.uv).xy;
    if (in.l.x < 0.0)  L = -C.x;
    if (in.r.x > 1.0)  R = -C.x;
    if (in.t.y > 1.0)  T = -C.y;
    if (in.b.y < 0.0)  B = -C.y;
    float div = 0.5 * (R - L + T - B);
    return float4(div, 0.0, 0.0, 1.0);
}

// Curl (vorticity scalar)
fragment float4 curlFS(VSOut in [[stage_in]],
                       texture2d<float> vel [[texture(0)]],
                       sampler s [[sampler(0)]]) {
    float L = vel.sample(s, in.l).y;
    float R = vel.sample(s, in.r).y;
    float T = vel.sample(s, in.t).x;
    float B = vel.sample(s, in.b).x;
    float vort = R - L - T + B;
    return float4(0.5 * vort, 0.0, 0.0, 1.0);
}

// Vorticity confinement adds force = curl * normalized gradient of |curl|
struct VorticityUniforms { float curl; float dt; };
fragment float4 vorticityFS(VSOut in [[stage_in]],
                            constant VorticityUniforms& vu [[buffer(0)]],
                            texture2d<float> vel [[texture(0)]],
                            texture2d<float> curlTex [[texture(1)]],
                            sampler s [[sampler(0)]]) {
    float L = curlTex.sample(s, in.l).x;
    float R = curlTex.sample(s, in.r).x;
    float T = curlTex.sample(s, in.t).x;
    float B = curlTex.sample(s, in.b).x;
    float C = curlTex.sample(s, in.uv).x;
    float2 force = 0.5 * float2(fabs(T) - fabs(B), fabs(R) - fabs(L));
    float len = length(force) + 1e-4;
    force = force / len;
    force *= vu.curl * C;
    force.y *= -1.0;
    float2 v = vel.sample(s, in.uv).xy;
    v += force * vu.dt;
    v = clamp(v, float2(-1000.0), float2(1000.0));
    return float4(v, 0.0, 1.0);
}

// Pressure Jacobi iteration
fragment float4 pressureFS(VSOut in [[stage_in]],
                           texture2d<float> pressure [[texture(0)]],
                           texture2d<float> divergence [[texture(1)]],
                           sampler s [[sampler(0)]]) {
    float L = pressure.sample(s, in.l).x;
    float R = pressure.sample(s, in.r).x;
    float T = pressure.sample(s, in.t).x;
    float B = pressure.sample(s, in.b).x;
    float div = divergence.sample(s, in.uv).x;
    float p = (L + R + B + T - div) * 0.25;
    return float4(p, 0.0, 0.0, 1.0);
}

// Gradient subtract (make velocity divergence-free)
fragment float4 gradientSubtractFS(VSOut in [[stage_in]],
                                   texture2d<float> pressure [[texture(0)]],
                                   texture2d<float> vel [[texture(1)]],
                                   sampler s [[sampler(0)]]) {
    float L = pressure.sample(s, in.l).x;
    float R = pressure.sample(s, in.r).x;
    float T = pressure.sample(s, in.t).x;
    float B = pressure.sample(s, in.b).x;
    float2 v = vel.sample(s, in.uv).xy;
    v -= float2(R - L, T - B);
    return float4(v, 0.0, 1.0);
}

// Clear multiply pass (value * existing)
struct ClearUniform { float value; };
fragment float4 clearFS(VSOut in [[stage_in]],
                        constant ClearUniform& cu [[buffer(0)]],
                        texture2d<float> src [[texture(0)]],
                        sampler s [[sampler(0)]]) {
    return cu.value * src.sample(s, in.uv);
}
