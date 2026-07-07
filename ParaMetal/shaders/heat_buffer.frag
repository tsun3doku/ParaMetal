#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "heat_palette.glsl"

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) in vec3 fragBaryCoord;
layout(location = 5) in vec2 fragIntrinsicCoord;

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    vec4 sourceParams; // x = temperature, y = isSource
} push;

// GBuffer outputs
layout(location = 0) out vec4 gAlbedo;
layout(location = 1) out vec4 gMaterial;

// Texture buffers for supporting halfedge data 
layout(set = 0, binding = 1) uniform isamplerBuffer S;  // Supporting halfedge per input halfedge
layout(set = 0, binding = 2) uniform samplerBuffer A;   // Supporting angle per input halfedge
layout(set = 0, binding = 3) uniform isamplerBuffer H;  // Intrinsic halfedge [origin, edge, face, next]
layout(set = 0, binding = 4) uniform isamplerBuffer E;  // Intrinsic edge [he0, he1]
layout(set = 0, binding = 5) uniform isamplerBuffer T;  // Intrinsic triangle [halfedge]
layout(set = 0, binding = 6) uniform samplerBuffer L;   // Intrinsic edge lengths

// Texture buffers for input mesh data
layout(set = 0, binding = 7) uniform isamplerBuffer H_input;
layout(set = 0, binding = 8) uniform isamplerBuffer E_input;
layout(set = 0, binding = 9) uniform isamplerBuffer T_input;
layout(set = 0, binding = 10) uniform samplerBuffer L_input;

struct SurfacePoint {
    vec3 position;
    float temperature;
    vec3 normal;
    float area;
    vec4 color;
};

// Heat temperature buffer 
layout(set = 0, binding = 11) readonly buffer HeatColors {
    SurfacePoint surfacePoints[];
} heatColors;

const float PI = 3.14159265359;
const int IMAX = 128;
const vec3 errorColor = vec3(1.0, 0.0, 1.0);
const float HEAT_ROUGHNESS = 0.05;
const float HEAT_METALNESS = 0.0;
const float HEAT_LIGHTING_MIX = 0.1;
const float HEAT_CONTOUR_SPACING = 20.0;

vec4 heatMaterial() {
    return vec4(HEAT_ROUGHNESS, HEAT_METALNESS, HEAT_LIGHTING_MIX, 0.0);
}

void writeHeatSurface(vec3 albedo) {
    gAlbedo = vec4(albedo, 1.0);
    gMaterial = heatMaterial();
}

vec3 heatColorFromTemperature(float temperature) {
    float normalized = clamp(temperature / TEMPERATURE_SCALE, 0.0, 1.0);
    return temperatureToColor(normalized);
}

float heatContourMask(float temperature) {
    float d = mod(temperature, HEAT_CONTOUR_SPACING);
    float distToContour = min(d, HEAT_CONTOUR_SPACING - d);
    float pixelGrad = length(vec2(dFdx(temperature), dFdy(temperature)));
    if (pixelGrad <= 1e-3) {
        return 1.0;
    }

    float contourHalfWidth = pixelGrad * 1.5;
    return smoothstep(0.0, contourHalfWidth, distToContour);
}

vec3 heatAlbedo(float temperature) {
    vec3 heatColor = heatColorFromTemperature(temperature);
    return mix(vec3(0.0), heatColor, heatContourMask(temperature));
}

bool ccw(vec2 p, vec2 q, vec2 r) {
    float det = (p.x - r.x) * (q.y - r.y) - (p.y - r.y) * (q.x - r.x);
    return det >= 0.0;
}

bool crossing(vec2 v_a, vec2 v_b, vec2 p) {
    vec2 origin = vec2(0.0, 0.0);
    bool t0 = ccw(origin, p, v_a);
    bool t1 = ccw(origin, p, v_b);
    bool u0 = ccw(v_a, v_b, origin);
    bool u1 = ccw(v_a, v_b, p);
    return ((t0 && !t1) || (!t0 && t1)) && ((u0 && !u1) || (!u0 && u1));
}

float area(vec2 p, vec2 q, vec2 r) {
    float det = (q.x * r.y + p.x * q.y + p.y * r.x) - (q.x * p.y + r.x * q.y + r.y * p.x);
    return det / 2.0;
}

vec3 clampBarycentric(vec3 b) {
    vec3 clamped = max(b, vec3(0.0));
    float sum = clamped.x + clamped.y + clamped.z;
    if (sum < 1e-8) {
        return vec3(1.0, 0.0, 0.0);
    }
    return clamped / sum;
}

int mate(int he) {
    if (he < 0) return -1;

    ivec4 halfedge = texelFetch(H, he);
    int edgeIdx = halfedge.g;
    if (edgeIdx < 0) return -1;
    
    ivec2 edge = texelFetch(E, edgeIdx).rg;
    return (edge.r == he) ? edge.g : edge.r;
}

bool loadInputTriangleChart(int inputTri, out ivec3 faceHEs, out vec2 triCoords[3]) {
    int h0 = texelFetch(T_input, inputTri).r;
    if (h0 < 0) return false;

    int h1 = texelFetch(H_input, h0).a;
    if (h1 < 0) return false;

    int h2 = texelFetch(H_input, h1).a;
    if (h2 < 0) return false;

    faceHEs = ivec3(h0, h1, h2);

    int e0 = texelFetch(H_input, h0).g;
    int e1 = texelFetch(H_input, h1).g;
    int e2 = texelFetch(H_input, h2).g;
    if (e0 < 0 || e1 < 0 || e2 < 0) return false;

    float a = texelFetch(L_input, e0).r;
    float b = texelFetch(L_input, e1).r;
    float c = texelFetch(L_input, e2).r;
    if (a <= 0.0 || b <= 0.0 || c <= 0.0) return false;

    triCoords[0] = vec2(0.0, 0.0);
    triCoords[1] = vec2(a, 0.0);
    float x = (a * a + c * c - b * b) / (2.0 * a);
    float y2 = max(c * c - x * x, 0.0);
    triCoords[2] = vec2(x, sqrt(y2));

    return true;
}

vec2 inputChartPoint(vec2 triCoords[3]) {
    return fragBaryCoord.x * triCoords[0] + fragBaryCoord.y * triCoords[1] + fragBaryCoord.z * triCoords[2];
}

bool findIntrinsicTriangleFromSeed(vec2 p, int inputHe, vec2 seedOrigin, float seedPhi, out int intrinsicTri, out vec3 baryCoords, out int failureCode) {
    int h0 = texelFetch(S, inputHe).r;
    float phi0 = seedPhi + texelFetch(A, inputHe).r;
    if (h0 < 0) return false;
    ivec4 he0 = texelFetch(H, h0);
    int e0 = he0.g;
    if (e0 < 0) return false;
    float l0 = texelFetch(L, e0).r;
    vec2 v0 = seedOrigin;
    vec2 v1 = v0 + vec2(l0 * cos(phi0), l0 * sin(phi0));
    for (int iter = 0; iter < IMAX; iter++) {
        ivec4 halfedge = texelFetch(H, h0);

        int t = halfedge.b;
        int h1 = halfedge.a;
        if (h1 < 0 || t < 0) return false;

        ivec4 he1 = texelFetch(H, h1);
        int h2 = he1.a;
        if (h2 < 0) break;

        int e1 = he1.g;
        ivec4 he2 = texelFetch(H, h2);
        int e2 = he2.g;
        if (e1 < 0 || e2 < 0) break;

        float l1 = texelFetch(L, e1).r;
        float l2 = texelFetch(L, e2).r;

        float alpha = acos(clamp((l0*l0 + l1*l1 - l2*l2) / (2.0*l0*l1), -1.0, 1.0));
        float phi1 = phi0 + PI - alpha;
        vec2 v2 = vec2(v1.x + l1*cos(phi1), v1.y + l1*sin(phi1));

        float a = area(v0, v1, v2);
        if (abs(a) < 1e-10) break;

        bool out_v1v2 = !ccw(v1, v2, p);
        if (out_v1v2 && crossing(v1, v2, p)) {
            int m1 = mate(h1);
            if (m1 != -1) {
                v0 = v2; l0 = l1; h0 = m1; phi0 = phi1 + PI;
                continue;
            }
            break;
        } else if (out_v1v2 || !ccw(v2, v0, p)) {
            int m2 = mate(h2);
            if (m2 != -1) {
                float beta = acos(clamp((l0*l0 + l2*l2 - l1*l1) / (2.0*l0*l2), -1.0, 1.0));
                v1 = v2; l0 = l2; h0 = m2; phi0 = phi0 + beta;
                continue;
            }
            break;
        } else {
            float a12 = area(v1, v2, p);
            float a20 = area(v2, v0, p);
            float a01 = a - a12 - a20;
            vec3 localBary = clampBarycentric(vec3(a12 / a, a20 / a, a01 / a));
            
            int href = texelFetch(T, t).r;
            if (href == h0) { baryCoords.x = localBary.x; baryCoords.y = localBary.y; }
            else if (href == h1) { baryCoords.x = localBary.y; baryCoords.y = localBary.z; }
            else { baryCoords.x = localBary.z; baryCoords.y = localBary.x; }
            baryCoords.z = 1.0 - baryCoords.x - baryCoords.y;
            intrinsicTri = t;

            return true;
        }
    }
    return false;
}

int findIntrinsicTriangle(int inputTri, vec2 p, out vec3 baryCoords) {
    ivec3 faceHEs;
    vec2 triCoords[3];
    if (!loadInputTriangleChart(inputTri, faceHEs, triCoords)) return -1;
    for (int i = 0; i < 3; ++i) {
        int inputHe = faceHEs[i];

        int supportHe = texelFetch(S, inputHe).r;
        if (supportHe < 0) continue;

        vec2 edgeDir = triCoords[(i + 1) % 3] - triCoords[i];
        float seedPhi = atan(edgeDir.y, edgeDir.x);
        int intrinsicTri = -1;
        int failureCode = -1;
        vec3 candidateBary = vec3(1, 0, 0);
        
        if (findIntrinsicTriangleFromSeed(p, inputHe, triCoords[i], seedPhi, intrinsicTri, candidateBary, failureCode)) {
            baryCoords = candidateBary;
            return intrinsicTri;
        }
    }
    return -1;
}

void main() {
    if (push.sourceParams.y > 0.5) {
        writeHeatSurface(heatColorFromTemperature(push.sourceParams.x));
        return;
    }

    int inputTri = gl_PrimitiveID;
    ivec3 faceHEs;
    vec2 triCoords[3];
    if (!loadInputTriangleChart(inputTri, faceHEs, triCoords)) {
        writeHeatSurface(errorColor);
        return;
    }

    vec3 baryCoords;
    int intrinsicTri = findIntrinsicTriangle(inputTri, inputChartPoint(triCoords), baryCoords);
    if (intrinsicTri < 0) {
        writeHeatSurface(errorColor);
        return;
    }

    int href = texelFetch(T, intrinsicTri).r;
    if (href < 0) {
        writeHeatSurface(errorColor);
        return;
    }

    ivec4 he_ref = texelFetch(H, href);
    int h1_ref = he_ref.a;
    ivec4 he1_ref = texelFetch(H, h1_ref);
    int h2_ref = he1_ref.a;
    
    int v0 = he_ref.r;
    int v1 = he1_ref.r;
    int v2 = texelFetch(H, h2_ref).r;

    if (v0 < 0 || v1 < 0 || v2 < 0) {
        writeHeatSurface(errorColor);
        return;
    }
    
    float temp0 = heatColors.surfacePoints[v0].temperature;
    float temp1 = heatColors.surfacePoints[v1].temperature;
    float temp2 = heatColors.surfacePoints[v2].temperature;

    float interpolatedTemp = baryCoords.x * temp0 + baryCoords.y * temp1 + baryCoords.z * temp2;
    writeHeatSurface(heatAlbedo(interpolatedTemp));
}