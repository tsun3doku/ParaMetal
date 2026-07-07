struct PushConstants {
    uint lutSize;
};

[[vk::binding(0)]]
[[vk::image_format("rg16f")]]
RWTexture2D<float2> brdfLut;

[[vk::push_constant]] ConstantBuffer<PushConstants> pushConstants;

static const float PI = 3.14159265359;

float radicalInverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 hammersley(uint i, uint count) {
    return float2(float(i) / float(count), radicalInverse(i));
}

float3 importanceSampleGGX(float2 xi, float3 n, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float geometrySchlickGGX(float nDotV, float roughness) {
    float a = roughness;
    float k = (a * a) / 2.0;
    return nDotV / max(nDotV * (1.0 - k) + k, 0.0001);
}

float geometrySmith(float nDotV, float nDotL, float roughness) {
    return geometrySchlickGGX(nDotV, roughness) * geometrySchlickGGX(nDotL, roughness);
}

float2 integrateBRDF(float nDotV, float roughness) {
    float3 v = float3(sqrt(max(1.0 - nDotV * nDotV, 0.0)), 0.0, nDotV);
    float a = 0.0;
    float b = 0.0;
    const uint sampleCount = 1024;

    for (uint i = 0; i < sampleCount; ++i) {
        float3 h = importanceSampleGGX(hammersley(i, sampleCount), float3(0.0, 0.0, 1.0), roughness);
        float3 l = normalize(2.0 * dot(v, h) * h - v);
        float nDotL = max(l.z, 0.0);
        float nDotH = max(h.z, 0.0);
        float vDotH = max(dot(v, h), 0.0);
        if (nDotL > 0.0) {
            float g = geometrySmith(nDotV, nDotL, roughness);
            float gVis = (g * vDotH) / max(nDotH * nDotV, 0.0001);
            float fc = pow(1.0 - vDotH, 5.0);
            a += (1.0 - fc) * gVis;
            b += fc * gVis;
        }
    }

    return float2(a, b) / float(sampleCount);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    if (threadId.x >= pushConstants.lutSize || threadId.y >= pushConstants.lutSize) {
        return;
    }

    float2 uv = (float2(threadId.xy) + 0.5) / float(pushConstants.lutSize);
    brdfLut[threadId.xy] = integrateBRDF(clamp(uv.x, 0.001, 1.0), clamp(uv.y, 0.001, 1.0));
}
