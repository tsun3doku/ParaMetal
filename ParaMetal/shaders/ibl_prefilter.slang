struct PushConstants {
    uint faceSize;
    uint mipLevel;
    uint mipCount;
    uint envMapSize;
};

[[vk::binding(0)]] TextureCube<float4> environmentMap;
[[vk::binding(1)]] SamplerState environmentSampler;
[[vk::binding(2)]]
[[vk::image_format("rgba16f")]]
RWTexture2DArray<float4> prefilteredCube;

[[vk::push_constant]] ConstantBuffer<PushConstants> pushConstants;

static const float PI = 3.14159265359;

float3 cubeDirection(uint face, float2 uv) {
    float2 p = uv * 2.0 - 1.0;
    if (face == 0) return normalize(float3(1.0, -p.y, -p.x));
    if (face == 1) return normalize(float3(-1.0, -p.y, p.x));
    if (face == 2) return normalize(float3(p.x, 1.0, p.y));
    if (face == 3) return normalize(float3(p.x, -1.0, -p.y));
    if (face == 4) return normalize(float3(p.x, -p.y, 1.0));
    return normalize(float3(-p.x, -p.y, -1.0));
}

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
    float3 h = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    float3 up = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, n));
    float3 bitangent = cross(n, tangent);
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

float dGGX(float dotNH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = dotNH * dotNH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    if (threadId.x >= pushConstants.faceSize || threadId.y >= pushConstants.faceSize || threadId.z >= 6) {
        return;
    }

    float2 uv = (float2(threadId.xy) + 0.5) / float(pushConstants.faceSize);
    float3 n = cubeDirection(threadId.z, uv);
    float3 r = n;
    float3 v = r;
    float roughness = pushConstants.mipCount > 1
        ? float(pushConstants.mipLevel) / float(pushConstants.mipCount - 1)
        : 0.0;

    const uint sampleCount = 512;
    const float envMapDim = float(pushConstants.envMapSize);
    const float omegaP = 4.0 * PI / (6.0 * envMapDim * envMapDim);
    float3 prefilteredColor = float3(0.0);
    float totalWeight = 0.0;

    for (uint i = 0; i < sampleCount; ++i) {
        float3 h = importanceSampleGGX(hammersley(i, sampleCount), n, roughness);
        float3 l = normalize(2.0 * dot(v, h) * h - v);
        float nDotL = max(dot(n, l), 0.0);
        if (nDotL > 0.0) {
            float nDotH = max(dot(n, h), 0.0);
            float vDotH = max(dot(v, h), 0.0);
            float pdf = dGGX(nDotH, roughness) * nDotH / (4.0 * vDotH) + 0.0001;
            float omegaS = 1.0 / (float(sampleCount) * pdf);
            float mipLevel = roughness == 0.0 ? 0.0 : max(0.5 * log2(omegaS / omegaP) + 1.0, 0.0);
            prefilteredColor += environmentMap.SampleLevel(environmentSampler, l, mipLevel).rgb * nDotL;
            totalWeight += nDotL;
        }
    }

    prefilteredColor /= max(totalWeight, 0.0001);
    prefilteredCube[uint3(threadId.xy, threadId.z)] = float4(prefilteredColor, 1.0);
}