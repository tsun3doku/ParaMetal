struct PushConstants {
    uint faceSize;
};

[[vk::binding(0)]] Texture2D<float4> sourceEquirect;
[[vk::binding(1)]] SamplerState sourceSampler;
[[vk::binding(2)]]
[[vk::image_format("rgba16f")]]
RWTexture2DArray<float4> outputCube;

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

float2 equirectUv(float3 dir) {
    float phi = atan2(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    return float2(phi / (2.0 * PI) + 0.5, 0.5 - theta / PI);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    if (threadId.x >= pushConstants.faceSize || threadId.y >= pushConstants.faceSize || threadId.z >= 6) {
        return;
    }

    float2 uv = (float2(threadId.xy) + 0.5) / float(pushConstants.faceSize);
    float3 dir = cubeDirection(threadId.z, uv);
    float3 color = sourceEquirect.SampleLevel(sourceSampler, equirectUv(dir), 0.0).rgb;
    outputCube[uint3(threadId.xy, threadId.z)] = float4(color, 1.0);
}
