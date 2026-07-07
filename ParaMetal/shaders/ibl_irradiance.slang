struct PushConstants {
    uint faceSize;
};

[[vk::binding(0)]] TextureCube<float4> environmentMap;
[[vk::binding(1)]] SamplerState environmentSampler;
[[vk::binding(2)]]
[[vk::image_format("rgba16f")]]
RWTexture2DArray<float4> irradianceCube;

[[vk::push_constant]] ConstantBuffer<PushConstants> pushConstants;

static const float PI = 3.14159265359;
static const float TWO_PI = PI * 2.0;
static const float HALF_PI = PI * 0.5;
static const float DeltaPhi = 0.025;
static const float DeltaTheta = 0.025;

float3 cubeDirection(uint face, float2 uv) {
    float2 p = uv * 2.0 - 1.0;
    if (face == 0) return normalize(float3(1.0, -p.y, -p.x));
    if (face == 1) return normalize(float3(-1.0, -p.y, p.x));
    if (face == 2) return normalize(float3(p.x, 1.0, p.y));
    if (face == 3) return normalize(float3(p.x, -1.0, -p.y));
    if (face == 4) return normalize(float3(p.x, -p.y, 1.0));
    return normalize(float3(-p.x, -p.y, -1.0));
}

void buildOrthonormalBasis(float3 n, out float3 tangent, out float3 bitangent) {
    float signN = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (signN + n.z);
    float b = n.x * n.y * a;
    tangent = float3(1.0 + signN * n.x * n.x * a, signN * b, -signN * n.x);
    bitangent = float3(b, signN + n.y * n.y * a, -n.y);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    if (threadId.x >= pushConstants.faceSize || threadId.y >= pushConstants.faceSize || threadId.z >= 6) {
        return;
    }

    float2 uv = (float2(threadId.xy) + 0.5) / float(pushConstants.faceSize);
    float3 n = cubeDirection(threadId.z, uv);
    float3 right;
    float3 up;
    buildOrthonormalBasis(n, right, up);

    float3 irradiance = float3(0.0);
    uint sampleCount = 0;

    const float sourceLod = 2.0;
    for (float phi = 0.0; phi < TWO_PI; phi += DeltaPhi) {
        for (float theta = 0.0; theta < HALF_PI; theta += DeltaTheta) {
            float3 tangentDir = cos(phi) * right + sin(phi) * up;
            float3 sampleDir = cos(theta) * n + sin(theta) * tangentDir;
            irradiance += environmentMap.SampleLevel(environmentSampler, sampleDir, sourceLod).rgb * cos(theta) * sin(theta);
            sampleCount++;
        }
    }

    irradiance = PI * irradiance / max(float(sampleCount), 1.0);
    irradianceCube[uint3(threadId.xy, threadId.z)] = float4(irradiance, 1.0);
}
