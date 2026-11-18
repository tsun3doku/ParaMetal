#include "palette_lut.slang"

struct SliceUniformBufferObject { float4x4 model; float4x4 view; float4x4 proj; float4 color; };
struct SlicePushConstants {
    int4 viewport; uint4 counts; float4 plane; float4 range; uint4 options;
};
[[vk::binding(0, 0)]] ConstantBuffer<SliceUniformBufferObject> ubo;
[[vk::binding(1, 0)]] StructuredBuffer<float4> velocity;
[[vk::binding(2, 0)]] StructuredBuffer<float4> nodeCentroids;
[[vk::binding(3, 0)]] StructuredBuffer<float4> nodeGradients;
[[vk::binding(4, 0)]] Sampler2D infernoLut;
[[vk::binding(5, 0)]] Sampler2D viridisLut;
[[vk::binding(6, 0)]] Sampler2D inferno2Lut;
[[vk::binding(7, 0)]] Sampler2D parulaLut;
[[vk::push_constant]] ConstantBuffer<SlicePushConstants> pc;

struct MeshVertex { [[vk::location(0)]] float3 position : POSITION; [[vk::location(1)]] uint fluidNodeId : FLUID_NODE_ID; };
struct FillVertexOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 canonicalPosition;
    [[vk::location(1)]] nointerpolation uint fluidNodeId;
    [[vk::location(2)]] float forwardDepth;
};
[shader("vertex")]
FillVertexOut vsMain(MeshVertex input) {
    const float4 viewPosition = mul(ubo.view, mul(ubo.model, float4(input.position, 1.0f)));
    FillVertexOut output;
    output.position = mul(ubo.proj, viewPosition);
    output.canonicalPosition = input.position;
    output.fluidNodeId = input.fluidNodeId;
    output.forwardDepth = -viewPosition.z;
    return output;
}
float3 sliceColor(float speed) {
    return velocityColor(infernoLut, viridisLut, inferno2Lut, parulaLut,
                         speed, pc.options.y, pc.range.x, pc.range.y);
}
[shader("fragment")]
float4 fsMain(FillVertexOut input) : SV_Target {
    if (pc.range.z > 0.0f && input.forwardDepth < pc.range.z) discard;
    const uint node = input.fluidNodeId;
    if (node >= pc.counts.x) discard;
    const float3 baseVelocity = velocity[node].xyz;
    const float3 delta = input.canonicalPosition - nodeCentroids[node].xyz;
    const float3 change = float3(dot(nodeGradients[node * 3u + 0u].xyz, delta),
                                 dot(nodeGradients[node * 3u + 1u].xyz, delta),
                                 dot(nodeGradients[node * 3u + 2u].xyz, delta));
    const float changeLength = length(change);
    const float baseLength = length(baseVelocity);
    const float3 boundedChange = changeLength > baseLength && changeLength > 0.0f
        ? change * (baseLength / changeLength) : change;
    return float4(sliceColor(length(baseVelocity + boundedChange)), 1.0f);
}
