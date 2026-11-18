#include "palette_lut.slang"

[[vk::binding(1, 0)]] Sampler2D infernoLut;
[[vk::binding(2, 0)]] Sampler2D viridisLut;
[[vk::binding(3, 0)]] Sampler2D inferno2Lut;
[[vk::binding(4, 0)]] Sampler2D parulaLut;

struct PushConstants {
    float minSpeed;
    float maxSpeed;
    uint  palette;
    float cullDistance;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[shader("fragment")]
float4 main(
    [[vk::location(0)]] float inSpeed,
    [[vk::location(1)]] float inForwardDepth) : SV_Target
{
    if (pc.cullDistance > 0.0f && inForwardDepth < pc.cullDistance) {
        discard;
    }

    float3 color = velocityColor(
        infernoLut, viridisLut, inferno2Lut, parulaLut,
        inSpeed, pc.palette, pc.minSpeed, pc.maxSpeed);
    return float4(color, 1.0f);
}
