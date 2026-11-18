struct SliceUniformBufferObject { float4x4 model; float4x4 view; float4x4 proj; float4 color; };
[[vk::binding(0, 0)]] ConstantBuffer<SliceUniformBufferObject> ubo;
struct WireVertex { [[vk::location(0)]] float3 position : POSITION; };
struct WireOut { float4 position : SV_Position; };
[shader("vertex")]
WireOut vsMain(WireVertex input) {
    WireOut output;
    output.position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.position, 1.0f))));
    return output;
}
[shader("fragment")]
float4 fsMain(WireOut input) : SV_Target { return float4(0.76f, 0.79f, 0.86f, 1.0f); }
