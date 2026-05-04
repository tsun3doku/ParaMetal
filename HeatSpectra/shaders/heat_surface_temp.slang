struct SurfacePoint
{
    float3 position;
    float  temperature;
    float3 normal;
    float  area;
    float4 color;
};

struct GMLSSurfaceStencil
{
    uint valueWeightOffset;
    uint valueWeightCount;
    uint gradientWeightOffset;
    uint gradientWeightCount;
};

struct GMLSSurfaceWeight
{
    uint cellIndex;
    float weight;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0)]] StructuredBuffer<uint> nodeTemperatureReadBuffer;
[[vk::binding(1)]] RWStructuredBuffer<SurfacePoint> surfaceBuffer;

[[vk::binding(10)]] StructuredBuffer<GMLSSurfaceStencil> gmlsSurfaceStencilBuffer;
[[vk::binding(11)]] StructuredBuffer<GMLSSurfaceWeight> gmlsSurfaceWeightBuffer;

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint vertexID = dtid.x;

    uint surfaceCount = 0;
    uint surfaceStride = 0;
    surfaceBuffer.GetDimensions(surfaceCount, surfaceStride);

    if (vertexID >= surfaceCount) return;

    uint nodeCount = 0;
    uint nodeStride = 0;
    nodeTemperatureReadBuffer.GetDimensions(nodeCount, nodeStride);

    float displayTemp = 0.0;

    uint stencilCount = 0;
    uint stencilStride = 0;
    gmlsSurfaceStencilBuffer.GetDimensions(stencilCount, stencilStride);
    if (vertexID >= stencilCount)
    {
        return;
    }

    GMLSSurfaceStencil stencil = gmlsSurfaceStencilBuffer[vertexID];
    if (stencil.valueWeightCount == 0u)
    {
        return;
    }

    for (uint i = 0; i < stencil.valueWeightCount; ++i)
    {
        GMLSSurfaceWeight weight = gmlsSurfaceWeightBuffer[stencil.valueWeightOffset + i];
        if (weight.cellIndex >= nodeCount)
        {
            continue;
        }

        displayTemp += weight.weight * asfloat(nodeTemperatureReadBuffer[weight.cellIndex]);
    }

    SurfacePoint p = surfaceBuffer[vertexID];
    p.temperature = displayTemp;
    surfaceBuffer[vertexID] = p;
}
