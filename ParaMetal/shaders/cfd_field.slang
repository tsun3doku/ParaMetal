static const uint CfdInvalidSeed = 0xFFFFFFFFu;
static const uint CfdSolidSeed   = 0xFFFFFFFEu;

struct CfdFieldParams {
    float4 gridMinCellSize;
    int4   gridDim;
    uint4  counts;
    float4 domainMin;
    float4 domainMax;
};

bool sampleCfdField(
    StructuredBuffer<uint> fluidSeedGrid,
    StructuredBuffer<uint> seedToFluidNode,
    StructuredBuffer<float4> velocity,
    StructuredBuffer<float4> nodeCentroids,
    StructuredBuffer<float4> nodeGradients,
    CfdFieldParams params,
    float3 p,
    out float3 outVelocity)
{
    outVelocity = float3(0.0f);

    const float cellSize = params.gridMinCellSize.w;
    const int3 dim = params.gridDim.xyz;
    if (cellSize <= 0.0f || any(dim <= int3(0))) {
        return false;
    }

    if (any(p < params.domainMin.xyz) || any(p > params.domainMax.xyz)) {
        return false;
    }

    const int3 voxel = int3(floor((p - params.gridMinCellSize.xyz) / cellSize));
    if (any(voxel < int3(0)) || any(voxel >= dim)) {
        return false;
    }

    const uint index = uint(voxel.x) + uint(voxel.y) * uint(dim.x) + uint(voxel.z) * uint(dim.x) * uint(dim.y);
    const uint seed = fluidSeedGrid[index];
    if (seed >= CfdSolidSeed || seed >= params.counts.x) {
        return false;
    }

    const uint node = seedToFluidNode[seed];
    if (node >= params.counts.y) {
        return false;
    }

    const float3 v = velocity[node].xyz;
    const float3 centroid = nodeCentroids[node].xyz;
    const float3 gradUx = nodeGradients[node * 3u + 0u].xyz;
    const float3 gradUy = nodeGradients[node * 3u + 1u].xyz;
    const float3 gradUz = nodeGradients[node * 3u + 2u].xyz;

    const float3 dx = p - centroid;
    const float3 du = float3(
        dot(gradUx, dx),
        dot(gradUy, dx),
        dot(gradUz, dx)
    );

    const float maxDu = length(v);
    const float duLen = length(du);
    const float3 safeDu = (duLen > maxDu) ? (du * (maxDu / duLen)) : du;

    outVelocity = v + safeDu;
    return true;
}
