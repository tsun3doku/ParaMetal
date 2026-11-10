static const uint InvalidSeed = 0xFFFFFFFFu;
static const uint SolidSeed   = 0xFFFFFFFEu;

struct LineVertex {
    float4 positionSpeed;  // xyz = world position, w = sampled speed
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>     fluidSeedGrid;
[[vk::binding(1, 0)]] StructuredBuffer<uint>     seedToFluidNode;
[[vk::binding(2, 0)]] StructuredBuffer<float4>   velocity;
[[vk::binding(3, 0)]] StructuredBuffer<float4>   seeds;
[[vk::binding(4, 0)]] RWStructuredBuffer<LineVertex> lineVertices;

struct PushConstants {
    float4 gridMinCellSize;      // xyz = grid min, w = cell size
    int4   gridDimMaxSegments;   // xyz = grid dim, w = segments per line
    float4 stepLengthPadding;    // x = spatial step
    uint4  counts;               // x = seed count, y = seed map count, z = fluid node count
    float4 domainMin;            // xyz = domain box min
    float4 domainMax;            // xyz = domain box max
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

// Fraction of p->next at which the segment leaves the domain box; p is inside.
float domainExitFraction(float3 p, float3 next) {
    const float3 delta = next - p;
    float fraction = 1.0f;
    for (int axis = 0; axis < 3; ++axis) {
        if (delta[axis] > 0.0f) {
            fraction = min(fraction, (pc.domainMax[axis] - p[axis]) / delta[axis]);
        } else if (delta[axis] < 0.0f) {
            fraction = min(fraction, (pc.domainMin[axis] - p[axis]) / delta[axis]);
        }
    }
    return clamp(fraction, 0.0f, 1.0f);
}

bool sampleVelocity(float3 p, out float3 direction, out float speed) {
    direction = float3(0.0f);
    speed = 0.0f;

    const float cellSize = pc.gridMinCellSize.w;
    const int3 dim = pc.gridDimMaxSegments.xyz;
    if (cellSize <= 0.0f || any(dim <= int3(0))) {
        return false;
    }
    // The seed grid is inflated beyond the domain, so padded voxels would
    // otherwise project live cells onto space outside the domain faces.
    if (any(p < pc.domainMin.xyz) || any(p > pc.domainMax.xyz)) {
        return false;
    }

    const int3 voxel = int3(floor((p - pc.gridMinCellSize.xyz) / cellSize));
    if (any(voxel < int3(0)) || any(voxel >= dim)) {
        return false;
    }

    const uint index = uint(voxel.x) +
        uint(voxel.y) * uint(dim.x) +
        uint(voxel.z) * uint(dim.x) * uint(dim.y);
    const uint seed = fluidSeedGrid[index];
    if (seed >= SolidSeed || seed >= pc.counts.y) {
        return false;
    }

    const uint node = seedToFluidNode[seed];
    if (node >= pc.counts.z) {
        return false;
    }

    const float3 v = velocity[node].xyz;
    speed = length(v);
    if (!(speed > 1e-6f)) {
        return false;
    }

    direction = v / speed;
    return true;
}

void writeCollapsed(uint base, uint firstSegment, uint segmentCount, float3 position) {
    LineVertex collapsed;
    collapsed.positionSpeed = float4(position, 0.0f);
    for (uint s = firstSegment; s < segmentCount; ++s) {
        lineVertices[base + 2u * s] = collapsed;
        lineVertices[base + 2u * s + 1u] = collapsed;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint seedIndex = id.x;
    const uint segmentCount = uint(pc.gridDimMaxSegments.w);
    if (seedIndex >= pc.counts.x || segmentCount == 0u) {
        return;
    }

    const uint base = seedIndex * segmentCount * 2u;
    const float stepLength = pc.stepLengthPadding.x;

    float3 p = seeds[seedIndex].xyz;
    float3 direction;
    float speed;
    if (stepLength <= 0.0f || !sampleVelocity(p, direction, speed)) {
        writeCollapsed(base, 0u, segmentCount, p);
        return;
    }

    uint written = 0u;
    for (uint s = 0; s < segmentCount; ++s) {
        float3 advance;
        float advanceSpeed;
        float3 midDirection;
        float midSpeed;
        if (sampleVelocity(p + direction * (0.5f * stepLength), midDirection, midSpeed)) {
            advance = stepLength * midDirection;
            advanceSpeed = midSpeed;
        } else {
            advance = stepLength * direction;
            advanceSpeed = speed;
        }

        // Stop on the domain face instead of overshooting it by one step.
        const float exitFraction = domainExitFraction(p, p + advance);

        LineVertex start;
        start.positionSpeed = float4(p, speed);

        LineVertex end;
        end.positionSpeed = float4(p + exitFraction * advance, advanceSpeed);

        lineVertices[base + 2u * s] = start;
        lineVertices[base + 2u * s + 1u] = end;
        written = s + 1u;

        if (exitFraction < 1.0f) {
            p += exitFraction * advance;
            break;
        }

        const float3 nextPosition = p + advance;
        float3 nextDirection;
        float nextSpeed;
        if (!sampleVelocity(nextPosition, nextDirection, nextSpeed)) {
            break;
        }
        p = nextPosition;
        direction = nextDirection;
        speed = nextSpeed;
    }

    writeCollapsed(base, written, segmentCount, p);
}
