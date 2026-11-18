#include "cfd_field.slang"

struct LineVertex {
    float4 positionSpeed;  // xyz = world position, w = sampled speed
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>     fluidSeedGrid;
[[vk::binding(1, 0)]] StructuredBuffer<uint>     seedToFluidNode;
[[vk::binding(2, 0)]] StructuredBuffer<float4>   velocity;
[[vk::binding(3, 0)]] StructuredBuffer<float4>   seeds;
[[vk::binding(4, 0)]] RWStructuredBuffer<LineVertex> lineVertices;
[[vk::binding(5, 0)]] StructuredBuffer<float4>   nodeCentroids;
[[vk::binding(6, 0)]] StructuredBuffer<float4>   nodeGradients;

struct PushConstants {
    float4 gridMinCellSize;      // xyz = grid min, w = cell size
    int4   gridDimMaxSegments;   // xyz = grid dim, w = segments per line
    float4 stepLengthPadding;    // x = spatial step
    uint4  counts;               // x = seed count, y = seed map count, z = fluid node count
    float4 domainMin;            // xyz = domain box min
    float4 domainMax;            // xyz = domain box max
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

CfdFieldParams fieldParams() {
    return {pc.gridMinCellSize, pc.gridDimMaxSegments,
            uint4(pc.counts.y, pc.counts.z, 0u, 0u), pc.domainMin, pc.domainMax};
}

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

    float3 linearV;
    if (!sampleCfdField(fluidSeedGrid, seedToFluidNode, velocity, nodeCentroids,
                        nodeGradients, fieldParams(), p, linearV)) {
        return false;
    }

    speed = length(linearV);
    if (!(speed > 1e-6f)) {
        return false;
    }

    direction = linearV / speed;
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
        float3 midDirection;
        float midSpeed;
        float3 advance;
        float advanceSpeed;

        if (sampleVelocity(p + direction * (0.5f * stepLength), midDirection, midSpeed)) {
            advance = stepLength * midDirection;
            advanceSpeed = midSpeed;
        } else {
            advance = stepLength * direction;
            advanceSpeed = speed;
        }

        const float exitFraction = domainExitFraction(p, p + advance);
        float3 nextPosition = p + exitFraction * advance;

        // Verify destination is inside valid fluid BEFORE writing the segment.
        // If it hit a solid or boundary, test a half-step; if that also fails,
        // stop cleanly at p so no segment is ever drawn into solid geometry.
        float3 nextDirection;
        float nextSpeed;
        if (!sampleVelocity(nextPosition, nextDirection, nextSpeed)) {
            nextPosition = p + (0.5f * exitFraction) * advance;
            if (!sampleVelocity(nextPosition, nextDirection, nextSpeed)) {
                break;
            }
        }

        LineVertex start;
        start.positionSpeed = float4(p, speed);

        LineVertex end;
        end.positionSpeed = float4(nextPosition, advanceSpeed);

        lineVertices[base + 2u * s] = start;
        lineVertices[base + 2u * s + 1u] = end;
        written = s + 1u;

        if (exitFraction < 1.0f) {
            p = nextPosition;
            break;
        }

        p = nextPosition;
        direction = nextDirection;
        speed = nextSpeed;
    }

    writeCollapsed(base, written, segmentCount, p);
}
