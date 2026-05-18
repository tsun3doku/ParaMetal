struct ContactIndex {
    uint offset;
    uint count;
    float contactK;
    uint _pad;
};

struct ContactSampleWeight {
    uint cellIndex;
    float weight;
};

struct ContactPushConstant {
    uint elementCount;
};

[[vk::binding(0)]] StructuredBuffer<uint> sourceFieldBuffer;
[[vk::binding(1)]] RWStructuredBuffer<float2> destAccumulator;
[[vk::binding(2)]] StructuredBuffer<ContactSampleWeight> contactEdges;
[[vk::binding(3)]] StructuredBuffer<ContactIndex> contactIndices;

[[vk::push_constant]] ConstantBuffer<ContactPushConstant> pushConstants;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    uint nodeID = threadId.x;
    if (nodeID >= pushConstants.elementCount) return;

    ContactIndex meta = contactIndices[nodeID];
    float contactK = meta.contactK;
    float gathered = 0.0;

    for (uint i = 0; i < meta.count; ++i) {
        ContactSampleWeight edge = contactEdges[meta.offset + i];
        gathered += edge.weight * asfloat(sourceFieldBuffer[edge.cellIndex]);
    }

    destAccumulator[nodeID] += float2(contactK, gathered);
}
