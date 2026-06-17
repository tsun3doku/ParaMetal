struct Node
{
    float volume;
    uint neighborOffset;
    uint neighborCount;
    uint interfaceNeighborCount;
};

struct GMLSInterface
{
    uint neighborIdx;
    float conductance;
};

struct MaterialNode
{
    float conductivityPerMass;
    float thermalMass;
    float density;
    float specificHeat;
    float conductivity;
    float fixedTemperatureValue;
    uint boundaryCondition;
    uint _pad0;
};

struct SimPlayback
{
    uint resetCounter;
    uint recordedTimelineFrames;
    uint timelineFrameCount;
    float deltaTime;
};

struct HeatModelPushConstant
{
    uint elementCount;
};

[[vk::binding(0)]] StructuredBuffer<Node> simNodeBuffer;
[[vk::binding(1)]] StructuredBuffer<GMLSInterface> gmlsInterfaceBuffer;
[[vk::binding(2)]] StructuredBuffer<MaterialNode> materialNodeBuffer;
[[vk::binding(3)]] ConstantBuffer<SimPlayback> playback;

[[vk::binding(4)]] StructuredBuffer<float> tempRead;
[[vk::binding(5)]] RWStructuredBuffer<float> tempWrite;

[[vk::binding(7)]] RWStructuredBuffer<float2> couplingAccumulator;

[[vk::push_constant]] ConstantBuffer<HeatModelPushConstant> pushConstants;

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint localNodeID = dtid.x;
    if (localNodeID >= pushConstants.elementCount) {
        return;
    }

    MaterialNode materialNode = materialNodeBuffer[localNodeID];
    if (materialNode.boundaryCondition == 1u) {
        tempWrite[localNodeID] = materialNode.fixedTemperatureValue;
        return;
    }

    Node node = simNodeBuffer[localNodeID];
    float thermalMass = max(materialNode.thermalMass, 1e-12);
    float currentTemp = tempRead[localNodeID];

    float totalConductance = 0.0;
    float totalFlux = 0.0;

    for (uint i = 0; i < node.neighborCount; ++i) {
        GMLSInterface iface = gmlsInterfaceBuffer[node.neighborOffset + i];
        float g = iface.conductance;
        uint neighborLocalID = iface.neighborIdx;
        float neighborTemp = tempRead[neighborLocalID];
        totalConductance += g;
        totalFlux += g * neighborTemp;
    }

    float dt = playback.deltaTime;
    float kappa = materialNode.conductivityPerMass;

    float2 contactAccum = couplingAccumulator[localNodeID];
    float contactK = contactAccum.x;
    float contactKT = contactAccum.y;

    float invMass = 1.0 / thermalMass;
    float denominator = 1.0 + dt * kappa * totalConductance + dt * contactK * invMass;
    float numerator = currentTemp + dt * kappa * totalFlux + dt * contactKT * invMass;
    float newTemp = numerator / denominator;

    tempWrite[localNodeID] = newTemp;

    couplingAccumulator[localNodeID] = float2(0.0, 0.0);
}
