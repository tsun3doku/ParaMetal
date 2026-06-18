#include "NodeContact.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphDataTypes.hpp"
#include "NodeGraphUtils.hpp"

#include "NodeGraphBridge.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeContactParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"

#include <cstdint>

const char* NodeContact::typeId() const {
    return nodegraphtypes::Contact;
}

void NodeContact::execute(NodeGraphKernelContext& context) const {
    const NodeGraphSocket* emitterSocket = context.node.input("SurfaceA");
    const NodeGraphSocket* receiverSocket = context.node.input("SurfaceB");
    const EvaluatedSocketValue* emitterInputValue =
        emitterSocket ? readEvaluatedInput(context.node, emitterSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* emitterInput = readInputValue(emitterInputValue);
    const EvaluatedSocketValue* receiverInputValue =
        receiverSocket ? readEvaluatedInput(context.node, receiverSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* receiverInput = readInputValue(receiverInputValue);

    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;

    NodeDataHandle emitterMeshHandle{};
    bool hasEmitterEndpoint = false;
    if (payloadRegistry && emitterInput && emitterInput->payloadHandle.key != 0) {
        emitterMeshHandle = payloadRegistry->resolveMeshHandle(emitterInput->dataType, emitterInput->payloadHandle);
        hasEmitterEndpoint = emitterMeshHandle.key != 0;
    }

    NodeDataHandle receiverMeshHandle{};
    bool hasReceiverEndpoint = false;
    if (payloadRegistry && receiverInput && receiverInput->payloadHandle.key != 0) {
        receiverMeshHandle = payloadRegistry->resolveMeshHandle(receiverInput->dataType, receiverInput->payloadHandle);
        hasReceiverEndpoint = receiverMeshHandle.key != 0;
    }

    const bool hasValidContact = hasEmitterEndpoint && hasReceiverEndpoint &&
        !(emitterMeshHandle == receiverMeshHandle);

    const ContactNodeParams params = readContactNodeParams(context.node);

    for (std::size_t outputIndex = 0;
         outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Contact || !hasValidContact) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        ContactData contactData{};
        contactData.pair.endpointA.payloadHandle = emitterInput->payloadHandle;
        contactData.pair.endpointA.meshHandle = emitterMeshHandle;
        contactData.pair.endpointB.payloadHandle = receiverInput->payloadHandle;
        contactData.pair.endpointB.meshHandle = receiverMeshHandle;
        contactData.pair.minNormalDot = static_cast<float>(params.minNormalDot);
        contactData.pair.contactRadius = static_cast<float>(params.contactRadius);
        contactData.pair.hasValidContact = true;
        contactData.active = true;

        const uint64_t payloadKey = NodeSocketKey(context.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, contactData, context.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeContact::computeOutputHashes(const NodeGraphKernelHashContext& context) const {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combineString(hash, nodegraphtypes::Contact);
    HashNodeCache::combineSocket(hash, context, "SurfaceA", HashDomain::Geometry);
    HashNodeCache::combineSocket(hash, context, "SurfaceB", HashDomain::Geometry);

    const ContactNodeParams params = readContactNodeParams(context.node);
    HashBuilder::combineFloat(hash, static_cast<float>(params.minNormalDot));
    HashBuilder::combineFloat(hash, static_cast<float>(params.contactRadius));

    HashValues values{};
    values.full = hash;
    values.geometry = hash;
    values.simulation = hash;
    return values;
}
