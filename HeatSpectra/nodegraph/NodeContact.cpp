#include "NodeContact.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphDataTypes.hpp"
#include "NodeGraphUtils.hpp"

#include "NodeGraphBridge.hpp"
#include "NodeGraphHash.hpp"
#include "NodeContactParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"

#include <cstdint>

const char* NodeContact::typeId() const {
    return nodegraphtypes::Contact;
}

void NodeContact::execute(NodeGraphKernelContext& context) const {
    const NodeGraphSocket* emitterSocket = context.node.input("Emitter");
    const NodeGraphSocket* receiverSocket = context.node.input("Receiver");
    const EvaluatedSocketValue* emitterInputValue =
        emitterSocket ? readEvaluatedInput(context.node, emitterSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* emitterInput = readInputValue(emitterInputValue);
    const EvaluatedSocketValue* receiverInputValue =
        receiverSocket ? readEvaluatedInput(context.node, receiverSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* receiverInput = readInputValue(receiverInputValue);

    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;
    NodeDataHandle emitterMeshHandle{};
    uint64_t emitterPayloadHash = 0;
    bool hasEmitterEndpoint = false;
    if (payloadRegistry && emitterInput &&
        emitterInput->payloadHandle.key != 0 &&
        (emitterInput->dataType == payloadtypes::Geometry ||
         emitterInput->dataType == payloadtypes::Remesh ||
         emitterInput->dataType == payloadtypes::HeatModel)) {
        emitterPayloadHash = payloadRegistry->resolvePayloadHash(emitterInput->payloadHandle);
        emitterMeshHandle = payloadRegistry->resolveMeshHandle(emitterInput->dataType, emitterInput->payloadHandle);
        hasEmitterEndpoint = emitterMeshHandle.key != 0;
    }

    NodeDataHandle receiverMeshHandle{};
    uint64_t receiverPayloadHash = 0;
    bool hasReceiverEndpoint = false;
    if (payloadRegistry && receiverInput &&
        receiverInput->payloadHandle.key != 0 &&
        (receiverInput->dataType == payloadtypes::Geometry ||
         receiverInput->dataType == payloadtypes::Remesh ||
         receiverInput->dataType == payloadtypes::HeatModel)) {
        receiverPayloadHash = payloadRegistry->resolvePayloadHash(receiverInput->payloadHandle);
        receiverMeshHandle = payloadRegistry->resolveMeshHandle(receiverInput->dataType, receiverInput->payloadHandle);
        hasReceiverEndpoint = receiverMeshHandle.key != 0;
    }

    const bool hasValidContact = hasEmitterEndpoint && hasReceiverEndpoint &&
        !(emitterMeshHandle == receiverMeshHandle);

    const ContactNodeParams params = readContactNodeParams(context.node);

    for (std::size_t outputIndex = 0; outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Contact ||
            !hasValidContact) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        ContactData contactData{};
        contactData.pair.endpointA.payloadHandle = emitterInput->payloadHandle;
        contactData.pair.endpointA.meshHandle = emitterMeshHandle;
        contactData.pair.endpointB.payloadHandle = receiverInput->payloadHandle;
        contactData.pair.endpointB.meshHandle = receiverMeshHandle;
        contactData.emitterPayloadHash = emitterPayloadHash;
        contactData.receiverPayloadHash = receiverPayloadHash;
        contactData.pair.minNormalDot = static_cast<float>(params.minNormalDot);
        contactData.pair.contactRadius = static_cast<float>(params.contactRadius);
        contactData.pair.hasValidContact = true;
        contactData.active = true;

        const uint64_t payloadKey = NodeSocketKey(context.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(contactData));
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

bool NodeContact::computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const {
    const NodeGraphSocket* emitterSocket = context.node.input("Emitter");
    const NodeGraphSocket* receiverSocket = context.node.input("Receiver");

    const EvaluatedSocketValue* emitterInputValue = emitterSocket ? readEvaluatedInput(context.node, emitterSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* emitterInput = readInputValue(emitterInputValue);
    if (emitterInput &&
        emitterInput->dataType != payloadtypes::Geometry &&
        emitterInput->dataType != payloadtypes::Remesh &&
        emitterInput->dataType != payloadtypes::HeatModel) {
        emitterInput = nullptr;
    }
    const EvaluatedSocketValue* receiverInputValue = receiverSocket ? readEvaluatedInput(context.node, receiverSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* receiverInput = readInputValue(receiverInputValue);
    if (receiverInput &&
        receiverInput->dataType != payloadtypes::Geometry &&
        receiverInput->dataType != payloadtypes::Remesh &&
        receiverInput->dataType != payloadtypes::HeatModel) {
        receiverInput = nullptr;
    }

    outHash = NodeGraphHash::start();
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(context.node.id.value));
    NodeGraphHash::combineInputHash(outHash, emitterInput);
    NodeGraphHash::combineInputHash(outHash, receiverInput);

    const ContactNodeParams params = readContactNodeParams(context.node);
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.minNormalDot));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.contactRadius));

    return true;
}
