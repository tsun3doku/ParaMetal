#include "NodeCFDParams.hpp"
#include "NodeGraphEditor.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeParamUtils.hpp"

CFDNodeParams readCFDNodeParams(const NodeGraphNode& node) {
    CFDNodeParams params{};
    params.enabled = NodeParamUtils::readBoolParam(node, nodegraphparams::cfd::Enabled, true);
    params.targetCourant = NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::TargetCourant, 0.5);
    params.timeStep = NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::TimeStep, 1.0 / 60.0);
    params.initialVelocity.x = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::VelocityX, 0.0));
    params.initialVelocity.y = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::VelocityY, 0.0));
    params.initialVelocity.z = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::VelocityZ, 0.0));
    params.pressureIterations = NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::PressureIterations, 200.0);
    params.pressureTolerance = NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::PressureTolerance, 1e-4);

    // Backward migration check: if -X boundary type is not present, migrate once
    const int face0Type = NodeParamUtils::readEnumParam(node, nodegraphparams::cfd::domainBoundaryTypeParam(0), -1);

    if (face0Type == -1) {
        // One-time backward migration:
        // -X VelocityInlet with old initialVelocity
        params.boundaries[0].type = CFDData::BoundaryType::VelocityInlet;
        params.boundaries[0].velocity = params.initialVelocity;

        // +X PressureOutlet
        params.boundaries[1].type = CFDData::BoundaryType::PressureOutlet;
        params.boundaries[1].velocity = glm::vec3(0.0f);

        // Faces 2..5 (±Y, ±Z) NoSlipWall
        for (uint32_t face = 2; face < 6; ++face) {
            params.boundaries[face].type = CFDData::BoundaryType::NoSlipWall;
            params.boundaries[face].velocity = glm::vec3(0.0f);
        }
    } else {
        for (uint32_t face = 0; face < 6; ++face) {
            const int typeInt = NodeParamUtils::readEnumParam(node, nodegraphparams::cfd::domainBoundaryTypeParam(face), static_cast<int>(CFDData::BoundaryType::NoSlipWall));
            params.boundaries[face].type = static_cast<CFDData::BoundaryType>(typeInt);
            params.boundaries[face].velocity.x = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::domainBoundaryVelocityXParam(face), 0.0));
            params.boundaries[face].velocity.y = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::domainBoundaryVelocityYParam(face), 0.0));
            params.boundaries[face].velocity.z = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::domainBoundaryVelocityZParam(face), 0.0));
        }
    }

    return params;
}

bool writeCFDNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const CFDNodeParams& params) {
    const bool baseOk =
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::Enabled, NodeGraphParamType::Bool, 0.0, 0, params.enabled}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::TargetCourant, NodeGraphParamType::Float, params.targetCourant}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::TimeStep, NodeGraphParamType::Float, params.timeStep}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::VelocityX, NodeGraphParamType::Float, static_cast<double>(params.initialVelocity.x)}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::VelocityY, NodeGraphParamType::Float, static_cast<double>(params.initialVelocity.y)}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::VelocityZ, NodeGraphParamType::Float, static_cast<double>(params.initialVelocity.z)}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::PressureIterations, NodeGraphParamType::Int, params.pressureIterations}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::PressureTolerance, NodeGraphParamType::Float, params.pressureTolerance});

    if (!baseOk) {
        return false;
    }

    for (uint32_t face = 0; face < 6; ++face) {
        const bool faceOk =
            editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::domainBoundaryTypeParam(face), NodeGraphParamType::Enum, 0.0, static_cast<int64_t>(params.boundaries[face].type), false}) &&
            editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::domainBoundaryVelocityXParam(face), NodeGraphParamType::Float, static_cast<double>(params.boundaries[face].velocity.x)}) &&
            editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::domainBoundaryVelocityYParam(face), NodeGraphParamType::Float, static_cast<double>(params.boundaries[face].velocity.y)}) &&
            editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::domainBoundaryVelocityZParam(face), NodeGraphParamType::Float, static_cast<double>(params.boundaries[face].velocity.z)});
        if (!faceOk) {
            return false;
        }
    }

    return true;
}
