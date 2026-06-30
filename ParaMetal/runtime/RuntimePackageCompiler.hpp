#pragma once

#include <cstdint>

#include "domain/HeatData.hpp"
#include "domain/PointData.hpp"
#include "nodegraph/NodeGraphProductTypes.hpp"
#include "nodegraph/NodeGraphRuntime.hpp"
#include "nodegraph/NodeGraphState.hpp"
#include "runtime/RuntimePackageManager.hpp"
#include "runtime/RuntimeProductManager.hpp"

class NodePayloadRegistry;

class RuntimePackageCompiler {
public:
    ModelPackage buildModelPackage(
        const GeometryData& geometry,
        const HashValues& geometryHashes) const;
    RemeshPackage buildRemeshPackage(
        const NodeGraphNode& node,
        const RemeshData& remesh,
        const HashValues& sourceGeometryHashes,
        const NodePayloadRegistry* payloadRegistry,
        const ProductHandle& sourceModelProduct,
        const NodeDataHandle& remeshHandle = {}) const;
    VoronoiPackage buildVoronoiPackage(
        const NodeGraphNode& node,
        const NodePayloadRegistry* payloadRegistry,
        const VoronoiData& voronoi,
        const ProductHandle& modelProduct,
        const ProductHandle& remeshProduct,
        const HashValues& authoredHashes,
        const NodeDataHandle& voronoiHandle = {}) const;
    HeatPackage buildHeatPackage(
        const NodeGraphNode& node,
        const NodePayloadRegistry* payloadRegistry,
        const RuntimePackageManager& packages,
        const NodeGraphEvaluationState& execState,
        const HeatData& heat,
        const HashValues& authoredHashes,
        const NodeDataHandle& heatHandle) const;
    PointPackage buildPointPackage(
        const NodePayloadRegistry* payloadRegistry,
        const PointData& pointData,
        const NodeDataHandle& pointHandle) const;
    ContactPackage buildContactPackage(
        const NodeGraphNode& node,
        const NodePayloadRegistry* payloadRegistry,
        const ContactData& contact,
        const ProductHandle& modelARemeshProduct,
        const ProductHandle& modelBRemeshProduct,
        const HashValues& authoredHashes,
        const NodeDataHandle& contactHandle = {}) const;

    void compileNode(
        const NodeGraphState& graphState,
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;

private:
    HashValues resolveHandleHashes(const NodePayloadRegistry* payloadRegistry, const NodeDataHandle& handle) const;

    void compileModel(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;
    void compileGroup(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;
    void compileTransform(
        const NodeGraphState& graphState,
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;
    void compileRemesh(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;
    void compileMeshPoints(
        const NodeGraphState& graphState,
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;
    void compilePoints(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;
    void compileMerge(
        const NodeGraphState& graphState,
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;
    void compileVoronoi(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;
    void compileContact(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;
    void compileHeatSolve(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        RuntimeProductManager& products,
        RuntimePackageManager& packages) const;

};
