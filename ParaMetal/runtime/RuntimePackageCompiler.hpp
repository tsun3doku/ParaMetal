#pragma once

#include <cstdint>

#include "domain/HeatData.hpp"
#include "domain/PointData.hpp"
#include "nodegraph/NodeGraphProductTypes.hpp"
#include "nodegraph/NodeGraphRuntime.hpp"
#include "runtime/RuntimeECS.hpp"
#include "runtime/RuntimePackages.hpp"

class NodePayloadRegistry;

class RuntimePackageCompiler {
public:
    ModelPackage buildModelPackage(
        const GeometryData& geometry) const;
    RemeshPackage buildRemeshPackage(
        const NodeGraphNode& node,
        const RemeshData& remesh,
        const NodePayloadRegistry* payloadRegistry,
        const ECSRegistry& registry,
        const NodeDataHandle& remeshHandle = {}) const;
    VoronoiPackage buildVoronoiPackage(
        const NodeGraphNode& node,
        const NodePayloadRegistry* payloadRegistry,
        const ECSRegistry& registry,
        const VoronoiData& voronoi,
        const NodeDataHandle& voronoiHandle = {}) const;
    HeatPackage buildHeatPackage(
        const NodeGraphNode& node,
        const NodePayloadRegistry* payloadRegistry,
        const ECSRegistry& registry,
        const HeatData& heat,
        const NodeDataHandle& heatHandle) const;
    PointPackage buildPointPackage(
        const NodePayloadRegistry* payloadRegistry,
        const PointData& pointData,
        const NodeDataHandle& pointHandle,
        const ECSRegistry& registry) const;
    ContactPackage buildContactPackage(
        const NodeGraphNode& node,
        const NodePayloadRegistry* payloadRegistry,
        const ECSRegistry& registry,
        const ContactData& contact,
        const NodeDataHandle& contactHandle = {}) const;

    void compileAndApply(
        const NodeGraphState& graphState,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;

private:
    void compileModel(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;
    void compileGroup(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;
    void compileTransform(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;
    void compileRemesh(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;
    void compileMeshPoints(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;
    void compilePoints(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;
    void compileMerge(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;
    void compileVoronoi(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;
    void compileContact(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;
    void compileHeatSolve(
        const NodeGraphNode& node,
        const NodeGraphEvaluationState& execState,
        const NodePayloadRegistry* payloadRegistry,
        ECSRegistry& registry,
        std::unordered_set<ECSEntity>& staleEntities) const;

    template <typename PackageT>
    static void applyPackage(ECSRegistry& registry, uint64_t socketKey, const PackageT& pkg, std::unordered_set<ECSEntity>& staleEntities);
};
