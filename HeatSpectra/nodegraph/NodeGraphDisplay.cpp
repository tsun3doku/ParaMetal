#include "NodeGraphDisplay.hpp"

#include "NodeGraphProductTypes.hpp"
#include "NodeGraphUtils.hpp"
#include "runtime/RuntimeECS.hpp"

std::unordered_set<uint64_t> NodeGraphDisplay::computeDisplaySelectedKeys(const NodeGraphState& graphState, const ECSRegistry& registry) const {
    // Build dependency graph
    std::unordered_map<uint64_t, std::vector<uint64_t>> depsByKey;

    for (auto entity : registry.view<ModelPackage>()) {
        depsByKey[static_cast<uint64_t>(entity)] = {};
    }
    for (auto entity : registry.view<RemeshPackage>()) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        const auto& package = registry.get<RemeshPackage>(entity);
        std::vector<uint64_t> deps;
        if (package.modelProductHandle.outputSocketKey != 0) {
            deps.push_back(package.modelProductHandle.outputSocketKey);
        }
        depsByKey[socketKey] = std::move(deps);
    }
    for (auto entity : registry.view<VoronoiPackage>()) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        const auto& package = registry.get<VoronoiPackage>(entity);
        std::vector<uint64_t> deps;
        for (const ProductHandle& handle : package.modelProducts) {
            if (handle.outputSocketKey != 0) {
                deps.push_back(handle.outputSocketKey);
            }
        }
        for (const ProductHandle& handle : package.modelRemeshProducts) {
            if (handle.outputSocketKey != 0) {
                deps.push_back(handle.outputSocketKey);
            }
        }
        depsByKey[socketKey] = std::move(deps);
    }
    for (auto entity : registry.view<ContactPackage>()) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        const auto& package = registry.get<ContactPackage>(entity);
        std::vector<uint64_t> deps;
        if (package.modelAModelProduct.outputSocketKey != 0) {
            deps.push_back(package.modelAModelProduct.outputSocketKey);
        }
        if (package.modelBModelProduct.outputSocketKey != 0) {
            deps.push_back(package.modelBModelProduct.outputSocketKey);
        }
        if (package.modelARemeshProduct.outputSocketKey != 0) {
            deps.push_back(package.modelARemeshProduct.outputSocketKey);
        }
        if (package.modelBRemeshProduct.outputSocketKey != 0) {
            deps.push_back(package.modelBRemeshProduct.outputSocketKey);
        }
        depsByKey[socketKey] = std::move(deps);
    }
    for (auto entity : registry.view<HeatPackage>()) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        const auto& package = registry.get<HeatPackage>(entity);
        std::vector<uint64_t> deps;
        for (const ProductHandle& handle : package.modelProducts) {
            if (handle.outputSocketKey != 0) {
                deps.push_back(handle.outputSocketKey);
            }
        }
        for (const ProductHandle& handle : package.remeshProducts) {
            if (handle.outputSocketKey != 0) {
                deps.push_back(handle.outputSocketKey);
            }
        }
        for (const ProductHandle& handle : package.voronoiProducts) {
            if (handle.isValid()) {
                deps.push_back(handle.outputSocketKey);
            }
        }
        for (const ProductHandle& handle : package.contactProducts) {
            if (handle.isValid()) {
                deps.push_back(handle.outputSocketKey);
            }
        }
        depsByKey[socketKey] = std::move(deps);
    }

    // Collect display enabled output socket keys
    std::unordered_set<uint64_t> selectedKeys;
    for (const NodeGraphNode& node : graphState.nodes) {
        if (!node.displayEnabled) {
            continue;
        }

        for (const NodeGraphSocket& output : node.outputs) {
            const auto payloadType = output.contract.producedPayloadType;
            if (payloadType != NodePayloadType::Geometry &&
                payloadType != NodePayloadType::Remesh &&
                payloadType != NodePayloadType::Voronoi &&
                payloadType != NodePayloadType::Contact &&
                payloadType != NodePayloadType::Heat) {
                continue;
            }

            selectedKeys.insert(makeSocketKey(node.id, output.id));
        }
    }

    // Walk dependency closure
    std::vector<uint64_t> stack;
    stack.reserve(selectedKeys.size());
    for (uint64_t key : selectedKeys) {
        stack.push_back(key);
    }

    while (!stack.empty()) {
        const uint64_t key = stack.back();
        stack.pop_back();

        auto it = depsByKey.find(key);
        if (it == depsByKey.end()) {
            continue;
        }

        for (uint64_t depKey : it->second) {
            if (selectedKeys.insert(depKey).second) {
                stack.push_back(depKey);
            }
        }
    }

    return selectedKeys;
}