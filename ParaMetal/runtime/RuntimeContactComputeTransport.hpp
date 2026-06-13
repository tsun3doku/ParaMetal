#pragma once

#include <unordered_set>
#include <vector>

#include "contact/ContactSystem.hpp"
#include "contact/ContactSystemComputeController.hpp"
#include "runtime/RuntimeECS.hpp"
#include "runtime/RuntimePackages.hpp"

class RuntimeContactComputeTransport {
public:
    void setController(ContactSystemComputeController* updatedController) {
        controller = updatedController;
    }

    void setECSRegistry(ECSRegistry* updatedRegistry) {
        ecsRegistry = updatedRegistry;
    }

    void sync(const ECSRegistry& registry) {
        if (!controller) {
            return;
        }

        std::unordered_set<uint64_t> nextSocketKeys;

        auto view = registry.view<ContactPackage>(entt::exclude<Stale>);
        for (auto entity : view) {
            uint64_t socketKey = static_cast<uint64_t>(entity);
            const auto& package = registry.get<ContactPackage>(entity);

            const uint64_t inputHash = buildConfigInputHash(socketKey, package);
            auto hashIt = appliedConfigInputHash.find(socketKey);
            if (inputHash != 0 && hashIt != appliedConfigInputHash.end() && hashIt->second == inputHash) {
                nextSocketKeys.insert(socketKey);
                continue;
            }

            if (!package.authored.active || !package.authored.pair.hasValidContact) {
                controller->disable(socketKey);
                removePublishedProduct(socketKey);
                appliedConfigInputHash.erase(socketKey);
                continue;
            }

            ContactSystemComputeController::Config config{};
            if (!tryBuildConfig(socketKey, package, config)) {
                continue;
            }
            controller->configure(socketKey, config);
            nextSocketKeys.insert(socketKey);
        }

        for (uint64_t socketKey : activeSocketKeys) {
            if (nextSocketKeys.find(socketKey) == nextSocketKeys.end()) {
                controller->disable(socketKey);
                removePublishedProduct(socketKey);
                appliedConfigInputHash.erase(socketKey);
            }
        }

        activeSocketKeys = std::move(nextSocketKeys);
    }

    void finalizeSync() {
        if (!controller) {
            return;
        }

        for (uint64_t socketKey : activeSocketKeys) {
            auto entity = static_cast<ECSEntity>(socketKey);
            const auto& package = ecsRegistry->get<ContactPackage>(entity);
            const uint64_t inputHash = buildConfigInputHash(socketKey, package);
            auto hashIt = appliedConfigInputHash.find(socketKey);
            const ContactProduct* product = tryGetProduct<ContactProduct>(*ecsRegistry, socketKey);
            if (!product || hashIt == appliedConfigInputHash.end() || hashIt->second != inputHash) {
                publishProduct(socketKey);
            }
        }
    }

private:
    bool tryBuildConfig(uint64_t socketKey, const ContactPackage& package, ContactSystemComputeController::Config& outConfig) const {
        if (socketKey == 0 || !ecsRegistry) {
            return false;
        }
        if (!package.authored.active || !package.authored.pair.hasValidContact) {
            return false;
        }

        const uint64_t modelAEntityId = package.modelAMeshHandle.key;
        const uint64_t modelBEntityId = package.modelBMeshHandle.key;
        if (modelAEntityId == 0 || modelBEntityId == 0) {
            return false;
        }

        const RemeshProduct* modelARemeshProduct =
            tryGetProduct<RemeshProduct>(*ecsRegistry, modelAEntityId);
        const RemeshProduct* modelBRemeshProduct =
            tryGetProduct<RemeshProduct>(*ecsRegistry, modelBEntityId);
        if (!modelARemeshProduct || !modelBRemeshProduct) {
            return false;
        }

        outConfig = {};
        outConfig.minNormalDot = package.authored.pair.minNormalDot;
        outConfig.contactRadius = package.authored.pair.contactRadius;
        outConfig.modelALocalToWorld = package.modelALocalToWorld;
        outConfig.modelAIntrinsicMesh = modelARemeshProduct->intrinsicMesh;
        outConfig.modelBLocalToWorld = package.modelBLocalToWorld;
        outConfig.modelBIntrinsicMesh = modelBRemeshProduct->intrinsicMesh;
        outConfig.modelARuntimeModelId = modelARemeshProduct->runtimeModelId;
        outConfig.modelBRuntimeModelId = modelBRemeshProduct->runtimeModelId;
        outConfig.modelBTriangleIndices = modelBRemeshProduct->intrinsicMesh.indices;
        if (!outConfig.isValid()) {
            return false;
        }

        outConfig.computeHash = buildConfigInputHash(socketKey, package);
        return true;
    }

    uint64_t buildConfigInputHash(uint64_t socketKey, const ContactPackage& package) const {
        (void)socketKey;
        uint64_t hash = package.packageHash;
        const RemeshProduct* modelARemeshProduct = tryGetProduct<RemeshProduct>(*ecsRegistry, package.modelAMeshHandle.key);
        const RemeshProduct* modelBRemeshProduct = tryGetProduct<RemeshProduct>(*ecsRegistry, package.modelBMeshHandle.key);
        if (!modelARemeshProduct || !modelBRemeshProduct) {
            return 0;
        }
        NodeGraphHash::combine(hash, modelARemeshProduct->productHash);
        NodeGraphHash::combine(hash, modelBRemeshProduct->productHash);
        return hash;
    }

    void removePublishedProduct(uint64_t socketKey) {
        if (socketKey == 0) {
            return;
        }

        auto entity = static_cast<ECSEntity>(socketKey);
        ecsRegistry->remove<ContactProduct>(entity);
    }

    void publishProduct(uint64_t socketKey) {
        if (!controller || socketKey == 0) {
            return;
        }

        ContactProduct product{};
        if (!buildProduct(socketKey, product)) {
            auto entity = static_cast<ECSEntity>(socketKey);
            ecsRegistry->remove<ContactProduct>(entity);
            return;
        }

        auto entity = static_cast<ECSEntity>(socketKey);
        const auto& package = ecsRegistry->get<ContactPackage>(entity);
        ecsRegistry->emplace_or_replace<ContactProduct>(entity, product);
        appliedConfigInputHash[socketKey] = buildConfigInputHash(socketKey, package);
    }

    bool buildProduct(uint64_t socketKey, ContactProduct& outProduct) const {
        outProduct = {};
        ContactSystem* system = controller ? controller->getSystem(socketKey) : nullptr;
        if (!system || !controller->getConfig(socketKey)) {
            return false;
        }

        const ContactCoupling* coupling = system->getContactCoupling();
        if (!coupling) {
            return false;
        }

        outProduct.coupling = *coupling;
        outProduct.contactPairBuffer = system->getContactPairBuffer();
        outProduct.contactPairBufferOffset = system->getContactPairBufferOffset();
        outProduct.modelARuntimeModelId = coupling->modelARuntimeModelId;
        outProduct.modelBRuntimeModelId = coupling->modelBRuntimeModelId;
        outProduct.outlineVertices = system->getOutlineVertices();
        outProduct.correspondenceVertices = system->getCorrespondenceVertices();
        outProduct.productHash = buildProductHash(outProduct);
        return outProduct.isValid();
    }

    ContactSystemComputeController* controller = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
    std::unordered_map<uint64_t, uint64_t> appliedConfigInputHash;
};
