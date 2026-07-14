#include "ContactSystemComputeController.hpp"

#include "ContactSystem.hpp"
#include "hash/HashProduct.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanDevice.hpp"

ContactSystemComputeController::ContactSystemComputeController(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    CommandPool& commandPoolRef)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      commandPool(commandPoolRef) {
}

ContactSystemComputeController::~ContactSystemComputeController() {
    disableAll();
}

void ContactSystemComputeController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return;
    }

    auto& system = systemsBySocket[socketKey];
    if (!system) {
        system = std::make_unique<ContactSystem>(vulkanDevice, memoryAllocator, commandPool);
    }

    const auto configIt = configuredConfigs.find(socketKey);
    if (configIt != configuredConfigs.end() && configIt->second.computeHash == config.computeHash) {
        return;
    }

    configuredConfigs[socketKey] = config;

    system->setParams(config.minNormalDot, config.contactRadius);
    system->setModelAState(config.modelALocalToWorld, config.modelAMesh, config.modelARuntimeModelId);
    system->setModelBState(config.modelBLocalToWorld, config.modelBMesh, config.modelBRuntimeModelId);
    system->ensureConfigured();
}

void ContactSystemComputeController::remove(uint64_t socketKey) {
    configuredConfigs.erase(socketKey);
    auto it = systemsBySocket.find(socketKey);
    if (it != systemsBySocket.end()) {
        if (it->second) {
            it->second->disable();
        }
        systemsBySocket.erase(it);
    }
}


void ContactSystemComputeController::disableAll() {
    configuredConfigs.clear();
    for (auto& [key, system] : systemsBySocket) {
        if (system) {
            system->disable();
        }
    }
    systemsBySocket.clear();
}

bool ContactSystemComputeController::buildProduct(uint64_t socketKey, ContactProduct& outProduct) const {
    outProduct = {};
    ContactSystem* system = getSystem(socketKey);
    if (!system || !getConfig(socketKey)) {
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
    HashProduct::seal(outProduct);
    return outProduct.isValid();
}

ContactSystem* ContactSystemComputeController::getSystem(uint64_t socketKey) const {
    const auto it = systemsBySocket.find(socketKey);
    if (it == systemsBySocket.end() || !it->second) {
        return nullptr;
    }
    return it->second.get();
}

const ContactSystemComputeController::Config* ContactSystemComputeController::getConfig(uint64_t socketKey) const {
    const auto it = configuredConfigs.find(socketKey);
    return it != configuredConfigs.end() ? &it->second : nullptr;
}
