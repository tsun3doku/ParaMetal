#include "ContactSystemComputeController.hpp"

#include "ContactSystem.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanDevice.hpp"

ContactSystemComputeController::ContactSystemComputeController(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    CommandPool& renderCommandPoolRef)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      renderCommandPool(renderCommandPoolRef) {
}

ContactSystemComputeController::~ContactSystemComputeController() {
    disableAll();
}

void ContactSystemComputeController::configure(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return;
    }

    auto& system = activeSystems[socketKey];
    if (!system) {
        system = std::make_unique<ContactSystem>(vulkanDevice, memoryAllocator, renderCommandPool);
    }

    const auto configIt = configuredConfigs.find(socketKey);
    if (configIt != configuredConfigs.end() && configIt->second.computeHash == config.computeHash) {
        return;
    }

    configuredConfigs[socketKey] = config;

    system->setParams(config.minNormalDot, config.contactRadius);
    system->setModelAState(config.modelALocalToWorld, config.modelAIntrinsicMesh, config.modelARuntimeModelId);
    system->setModelBState(config.modelBLocalToWorld, config.modelBIntrinsicMesh, config.modelBRuntimeModelId);
    system->setModelBTriangleIndices(config.modelBTriangleIndices);
    system->ensureConfigured();
}

void ContactSystemComputeController::disable(uint64_t socketKey) {
    configuredConfigs.erase(socketKey);
    auto it = activeSystems.find(socketKey);
    if (it != activeSystems.end()) {
        if (it->second) {
            it->second->disable();
        }
        activeSystems.erase(it);
    }
}

void ContactSystemComputeController::disableAll() {
    configuredConfigs.clear();
    for (auto& [key, system] : activeSystems) {
        if (system) {
            system->disable();
        }
    }
    activeSystems.clear();
}

ContactSystem* ContactSystemComputeController::getSystem(uint64_t socketKey) const {
    const auto it = activeSystems.find(socketKey);
    if (it == activeSystems.end() || !it->second) {
        return nullptr;
    }
    return it->second.get();
}

const ContactSystemComputeController::Config* ContactSystemComputeController::getConfig(uint64_t socketKey) const {
    const auto it = configuredConfigs.find(socketKey);
    return it != configuredConfigs.end() ? &it->second : nullptr;
}
