#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>
#include <memory>

#include "domain/HeatProbeData.hpp"

class HeatProbeRuntime {
public:
    struct FragmentCentroid {
        uint32_t runtimeModelId = 0;
        uint32_t localIndex = 0;
        glm::vec3 worldCentroid{0.0f};
    };

    struct StencilEntry {
        uint32_t runtimeModelId = 0;
        uint32_t localIndex = 0;
        float weight = 0.0f;
    };

    struct ProbeChannel {
        uint32_t id = 0;
        std::string name;
        glm::vec3 worldPosition{0.0f};
        bool enabled = true;
        float currentTemperatureC = 20.0f;
        std::vector<StencilEntry> stencilWeights;
        bool weightsDirty = true;
    };

    HeatProbeRuntime();
    ~HeatProbeRuntime();

    HeatProbeRuntime(const HeatProbeRuntime&) = delete;
    HeatProbeRuntime& operator=(const HeatProbeRuntime&) = delete;

    void configure(
        bool enabled,
        uint32_t opcPort,
        const std::vector<HeatProbeItem>& items);

    bool rebuildWeights(
        const std::vector<FragmentCentroid>& domainCentroids,
        const std::unordered_map<uint32_t, std::array<float, 16>>& modelLocalToWorld);

    void sample(const std::unordered_map<uint32_t, std::vector<float>>& temperaturesByModel);
    void updateServer();

    void cleanup();

    bool isEnabled() const { return enabled; }
    uint32_t getOpcPort() const { return opcPort; }
    const std::vector<ProbeChannel>& getChannels() const { return channels; }

    float getTemperatureC(size_t index = 0) const {
        return (index < channels.size()) ? channels[index].currentTemperatureC : 20.0f;
    }

    const float* temperatureData() const { return temperaturesC.data(); }
    std::size_t temperatureCount() const { return temperaturesC.size(); }

    bool hasDirtyWeights() const {
        for (const auto& ch : channels) {
            if (ch.weightsDirty || ch.stencilWeights.empty()) return true;
        }
        return false;
    }

private:
    void startServer();
    void stopServer();

    bool enabled = true;
    uint32_t opcPort = 4840;
    std::vector<ProbeChannel> channels;
    std::vector<float> temperaturesC;

    class ServerImpl;
    std::unique_ptr<ServerImpl> server;
};
