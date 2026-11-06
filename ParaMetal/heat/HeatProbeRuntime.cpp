#include "HeatProbeRuntime.hpp"
#include "util/GMLS.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#pragma comment(lib, "ws2_32.lib")

class HeatProbeRuntime::ServerImpl {
public:
    SOCKET listenSocket = INVALID_SOCKET;
    uint32_t activePort = 0;
    std::vector<SOCKET> clientSockets;
    bool wsaInitialized = false;

    ServerImpl() {
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) == 0) {
            wsaInitialized = true;
        }
    }

    ~ServerImpl() {
        stop();
        if (wsaInitialized) {
            WSACleanup();
        }
    }

    bool start(uint32_t port) {
        if (!wsaInitialized) return false;
        if (listenSocket != INVALID_SOCKET && activePort == port) return true;

        stop();

        listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listenSocket == INVALID_SOCKET) {
            return false;
        }

        u_long nonBlocking = 1;
        ioctlsocket(listenSocket, FIONBIO, &nonBlocking);

        BOOL reuse = TRUE;
        setsockopt(listenSocket, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse));

        sockaddr_in serverAddr{};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
        serverAddr.sin_port = htons(static_cast<u_short>(port));

        if (bind(listenSocket, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == SOCKET_ERROR) {
            closesocket(listenSocket);
            listenSocket = INVALID_SOCKET;
            return false;
        }

        if (listen(listenSocket, 8) == SOCKET_ERROR) {
            closesocket(listenSocket);
            listenSocket = INVALID_SOCKET;
            return false;
        }

        activePort = port;
        return true;
    }

    void stop() {
        for (SOCKET client : clientSockets) {
            if (client != INVALID_SOCKET) {
                closesocket(client);
            }
        }
        clientSockets.clear();

        if (listenSocket != INVALID_SOCKET) {
            closesocket(listenSocket);
            listenSocket = INVALID_SOCKET;
        }
        activePort = 0;
    }

    void update(const std::vector<ProbeChannel>& channels) {
        if (listenSocket == INVALID_SOCKET) return;

        sockaddr_in clientAddr{};
        int clientAddrLen = sizeof(clientAddr);
        SOCKET newClient = accept(listenSocket, reinterpret_cast<sockaddr*>(&clientAddr), &clientAddrLen);
        if (newClient != INVALID_SOCKET) {
            u_long nonBlocking = 1;
            ioctlsocket(newClient, FIONBIO, &nonBlocking);
            clientSockets.push_back(newClient);
        }

        const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        std::ostringstream json;
        json << "{\"timestampMs\":" << now << ",\"probes\":{";
        for (size_t i = 0; i < channels.size(); ++i) {
            const auto& ch = channels[i];
            if (i > 0) json << ",";
            json << "\"" << ch.name << "\":" << ch.currentTemperatureC;
        }
        json << "},\"items\":[";
        for (size_t i = 0; i < channels.size(); ++i) {
            const auto& ch = channels[i];
            if (i > 0) json << ",";
            json << "{\"id\":" << ch.id
                 << ",\"name\":\"" << ch.name << "\""
                 << ",\"temperatureC\":" << ch.currentTemperatureC
                 << ",\"x\":" << ch.worldPosition.x
                 << ",\"y\":" << ch.worldPosition.y
                 << ",\"z\":" << ch.worldPosition.z
                 << ",\"enabled\":" << (ch.enabled ? "true" : "false") << "}";
        }
        json << "]}\n";

        const std::string jsonPayload = json.str();
        const int jsonLen = static_cast<int>(jsonPayload.size());

        char buffer[1024];

        for (auto it = clientSockets.begin(); it != clientSockets.end();) {
            SOCKET client = *it;
            int bytesRecv = recv(client, buffer, sizeof(buffer) - 1, 0);

            if (bytesRecv > 0) {
                buffer[bytesRecv] = '\0';
                if (strstr(buffer, "GET") != nullptr) {
                    std::ostringstream http;
                    http << "HTTP/1.1 200 OK\r\n"
                         << "Content-Type: application/json\r\n"
                         << "Access-Control-Allow-Origin: *\r\n"
                         << "Content-Length: " << jsonLen << "\r\n"
                         << "Connection: close\r\n\r\n"
                         << jsonPayload;
                    const std::string httpResponse = http.str();
                    send(client, httpResponse.data(), static_cast<int>(httpResponse.size()), 0);
                    closesocket(client);
                    it = clientSockets.erase(it);
                    continue;
                }
            } else if (bytesRecv == 0) {
                closesocket(client);
                it = clientSockets.erase(it);
                continue;
            } else {
                int err = WSAGetLastError();
                if (err != WSAEWOULDBLOCK) {
                    closesocket(client);
                    it = clientSockets.erase(it);
                    continue;
                }
            }

            int sendRes = send(client, jsonPayload.data(), jsonLen, 0);
            if (sendRes == SOCKET_ERROR) {
                int err = WSAGetLastError();
                if (err != WSAEWOULDBLOCK) {
                    closesocket(client);
                    it = clientSockets.erase(it);
                    continue;
                }
            }
            ++it;
        }
    }
};

HeatProbeRuntime::HeatProbeRuntime() = default;

HeatProbeRuntime::~HeatProbeRuntime() {
    cleanup();
}

void HeatProbeRuntime::configure(
    bool isEnabled,
    uint32_t port,
    const std::vector<HeatProbeItem>& items) {
    const bool portChanged = (opcPort != port);
    enabled = isEnabled;
    opcPort = (port > 0) ? port : 4840;

    std::vector<ProbeChannel> updatedChannels;
    updatedChannels.reserve(items.size());

    for (const auto& item : items) {
        ProbeChannel ch{};
        ch.id = item.id;
        ch.name = item.name.empty() ? ("Probe_" + std::to_string(item.id)) : item.name;
        ch.worldPosition = item.worldPosition;
        ch.enabled = item.enabled;

        auto existingIt = std::find_if(channels.begin(), channels.end(),
            [&item](const ProbeChannel& c) { return c.id == item.id; });
        if (existingIt != channels.end()) {
            ch.currentTemperatureC = existingIt->currentTemperatureC;
            const bool posChanged = (glm::distance(existingIt->worldPosition, item.worldPosition) > 1e-5f);
            if (!posChanged && !existingIt->stencilWeights.empty()) {
                ch.stencilWeights = existingIt->stencilWeights;
                ch.weightsDirty = false;
            } else {
                ch.weightsDirty = true;
            }
        } else {
            ch.weightsDirty = true;
        }
        updatedChannels.push_back(std::move(ch));
    }

    channels = std::move(updatedChannels);

    if (!enabled) {
        stopServer();
    } else if (portChanged || !server) {
        startServer();
    }
}

void HeatProbeRuntime::startServer() {
    if (!server) {
        server = std::make_unique<ServerImpl>();
    }
    server->start(opcPort);
}

void HeatProbeRuntime::stopServer() {
    if (server) {
        server->stop();
    }
}

void HeatProbeRuntime::cleanup() {
    stopServer();
    server.reset();
    channels.clear();
}

bool HeatProbeRuntime::rebuildWeights(
    const topology::GlobalDomain& domain,
    const std::unordered_map<uint32_t, std::vector<uint32_t>>& fragmentIdsByModelId,
    const std::unordered_map<uint32_t, std::array<float, 16>>& modelLocalToWorld) {
    if (!domain.isValid() || fragmentIdsByModelId.empty() || channels.empty()) {
        return false;
    }

    struct CachedCentroid {
        uint32_t runtimeModelId = 0;
        uint32_t localIndex = 0;
        glm::vec3 worldCentroid{0.0f};
    };

    (void)modelLocalToWorld;

    std::vector<CachedCentroid> domainCentroids;
    for (const auto& [runtimeModelId, fragmentIds] : fragmentIdsByModelId) {
        for (uint32_t local = 0; local < static_cast<uint32_t>(fragmentIds.size()); ++local) {
            uint32_t globalFragId = fragmentIds[local];
            if (globalFragId >= domain.fragments.size()) continue;

            const glm::vec3 worldC = glm::vec3(domain.fragments[globalFragId].centroid);
            domainCentroids.push_back({runtimeModelId, local, worldC});
        }
    }

    if (domainCentroids.empty()) {
        return false;
    }

    bool anyRebuilt = false;

    for (auto& ch : channels) {
        if (!ch.weightsDirty && !ch.stencilWeights.empty()) {
            continue;
        }

        struct CandidateFragment {
            uint32_t runtimeModelId = 0;
            uint32_t localIndex = 0;
            glm::vec3 worldCentroid{0.0f};
            float distSq = 0.0f;
        };

        uint32_t targetModelId = 0;
        float minDistSq = std::numeric_limits<float>::max();
        for (const auto& dc : domainCentroids) {
            const glm::vec3 diff = dc.worldCentroid - ch.worldPosition;
            const float distSq = glm::dot(diff, diff);
            if (distSq < minDistSq) {
                minDistSq = distSq;
                targetModelId = dc.runtimeModelId;
            }
        }

        std::vector<CandidateFragment> candidates;
        candidates.reserve(domainCentroids.size());
        for (const auto& dc : domainCentroids) {
            if (dc.runtimeModelId != targetModelId) {
                continue;
            }
            const glm::vec3 diff = dc.worldCentroid - ch.worldPosition;
            const float distSq = glm::dot(diff, diff);
            candidates.push_back({dc.runtimeModelId, dc.localIndex, dc.worldCentroid, distSq});
        }

        constexpr size_t SupportCount = 20;
        const size_t k = std::min(SupportCount, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
            [](const CandidateFragment& a, const CandidateFragment& b) {
                return a.distSq < b.distSq;
            });

        std::vector<glm::dvec3> sourcePositions;
        sourcePositions.reserve(k);
        double maxDistSq = 0.0;
        for (size_t i = 0; i < k; ++i) {
            sourcePositions.push_back(glm::dvec3(candidates[i].worldCentroid));
            maxDistSq = std::max(maxDistSq, static_cast<double>(candidates[i].distSq));
        }

        const double kernelRadius = std::max(std::sqrt(maxDistSq) * 2.0, 1e-4);

        std::vector<double> computedValueWeights;
        std::vector<glm::dvec3> computedGradientWeights;
        const bool success = GMLS::computeSurfaceWeights(
            glm::dvec3(ch.worldPosition),
            sourcePositions,
            kernelRadius,
            computedValueWeights,
            computedGradientWeights);

        ch.stencilWeights.clear();

        if (success && computedValueWeights.size() == k) {
            double sum = 0.0;
            bool allFinite = true;
            for (double w : computedValueWeights) {
                if (!std::isfinite(w)) allFinite = false;
                sum += w;
            }

            if (allFinite && std::isfinite(sum)) {
                ch.stencilWeights.reserve(k);
                for (size_t i = 0; i < k; ++i) {
                    ch.stencilWeights.push_back({
                        candidates[i].runtimeModelId,
                        candidates[i].localIndex,
                        static_cast<float>(computedValueWeights[i])
                    });
                }
                ch.weightsDirty = false;
                anyRebuilt = true;
                continue;
            }
        }

        std::cerr << "[HeatProbeRuntime-Diag] GMLS stencil failed for probe '" << ch.name
                  << "' (id=" << ch.id << ") at position ("
                  << ch.worldPosition.x << ", " << ch.worldPosition.y << ", " << ch.worldPosition.z
                  << ") with support k=" << k << std::endl;
        ch.weightsDirty = false;
    }

    return anyRebuilt;
}

void HeatProbeRuntime::sample(const std::unordered_map<uint32_t, std::vector<float>>& temperaturesByModel) {
    if (!enabled || channels.empty()) {
        return;
    }

    for (auto& ch : channels) {
        if (!ch.enabled || ch.stencilWeights.empty()) {
            continue;
        }

        float sampledTemp = 0.0f;
        float weightSum = 0.0f;
        for (const auto& entry : ch.stencilWeights) {
            auto it = temperaturesByModel.find(entry.runtimeModelId);
            if (it != temperaturesByModel.end() && entry.localIndex < it->second.size()) {
                sampledTemp += entry.weight * it->second[entry.localIndex];
                weightSum += entry.weight;
            }
        }

        if (std::abs(weightSum) > 1e-5f) {
            ch.currentTemperatureC = sampledTemp / weightSum;
        }
    }
}

void HeatProbeRuntime::updateServer() {
    if (enabled && server && !channels.empty()) {
        server->update(channels);
    }
}
