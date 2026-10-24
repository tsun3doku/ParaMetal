#include "SimGraphRuntime.hpp"

#include "SimNode.hpp"
#include "framegraph/ComputePass.hpp"
#include "framegraph/FrameSync.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <algorithm>
#include <iostream>
#include <utility>

SimGraphRuntime::~SimGraphRuntime() {
    cleanup();
}

bool SimGraphRuntime::initialize(VulkanDevice& inVulkanDevice, FrameSync& inFrameSync) {
    cleanup();

    vulkanDevice = &inVulkanDevice;
    frameSync = &inFrameSync;

    return true;
}

void SimGraphRuntime::cleanup() {
    clear();

    vulkanDevice = nullptr;
    frameSync = nullptr;
}

void SimGraphRuntime::registerNode(uint64_t socketKey, SimNode* node) {
    if (socketKey == 0 || !node) {
        return;
    }
    for (RegisteredNode& registered : nodes) {
        if (registered.socketKey == socketKey) {
            if (registered.node != node) {
                graphDirty = true;
            }
            registered.node = node;
            return;
        }
    }
    nodes.push_back({ socketKey, node });
    graphDirty = true;
}

void SimGraphRuntime::unregisterNode(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    for (size_t i = nodes.size(); i > 0; --i) {
        if (nodes[i - 1].socketKey == socketKey) {
            nodes.erase(nodes.begin() + static_cast<ptrdiff_t>(i - 1));
            graphDirty = true;
        }
    }
    for (size_t i = connections.size(); i > 0; --i) {
        const SimConnection& connection = connections[i - 1];
        if (connection.producerSocket == socketKey || connection.consumerSocket == socketKey) {
            connections.erase(connections.begin() + static_cast<ptrdiff_t>(i - 1));
        }
    }
}

void SimGraphRuntime::connect(const SimConnection& connection) {
    if (connection.connectionId == 0 || connection.producerSocket == 0 || connection.consumerSocket == 0) {
        return;
    }
    if (connection.mode != SimConnectionMode::Staged) {
        std::cerr << "[SimGraphRuntime] connection " << connection.connectionId
                  << " rejected: only staged connections are implemented" << std::endl;
        return;
    }
    if (connection.producerSocket == connection.consumerSocket) {
        std::cerr << "[SimGraphRuntime] connection " << connection.connectionId
                  << " rejected: self-connection" << std::endl;
        return;
    }

    for (SimConnection& existing : connections) {
        if (existing.connectionId == connection.connectionId) {
            existing = connection;
            graphDirty = true;
            return;
        }
    }
    for (const SimConnection& existing : connections) {
        if (existing.producerSocket == connection.producerSocket &&
            existing.consumerSocket == connection.consumerSocket) {
            std::cerr << "[SimGraphRuntime] connection " << connection.connectionId
                      << " rejected: socket pair already connected" << std::endl;
            return;
        }
    }

    connections.push_back(connection);
    graphDirty = true;
}

void SimGraphRuntime::disconnect(uint64_t connectionId) {
    if (connectionId == 0) {
        return;
    }
    for (size_t i = connections.size(); i > 0; --i) {
        if (connections[i - 1].connectionId == connectionId) {
            connections.erase(connections.begin() + static_cast<ptrdiff_t>(i - 1));
            graphDirty = true;
        }
    }
}

int SimGraphRuntime::findNodeIndex(uint64_t socketKey) const {
    const auto it = nodeIndexBySocket.find(socketKey);
    return it != nodeIndexBySocket.end() ? static_cast<int>(it->second) : -1;
}

bool SimGraphRuntime::hasPublication(const SimNode& node) const {
    return node.publicationReadySemaphore() != VK_NULL_HANDLE && node.publicationReadyValue() > 0;
}

bool SimGraphRuntime::rebuildGraph() {
    nodeIndexBySocket.clear();
    incomingConnectionsByNode.assign(nodes.size(), {});
    topologicalOrder.clear();
    orderValid = false;

    for (size_t i = 0; i < nodes.size(); ++i) {
        nodeIndexBySocket.emplace(nodes[i].socketKey, i);
    }

    std::vector<std::vector<size_t>> outgoing(nodes.size());
    std::vector<uint32_t> inDegree(nodes.size(), 0);
    for (size_t connectionIndex = 0; connectionIndex < connections.size(); ++connectionIndex) {
        const SimConnection& connection = connections[connectionIndex];
        const auto producerIt = nodeIndexBySocket.find(connection.producerSocket);
        const auto consumerIt = nodeIndexBySocket.find(connection.consumerSocket);
        if (producerIt == nodeIndexBySocket.end() || consumerIt == nodeIndexBySocket.end()) {
            continue; // dormant until both endpoints are registered
        }
        if (producerIt->second == consumerIt->second) {
            continue;
        }
        outgoing[producerIt->second].push_back(consumerIt->second);
        ++inDegree[consumerIt->second];
        incomingConnectionsByNode[consumerIt->second].push_back(connectionIndex);
    }

    std::vector<size_t> ready;
    ready.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (inDegree[i] == 0) {
            ready.push_back(i);
        }
    }

    size_t cursor = 0;
    while (cursor < ready.size()) {
        const size_t nodeIndex = ready[cursor++];
        topologicalOrder.push_back(nodeIndex);
        for (size_t downstream : outgoing[nodeIndex]) {
            if (inDegree[downstream] == 0) {
                continue;
            }
            if (--inDegree[downstream] == 0) {
                ready.push_back(downstream);
            }
        }
    }

    if (topologicalOrder.size() != nodes.size()) {
        std::cerr << "[SimGraphRuntime] simulation graph rejected: cycle or invalid dependency chain" << std::endl;
        topologicalOrder.clear();
        return false;
    }

    orderValid = true;
    return true;
}

void SimGraphRuntime::bindIncomingConnections(size_t nodeIndex) {
    SimNode* node = nodes[nodeIndex].node;
    node->clearConsumedInputs();

    for (size_t connectionIndex : incomingConnectionsByNode[nodeIndex]) {
        const SimConnection& connection = connections[connectionIndex];
        const int producerIndex = findNodeIndex(connection.producerSocket);
        if (producerIndex < 0) {
            continue;
        }
        SimNode* producer = nodes[static_cast<size_t>(producerIndex)].node;
        if (!hasPublication(*producer)) {
            continue;
        }
        node->bindConsumedInput(
            connection.connectionId,
            producer->publicationReadySemaphore(),
            producer->publicationReadyValue());
    }
}

SimTickResult SimGraphRuntime::tick(uint32_t frameIndex) {
    SimTickResult result{};

    if (vulkanDevice == nullptr || frameSync == nullptr) {
        return result;
    }

    // Slot fence before any work is scheduled, so a pass cannot write resources a
    // previous submission on the same slot is still using.
    frameSync->prepareExternalFrame(frameIndex);

    if (!enabled) {
        return result;
    }

    if (graphDirty) {
        graphDirty = false;
        rebuildGraph();
    }
    if (!orderValid) {
        return result;
    }

    for (size_t nodeIndex : topologicalOrder) {
        // Bind before the node updates: a consumer decides what it consumes from
        // the publications the graph has confirmed are available this frame.
        bindIncomingConnections(nodeIndex);
        nodes[nodeIndex].node->update();
    }

    result = recordAndSubmit(frameIndex);
    return result;
}

SimTickResult SimGraphRuntime::recordAndSubmit(uint32_t frameIndex) {
    SimTickResult result{};

    std::vector<size_t> activeNodes;
    activeNodes.reserve(topologicalOrder.size());
    for (size_t nodeIndex : topologicalOrder) {
        if (nodes[nodeIndex].node->hasDispatchableComputeWork()) {
            activeNodes.push_back(nodeIndex);
        }
    }

    if (activeNodes.empty()) {
        result.status = SimStatus::Idle;
        return result;
    }

    // Total GPU compute work performed by this frame's nodes, each from its own
    // stream events. Nodes may overlap, so this is work done, not wall time.
    float computeMs = 0.0f;
    uint64_t completedSteps = 0;
    for (size_t nodeIndex : activeNodes) {
        computeMs += nodes[nodeIndex].node->computeGpuMs();
        completedSteps += nodes[nodeIndex].node->completedStepCount();
    }
    result.computeGpuMs = computeMs;
    result.completedStepCount = completedSteps;

    std::vector<size_t> submissionIndexByNode(nodes.size(), SIZE_MAX);
    std::vector<FrameComputeSubmission> submissions;
    submissions.reserve(activeNodes.size());

    const size_t simulationCount = activeNodes.size();
    for (size_t i = 0; i < simulationCount; ++i) {
        const size_t nodeIndex = activeNodes[i];
        submissionIndexByNode[nodeIndex] = i;

        SimNode* node = nodes[nodeIndex].node;
        const std::vector<VkCommandBuffer>& commandBuffers = node->getComputeCommandBuffers();
        if (frameIndex >= commandBuffers.size()) {
            std::cerr << "[SimGraphRuntime] frame " << frameIndex
                      << " exceeds command buffer count for socket " << nodes[nodeIndex].socketKey << std::endl;
            result.status = SimStatus::Failed;
            return result;
        }
        VkCommandBuffer commandBuffer = commandBuffers[frameIndex];

        vkResetCommandBuffer(commandBuffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

        node->recordComputeCommands(commandBuffer, frameIndex);

        FrameComputeSubmission submission{};
        submission.commandBuffer = commandBuffer;
        submission.synchronization = node->getSynchronization();
        submissions.push_back(submission);
    }

    // Staged producer-ready waits and producer release signals are attached to the
    // consuming submission, but only for the connections the consumer actually
    // consumed. A producer's release value is signaled once, on its last consuming
    // submission: earlier copies are ordered before it on the same queue, and
    // signaling one timeline value twice would be invalid.
    struct PendingRelease {
        SimNode* producer = nullptr;
        uint64_t value = 0;
    };
    std::vector<PendingRelease> pendingReleases;
    std::vector<std::vector<std::pair<VkSemaphore, uint64_t>>> addedWaits(activeNodes.size());
    std::vector<size_t> lastConsumingSubmissionByNode(nodes.size(), SIZE_MAX);

    for (size_t connectionIndex = 0; connectionIndex < connections.size(); ++connectionIndex) {
        const SimConnection& connection = connections[connectionIndex];
        const int producerIndex = findNodeIndex(connection.producerSocket);
        const int consumerIndex = findNodeIndex(connection.consumerSocket);
        if (producerIndex < 0 || consumerIndex < 0 || producerIndex == consumerIndex) {
            continue;
        }

        SimNode* producer = nodes[static_cast<size_t>(producerIndex)].node;
        SimNode* consumer = nodes[static_cast<size_t>(consumerIndex)].node;
        if (!hasPublication(*producer)) {
            continue;
        }
        if (!consumer->consumedInput(connection.connectionId)) {
            continue;
        }

        const size_t submissionIndex = submissionIndexByNode[static_cast<size_t>(consumerIndex)];
        if (submissionIndex == SIZE_MAX) {
            // Consumed without a dispatchable submission: nothing will read the
            // publication, so no wait or release may be attached to it.
            continue;
        }

        const VkSemaphore readySemaphore = producer->publicationReadySemaphore();
        const uint64_t readyValue = producer->publicationReadyValue();
        std::vector<std::pair<VkSemaphore, uint64_t>>& waits = addedWaits[submissionIndex];
        const bool alreadyWaiting = std::find(waits.begin(), waits.end(),
            std::make_pair(readySemaphore, readyValue)) != waits.end();
        if (!alreadyWaiting) {
            submissions[submissionIndex].synchronization.addWait(
                readySemaphore, readyValue, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
            waits.push_back({ readySemaphore, readyValue });
        }

        size_t& lastSubmission = lastConsumingSubmissionByNode[static_cast<size_t>(producerIndex)];
        lastSubmission = (lastSubmission == SIZE_MAX)
            ? submissionIndex
            : std::max(lastSubmission, submissionIndex);
    }

    for (size_t producerIndex = 0; producerIndex < nodes.size(); ++producerIndex) {
        const size_t signalIndex = lastConsumingSubmissionByNode[producerIndex];
        if (signalIndex == SIZE_MAX) {
            continue;
        }

        SimNode* producer = nodes[producerIndex].node;
        if (!hasPublication(*producer)) {
            continue;
        }

        const uint64_t releaseValue = producer->peekPublicationRelease();
        submissions[signalIndex].synchronization.addSignal(
            producer->publicationReleaseSemaphore(), releaseValue);
        pendingReleases.push_back({ producer, releaseValue });
    }

    const VkResult submitResult = frameSync->submitComputeBatch(
        vulkanDevice->getComputeQueue(),
        submissions);
    if (submitResult != VK_SUCCESS) {
        std::cerr << "[SimGraphRuntime] compute submission failed VkResult="
                  << static_cast<int>(submitResult) << " frame=" << frameIndex << std::endl;
        result.status = submitResult == VK_ERROR_DEVICE_LOST ? SimStatus::DeviceLost : SimStatus::Failed;
        return result;
    }

    // The signaling submissions were accepted: advance each producer's timeline
    // and make its next step wait for the value it released.
    for (const PendingRelease& release : pendingReleases) {
        release.producer->commitPublicationRelease(release.value);
        release.producer->setPublicationReleaseWait(release.value);
    }

    result.status = SimStatus::Ok;
    return result;
}

void SimGraphRuntime::clear() {
    nodes.clear();
    connections.clear();
    nodeIndexBySocket.clear();
    topologicalOrder.clear();
    incomingConnectionsByNode.clear();
    graphDirty = true;
    orderValid = true;
}
