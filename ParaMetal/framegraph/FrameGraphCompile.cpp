#include <algorithm>
#include <cstdint>
#include <deque>
#include <queue>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "FrameGraphTypes.hpp"
#include "FrameGraph.hpp"

bool FrameGraph::hasLivePasses(const std::vector<uint8_t>& livePasses) const {
    for (uint8_t live : livePasses) {
        if (live != 0) {
            return true;
        }
    }
    return false;
}

bool FrameGraph::remapAttachmentRef(framegraph::AttachmentReference& ref, const std::vector<int32_t>& resourceRemap) const {
    if (framegraph::toIndex(ref.resourceId) >= resourceRemap.size()) {
        return false;
    }

    const int32_t mapped = resourceRemap[framegraph::toIndex(ref.resourceId)];
    if (mapped < 0) {
        return false;
    }

    ref.resourceId = static_cast<uint32_t>(mapped);
    return true;
}

void FrameGraph::addDependencyEdge(
    std::vector<std::unordered_set<uint32_t>>& adjacency,
    std::vector<uint32_t>& indegree,
    uint32_t src,
    uint32_t dst) const {
    if (src == dst) {
        return;
    }
    if (adjacency[src].insert(dst).second) {
        ++indegree[dst];
    }
}

bool FrameGraph::buildGraph(framegraph::ImageFormat swapchainImageFormat, framegraph::Extent2D extent) {
    resourceDecls.clear();
    passDecls.clear();
    resourceIdByName.clear();
    attachmentResourceOrder.clear();

    if (registeredResourceDecls.empty() || registeredPassDecls.empty()) {
        std::cerr << "[FrameGraph] No registered resources/passes" << std::endl;
        return false;
    }

    resourceDecls.reserve(registeredResourceDecls.size());
    for (size_t index = 0; index < registeredResourceDecls.size(); ++index) {
        ResourceDescription desc = registeredResourceDecls[index];
        desc.id = static_cast<uint32_t>(index);
        if (desc.memoryProperties == framegraph::MemoryProperty::None) {
            desc.memoryProperties = framegraph::MemoryProperty::DeviceLocal;
        }
        desc.extent = extent;
        if (desc.useSwapchainFormat) {
            desc.format = swapchainImageFormat;
        }
        if (desc.name.empty()) {
            std::cerr << "[FrameGraph] Resource has empty name" << std::endl;
            return false;
        }
        resourceDecls.push_back(desc);
    }

    passDecls = registeredPassDecls;
    for (size_t passIndex = 0; passIndex < passDecls.size(); ++passIndex) {
        passDecls[passIndex].id = static_cast<uint32_t>(passIndex);
    }

    if (!cullUnusedGraph()) {
        return false;
    }
    if (!compilePassDag()) {
        return false;
    }
    return true;
}

bool FrameGraph::cullUnusedGraph() {
    if (resourceDecls.empty() || passDecls.empty()) {
        return true;
    }

    const uint32_t resourceCount = static_cast<uint32_t>(resourceDecls.size());
    const uint32_t passCount = static_cast<uint32_t>(passDecls.size());

    std::vector<std::vector<framegraph::ResourceUse>> passUses(passCount);
    std::vector<std::vector<uint32_t>> writersByResource(resourceCount);
    for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
        passUses[passIndex] = buildPassResourceUses(passDecls[passIndex]);
        for (const framegraph::ResourceUse& use : passUses[passIndex]) {
            if (framegraph::toIndex(use.resourceId) >= resourceCount || !use.write) {
                continue;
            }
            writersByResource[framegraph::toIndex(use.resourceId)].push_back(passIndex);
        }
    }

    std::vector<uint8_t> liveResources(resourceCount, 0);
    std::vector<uint8_t> livePasses(passCount, 0);
    std::deque<uint32_t> resourceQueue;

    for (uint32_t resourceId = 0; resourceId < resourceCount; ++resourceId) {
        if (!resourceDecls[resourceId].isGraphOutput) {
            continue;
        }
        liveResources[resourceId] = 1;
        resourceQueue.push_back(resourceId);
    }

    while (!resourceQueue.empty()) {
        const uint32_t requiredResource = resourceQueue.front();
        resourceQueue.pop_front();

        for (uint32_t writerPass : writersByResource[requiredResource]) {
            if (writerPass >= passCount || livePasses[writerPass]) {
                continue;
            }

            livePasses[writerPass] = 1;
            for (const framegraph::ResourceUse& use : passUses[writerPass]) {
                if (framegraph::toIndex(use.resourceId) >= resourceCount || liveResources[framegraph::toIndex(use.resourceId)]) {
                    continue;
                }
                liveResources[framegraph::toIndex(use.resourceId)] = 1;
                resourceQueue.push_back(framegraph::toIndex(use.resourceId));
            }
        }
    }

    if (!hasLivePasses(livePasses)) {
        std::fill(livePasses.begin(), livePasses.end(), 1);
        std::fill(liveResources.begin(), liveResources.end(), 1);
    }

    std::vector<int32_t> resourceRemap(resourceCount, -1);
    std::vector<ResourceDescription> culledResources;
    culledResources.reserve(resourceCount);
    for (uint32_t resourceId = 0; resourceId < resourceCount; ++resourceId) {
        if (!liveResources[resourceId]) {
            continue;
        }
        ResourceDescription desc = resourceDecls[resourceId];
        const uint32_t newId = static_cast<uint32_t>(culledResources.size());
        resourceRemap[resourceId] = static_cast<int32_t>(newId);
        desc.id = newId;
        culledResources.push_back(std::move(desc));
    }

    std::vector<framegraph::PassDescription> culledPasses;
    culledPasses.reserve(passCount);
    for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
        if (!livePasses[passIndex]) {
            continue;
        }

        framegraph::PassDescription pass = passDecls[passIndex];

        std::vector<framegraph::AttachmentReference> remappedColors;
        remappedColors.reserve(pass.colors.size());
        for (framegraph::AttachmentReference ref : pass.colors) {
            if (remapAttachmentRef(ref, resourceRemap)) {
                remappedColors.push_back(std::move(ref));
            }
        }
        pass.colors = std::move(remappedColors);

        std::vector<framegraph::AttachmentReference> remappedResolves;
        remappedResolves.reserve(pass.resolves.size());
        for (framegraph::AttachmentReference ref : pass.resolves) {
            if (remapAttachmentRef(ref, resourceRemap)) {
                remappedResolves.push_back(std::move(ref));
            }
        }
        pass.resolves = std::move(remappedResolves);

        std::vector<framegraph::AttachmentReference> remappedInputs;
        remappedInputs.reserve(pass.inputs.size());
        for (framegraph::AttachmentReference ref : pass.inputs) {
            if (remapAttachmentRef(ref, resourceRemap)) {
                remappedInputs.push_back(std::move(ref));
            }
        }
        pass.inputs = std::move(remappedInputs);

        if (pass.depthStencil.has_value()) {
            framegraph::AttachmentReference ref = pass.depthStencil.value();
            if (remapAttachmentRef(ref, resourceRemap)) {
                pass.depthStencil = std::move(ref);
            }
            else {
                pass.depthStencil.reset();
            }
        }

        if (pass.depthResolve.has_value()) {
            framegraph::AttachmentReference ref = pass.depthResolve.value();
            if (remapAttachmentRef(ref, resourceRemap)) {
                pass.depthResolve = std::move(ref);
            }
            else {
                pass.depthResolve.reset();
            }
        }

        std::vector<framegraph::ResourceUse> remappedAdditionalUses;
        remappedAdditionalUses.reserve(pass.additionalUses.size());
        for (framegraph::ResourceUse use : pass.additionalUses) {
            if (framegraph::toIndex(use.resourceId) >= resourceRemap.size()) {
                continue;
            }
            const int32_t mapped = resourceRemap[framegraph::toIndex(use.resourceId)];
            if (mapped < 0) {
                continue;
            }
            use.resourceId = static_cast<uint32_t>(mapped);
            remappedAdditionalUses.push_back(use);
        }
        pass.additionalUses = std::move(remappedAdditionalUses);

        pass.id = static_cast<uint32_t>(culledPasses.size());
        culledPasses.push_back(std::move(pass));
    }

    resourceDecls = std::move(culledResources);
    passDecls = std::move(culledPasses);

    attachmentResourceOrder.clear();
    attachmentResourceOrder.reserve(resourceDecls.size());
    resourceIdByName.clear();
    for (uint32_t resourceId = 0; resourceId < resourceDecls.size(); ++resourceId) {
        ResourceDescription& desc = resourceDecls[resourceId];
        if (desc.name.empty()) {
            std::cerr << "[FrameGraph] Resource has empty name after culling" << std::endl;
            return false;
        }
        desc.id = resourceId;
        if (!resourceIdByName.emplace(desc.name, resourceId).second) {
            std::cerr << "[FrameGraph] Resource names must be unique" << std::endl;
            return false;
        }
        if (desc.isAttachment) {
            attachmentResourceOrder.push_back(resourceId);
        }
    }

    if (passDecls.empty()) {
        std::cerr << "[FrameGraph] Culling removed all passes" << std::endl;
        return false;
    }

    return true;
}

bool FrameGraph::compilePassDag() {
    passIdByName.clear();
    passSyncEdges.clear();

    const uint32_t passCount = static_cast<uint32_t>(passDecls.size());
    if (passCount == 0) {
        return true;
    }

    std::vector<std::unordered_set<uint32_t>> adjacency(passCount);
    std::vector<uint32_t> indegree(passCount, 0);

    std::vector<std::vector<framegraph::ResourceUse>> passUses(passCount);
    for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
        passUses[passIndex] = buildPassResourceUses(passDecls[passIndex]);
    }

    std::vector<framegraph::PassSyncEdge> preRemapPassSyncEdges;
    auto addPreRemapPassSyncEdge = [&preRemapPassSyncEdges](
        uint32_t srcPass,
        uint32_t dstPass,
        framegraph::ResourceId resourceId,
        framegraph::UsageType srcUsage,
        framegraph::UsageType dstUsage,
        bool srcWrite,
        bool dstWrite) {
            if (srcPass == dstPass) {
                return;
            }

            for (const framegraph::PassSyncEdge& existing : preRemapPassSyncEdges) {
                if (framegraph::toIndex(existing.srcPass) == srcPass &&
                    framegraph::toIndex(existing.dstPass) == dstPass &&
                    framegraph::toIndex(existing.resourceId) == framegraph::toIndex(resourceId) &&
                    existing.srcUsage == srcUsage &&
                    existing.dstUsage == dstUsage &&
                    existing.srcWrite == srcWrite &&
                    existing.dstWrite == dstWrite) {
                    return;
                }
            }

            preRemapPassSyncEdges.push_back({
                framegraph::PassId(srcPass),
                framegraph::PassId(dstPass),
                resourceId,
                srcUsage,
                dstUsage,
                srcWrite,
                dstWrite
            });
    };

    for (uint32_t resourceId = 0; resourceId < resourceDecls.size(); ++resourceId) {
        int32_t lastWriter = -1;
        std::vector<framegraph::ResourceUse> writerUses;
        std::vector<uint32_t> readersSinceLastWriter;
        std::vector<std::pair<uint32_t, framegraph::ResourceUse>> readerUsesSinceLastWriter;

        for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
            std::vector<framegraph::ResourceUse> usesForResource;
            usesForResource.reserve(4);
            for (const framegraph::ResourceUse& use : passUses[passIndex]) {
                if (framegraph::toIndex(use.resourceId) == resourceId) {
                    usesForResource.push_back(use);
                }
            }

            if (usesForResource.empty()) {
                continue;
            }

            bool hasReadUse = false;
            bool hasWriteUse = false;
            for (const framegraph::ResourceUse& use : usesForResource) {
                hasReadUse = hasReadUse || !use.write;
                hasWriteUse = hasWriteUse || use.write;
            }

            if (hasReadUse && lastWriter >= 0) {
                addDependencyEdge(adjacency, indegree, static_cast<uint32_t>(lastWriter), passIndex);
                for (const framegraph::ResourceUse& writerUse : writerUses) {
                    for (const framegraph::ResourceUse& readUse : usesForResource) {
                        if (readUse.write) {
                            continue;
                        }
                        addPreRemapPassSyncEdge(
                            static_cast<uint32_t>(lastWriter),
                            passIndex,
                            framegraph::ResourceId(resourceId),
                            writerUse.usage,
                            readUse.usage,
                            writerUse.write,
                            readUse.write);
                    }
                }
            }

            if (hasWriteUse) {
                if (lastWriter >= 0) {
                    addDependencyEdge(adjacency, indegree, static_cast<uint32_t>(lastWriter), passIndex);
                    for (const framegraph::ResourceUse& writerUse : writerUses) {
                        for (const framegraph::ResourceUse& dstUse : usesForResource) {
                            if (!dstUse.write) {
                                continue;
                            }
                            addPreRemapPassSyncEdge(
                                static_cast<uint32_t>(lastWriter),
                                passIndex,
                                framegraph::ResourceId(resourceId),
                                writerUse.usage,
                                dstUse.usage,
                                writerUse.write,
                                dstUse.write);
                        }
                    }
                }
                for (uint32_t readerPass : readersSinceLastWriter) {
                    addDependencyEdge(adjacency, indegree, readerPass, passIndex);
                }
                for (const auto& reader : readerUsesSinceLastWriter) {
                    for (const framegraph::ResourceUse& dstUse : usesForResource) {
                        if (!dstUse.write) {
                            continue;
                        }
                        addPreRemapPassSyncEdge(
                            reader.first,
                            passIndex,
                            framegraph::ResourceId(resourceId),
                            reader.second.usage,
                            dstUse.usage,
                            reader.second.write,
                            dstUse.write);
                    }
                }

                lastWriter = static_cast<int32_t>(passIndex);
                writerUses.clear();
                for (const framegraph::ResourceUse& use : usesForResource) {
                    if (use.write) {
                        writerUses.push_back(use);
                    }
                }
                readersSinceLastWriter.clear();
                readerUsesSinceLastWriter.clear();
            }

            if (hasReadUse) {
                if (readersSinceLastWriter.empty() || readersSinceLastWriter.back() != passIndex) {
                    readersSinceLastWriter.push_back(passIndex);
                }
                for (const framegraph::ResourceUse& use : usesForResource) {
                    if (use.write) {
                        continue;
                    }

                    bool duplicate = false;
                    for (const auto& reader : readerUsesSinceLastWriter) {
                        if (reader.first == passIndex &&
                            reader.second.usage == use.usage &&
                            reader.second.write == use.write) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        readerUsesSinceLastWriter.push_back({ passIndex, use });
                    }
                }
            }
        }
    }

    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> ready;
    for (uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
        if (indegree[passIndex] == 0) {
            ready.push(passIndex);
        }
    }

    std::vector<uint32_t> sortedOrder;
    sortedOrder.reserve(passCount);
    while (!ready.empty()) {
        const uint32_t passIndex = ready.top();
        ready.pop();
        sortedOrder.push_back(passIndex);

        for (uint32_t dstPass : adjacency[passIndex]) {
            if (--indegree[dstPass] == 0) {
                ready.push(dstPass);
            }
        }
    }

    if (sortedOrder.size() != passCount) {
        std::cerr << "[FrameGraph] Pass DAG has a cycle" << std::endl;
        return false;
    }

    std::vector<framegraph::PassDescription> sortedPasses;
    std::vector<uint32_t> newIndexByOld(passCount, UINT32_MAX);
    sortedPasses.reserve(passCount);
    for (uint32_t sortedIndex = 0; sortedIndex < passCount; ++sortedIndex) {
        newIndexByOld[sortedOrder[sortedIndex]] = sortedIndex;
        framegraph::PassDescription passDesc = std::move(passDecls[sortedOrder[sortedIndex]]);
        passDesc.id = sortedIndex;
        if (!passIdByName.emplace(passDesc.name, sortedIndex).second) {
            std::cerr << "[FrameGraph] Pass names must be unique" << std::endl;
            return false;
        }
        sortedPasses.push_back(std::move(passDesc));
    }

    passSyncEdges.clear();

    for (const framegraph::PassSyncEdge& edge : preRemapPassSyncEdges) {
        const uint32_t srcNew = newIndexByOld[framegraph::toIndex(edge.srcPass)];
        const uint32_t dstNew = newIndexByOld[framegraph::toIndex(edge.dstPass)];
        if (srcNew == UINT32_MAX || dstNew == UINT32_MAX || srcNew == dstNew) {
            continue;
        }

        passSyncEdges.push_back({
            framegraph::PassId(srcNew),
            framegraph::PassId(dstNew),
            edge.resourceId,
            edge.srcUsage,
            edge.dstUsage,
            edge.srcWrite,
            edge.dstWrite
        });
    }

    passDecls = std::move(sortedPasses);
    return true;
}

std::vector<framegraph::ResourceUse> FrameGraph::buildPassResourceUses(const framegraph::PassDescription& passDesc) const {
    std::vector<framegraph::ResourceUse> uses;
    uses.reserve(passDesc.colors.size() + passDesc.resolves.size() + passDesc.inputs.size() + passDesc.additionalUses.size() + 2);

    for (const framegraph::AttachmentReference& ref : passDesc.colors) {
        uses.push_back({ ref.resourceId, framegraph::UsageType::ColorAttachment, true });
    }
    for (const framegraph::AttachmentReference& ref : passDesc.resolves) {
        uses.push_back({ ref.resourceId, framegraph::UsageType::ColorAttachment, true });
    }
    for (const framegraph::AttachmentReference& ref : passDesc.inputs) {
        uses.push_back({ ref.resourceId, framegraph::UsageType::InputAttachment, false });
    }
    if (passDesc.depthStencil.has_value()) {
        uses.push_back({ passDesc.depthStencil->resourceId, framegraph::UsageType::DepthStencilAttachment, !passDesc.depthReadOnly });
    }
    if (passDesc.depthResolve.has_value()) {
        uses.push_back({ passDesc.depthResolve->resourceId, framegraph::UsageType::DepthStencilAttachment, true });
    }

    for (const framegraph::ResourceUse& use : passDesc.additionalUses) {
        bool duplicate = false;
        for (const framegraph::ResourceUse& existing : uses) {
            if (framegraph::toIndex(existing.resourceId) == framegraph::toIndex(use.resourceId) &&
                existing.usage == use.usage &&
                existing.write == use.write) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            uses.push_back(use);
        }
    }

    return uses;
}
