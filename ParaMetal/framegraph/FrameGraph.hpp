#pragma once

#include "FrameGraphTypes.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class FrameGraph {
public:
    FrameGraph();
    ~FrameGraph();

    void clearGraphDesc();
    framegraph::ResourceId addImageResource(framegraph::ImageResourceCreateInfo createInfo);
    void addPassDesc(framegraph::PassDescription passDesc);
    bool compile(framegraph::ImageFormat swapchainImageFormat, framegraph::Extent2D extent);
    framegraph::ResourceId getResourceId(std::string_view resourceName) const;
    framegraph::PassId getPassId(std::string_view passName) const;
    const framegraph::FrameGraphResult& getFrameGraphResult() const;
    std::vector<std::string> getPassNames() const;

private:
    using ResourceDescription = framegraph::ResourceDefinition;

    bool buildGraph(framegraph::ImageFormat swapchainImageFormat, framegraph::Extent2D extent);
    bool cullUnusedGraph();
    bool compilePassDag();

    std::vector<framegraph::ResourceUse> buildPassResourceUses(const framegraph::PassDescription& passDesc) const;
    bool hasLivePasses(const std::vector<uint8_t>& livePasses) const;
    bool remapAttachmentRef(framegraph::AttachmentReference& ref, const std::vector<int32_t>& resourceRemap) const;
    void addDependencyEdge(std::vector<std::unordered_set<uint32_t>>& adjacency, std::vector<uint32_t>& indegree, uint32_t src, uint32_t dst) const;
    framegraph::ResourceId addResourceDesc(ResourceDescription desc);
    void rebuildFrameGraphResult();

    std::vector<ResourceDescription> registeredResourceDecls;
    std::vector<framegraph::PassDescription> registeredPassDecls;
    std::vector<ResourceDescription> resourceDecls;
    std::vector<framegraph::PassDescription> passDecls;

    std::unordered_map<std::string, framegraph::ResourceId> resourceIdByName;
    std::unordered_map<std::string, framegraph::PassId> passIdByName;

    std::vector<framegraph::ResourceId> attachmentResourceOrder;
    std::vector<framegraph::PassSyncEdge> passSyncEdges;

    framegraph::FrameGraphResult frameGraphResult;
};


