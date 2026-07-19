#pragma once

#include <vulkan/vulkan.h>

#include "util/GlyphText.hpp"

#include <string>
#include <vector>

class ScreenTextRenderer;

class TimingRenderer {
public:
    explicit TimingRenderer(ScreenTextRenderer& textRenderer);
    ~TimingRenderer();

    void setLines(const std::vector<std::string>& lines);
    void render(VkCommandBuffer commandBuffer, uint32_t currentFrame, VkExtent2D extent);
    void cleanup();

private:
    void buildGlyphInstances();

    ScreenTextRenderer& textRenderer;
    std::vector<std::string> activeLines;
    std::vector<GlyphText::GlyphInstance> glyphInstances;
    uint32_t maxGlyphCapacity = 512;
};
