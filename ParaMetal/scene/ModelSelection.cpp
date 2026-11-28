#include "ModelSelection.hpp"
#include "PickReadback.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "framegraph/VkFrameGraphRuntime.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "Model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

ModelSelection::ModelSelection(
    VulkanDevice& device,
    VkFrameGraphRuntime& runtime,
    ModelRegistry& modelRegistry,
    framegraph::ResourceId pickResourceId)
    : vulkanDevice(device), frameGraphRuntime(runtime), modelRegistry(modelRegistry), pickResourceId(pickResourceId),
      pickReadback(std::make_unique<PickReadback>(device)) {
    if (!pickReadback->isInitialized()) {
        std::cerr << "[ModelSelection] Initialization failed" << std::endl;
        cleanup();
        return;
    }
    initialized = true;
}

ModelSelection::~ModelSelection() {
    cleanup();
}

bool ModelSelection::getSelected() const {
    return !selectedModelIDs.empty();
}

void ModelSelection::addSelectedModelID(uint32_t id) {
    if (std::find(selectedModelIDs.begin(), selectedModelIDs.end(), id) == selectedModelIDs.end()) {
        selectedModelIDs.push_back(id);
    }
}

void ModelSelection::clearSelection() {
    selectedModelIDs.clear();
}

bool ModelSelection::isModelSelected(uint32_t id) const {
    return std::find(selectedModelIDs.begin(), selectedModelIDs.end(), id) != selectedModelIDs.end();
}

const std::vector<uint32_t>& ModelSelection::getSelectedModelIDsRenderThread() const {
    return selectedModelIDs;
}

uint32_t ModelSelection::getSelectedModelID() const {
    return selectedModelIDs.empty() ? 0 : selectedModelIDs.front();
}

void ModelSelection::setOutlineColor(const glm::vec3& color) {
    outlineColor = color;
}

glm::vec3 ModelSelection::getOutlineColor() const {
    return outlineColor;
}

void ModelSelection::setOutlineThickness(float thickness) {
    outlineThickness = thickness;
}

float ModelSelection::getOutlineThickness() const {
    return outlineThickness;
}

void ModelSelection::queuePickRequest(int x, int y, bool shiftPressed, float mouseX, float mouseY) {
    PickingRequest request;
    request.x = x;
    request.y = y;
    request.shiftPressed = shiftPressed;
    request.mouseX = mouseX;
    request.mouseY = mouseY;
    
    std::lock_guard<std::mutex> lock(pickingQueueMutex);
    pickingRequestQueue.push(request);
}

void ModelSelection::processPickingRequests(uint32_t currentFrame) {
    if (!initialized) {
        return;
    }
    
    while (true) {
        PickingRequest request;
        {
            std::lock_guard<std::mutex> lock(pickingQueueMutex);
            if (pickingRequestQueue.empty()) {
                break;
            }
            request = pickingRequestQueue.front();
            pickingRequestQueue.pop();
        }
        
        PickedResult result = pickAtPosition(request.x, request.y, currentFrame);
        lastPickedResult = result;  
        lastPickRequest = request; 
        pendingPicks.push_back(ModelSelectionPick{result, request.shiftPressed});
    }
}

std::vector<ModelSelectionPick> ModelSelection::takePendingPicks() {
    std::vector<ModelSelectionPick> picks;
    picks.swap(pendingPicks);
    return picks;
}

void ModelSelection::discardPendingPicks() {
    {
        std::lock_guard<std::mutex> lock(pickingQueueMutex);
        std::queue<PickingRequest> empty;
        pickingRequestQueue.swap(empty);
    }

    pendingPicks.clear();
    lastPickedResult = {};
}

PickedResult ModelSelection::pickAtPosition(int x, int y, uint32_t currentFrame) {
    if (!initialized || !pickReadback || !pickReadback->isInitialized()) {
        return PickedResult();
    }
    if (x < 0 || y < 0) {
        return PickedResult();
    }

    const auto& pickImages = frameGraphRuntime.getResourceImages(pickResourceId);
    if (currentFrame >= pickImages.size() || pickImages[currentFrame] == VK_NULL_HANDLE) {
        return PickedResult();
    }

    uint32_t pickId = 0;
    if (!pickReadback->readPixel(pickImages[currentFrame], x, y, pickId)) {
        return PickedResult();
    }

    PickedResult result;
    result.pickId = pickId;

    if (pickid::typeOf(pickId) == pickid::GizmoType) {
        const uint32_t axis = pickid::gizmoAxisOf(pickId);
        if (axis >= 1 && axis <= 3) {
            result.type = PickedType::Gizmo;
            result.gizmoAxis = static_cast<PickedGizmoAxis>(axis);
            result.gizmoMode = pickid::gizmoModeOf(pickId);
        }
    } else if (pickid::typeOf(pickId) == pickid::ModelType) {
        const uint32_t modelId = pickid::payloadOf(pickId);
        if (!modelRegistry.hasModel(modelId)) {
            result.type = PickedType::None;
            return result;
        }
        result.type = PickedType::Model;
        result.modelID = modelId;
    } else {
        result.type = PickedType::None;
    }
    
    return result;
}

void ModelSelection::cleanup() {
    pickReadback.reset();
    initialized = false;
}


