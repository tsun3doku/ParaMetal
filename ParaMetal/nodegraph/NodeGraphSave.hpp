#pragma once

#include "NodeGraphTypes.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <QString>

class NodeGraphSave {
public:
    struct Viewport {
        glm::vec3 lookAt{0.0f};
        glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
        float radius = 2.0f;
        float fov = 45.0f;
    };

    struct Data {
        NodeGraphState graph;
        Viewport viewport;
        uint32_t nextNodeId = 1;
        uint32_t nextSocketId = 1;
        uint32_t nextEdgeId = 1;
    };

    static bool save(const Data& data, const QString& filePath, QString* outError = nullptr);
    static bool load(Data& outData, const QString& filePath, QString* outError = nullptr);
};
