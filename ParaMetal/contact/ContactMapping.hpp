#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

#include "contact/ContactTypes.hpp"

struct Quadrature {
    static constexpr uint32_t count = 7u;
    static constexpr glm::vec3 bary[count] = {
        glm::vec3(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f),
        glm::vec3(0.0597158717f, 0.4701420641f, 0.4701420641f),
        glm::vec3(0.4701420641f, 0.0597158717f, 0.4701420641f),
        glm::vec3(0.4701420641f, 0.4701420641f, 0.0597158717f),
        glm::vec3(0.7974269853f, 0.1012865073f, 0.1012865073f),
        glm::vec3(0.1012865073f, 0.7974269853f, 0.1012865073f),
        glm::vec3(0.1012865073f, 0.1012865073f, 0.7974269853f),
    };
    static constexpr float weights[count] = {
        0.2250000000f,
        0.1323941527f,
        0.1323941527f,
        0.1323941527f,
        0.1259391805f,
        0.1259391805f,
        0.1259391805f,
    };
};

struct ContactVertex {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f};
};

struct ContactTriangle {
    glm::vec3 normal{0.0f};
    float area = 0.0f;
    uint32_t vertexIndices[3]{};
};

struct ContactMesh {
    std::vector<ContactVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<ContactTriangle> triangles;

    bool isValid() const {
        return !vertices.empty() && !indices.empty() && !triangles.empty();
    }
};

ContactMesh buildContactMesh(
    const std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& normals,
    const std::vector<uint32_t>& indices);

void buildContactPairs(
    const ContactMesh& modelAMesh,
    const std::array<float, 16>& modelALocalToWorld,
    const ContactMesh& modelBMesh,
    const std::array<float, 16>& modelBLocalToWorld,
    std::vector<ContactPair>& contactPairs,
    std::vector<ContactLineVertex>& outOutlineVertices,
    std::vector<ContactLineVertex>& outCorrespondenceVertices,
    float contactRadius,
    float minNormalDot);