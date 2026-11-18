#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <cstdint>

class Camera;
class Model;

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct RayHit {
    bool hit = false;
    float distance = std::numeric_limits<float>::max();
    glm::vec3 worldPosition{0.0f};
    glm::vec3 worldNormal{0.0f};
    uint32_t triangleIndex = 0;
};

struct SlabIntersection {
    bool hit = false;
    float tNear = 0.0f;
    float tFar = 0.0f;
    int nearFace = -1;
    int farFace = -1;
};

class MousePicker {
public:
    static Ray screenToWorldRay(
        float mouseX, 
        float mouseY, 
        uint32_t screenWidth, 
        uint32_t screenHeight,
        const glm::mat4& view, 
        const glm::mat4& proj);

    static Ray screenToWorldRay(
        float mouseX, 
        float mouseY, 
        uint32_t screenWidth, 
        uint32_t screenHeight,
        const Camera& camera);
    
    static bool rayIntersectsAABB(
        const Ray& ray, 
        const glm::vec3& aabbMin, 
        const glm::vec3& aabbMax);

    static SlabIntersection intersectSlab(
        const Ray& ray, 
        const glm::vec3& aabbMin, const glm::vec3& aabbMax);

    static bool rayIntersectsAABBFace(const Ray& ray, const glm::vec3& aabbMin, const glm::vec3& aabbMax,
                                      float& outT, int& outFace);
    
    static bool rayIntersectsSphere(const Ray& ray, const glm::vec3& center, float radius);

    static bool rayIntersectsTriangle(
        const Ray& ray,
        const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
        float& outT, glm::vec3& outNormal,
        bool cullBackFace = false);

    static bool raycastModel(
        const Ray& ray,
        const Model& model,
        const glm::mat4& modelMatrix,
        RayHit& outHit,
        bool cullBackFace = false);

    static bool raycastModel(
        const Ray& ray,
        const Model& model,
        RayHit& outHit,
        bool cullBackFace = false);
};
