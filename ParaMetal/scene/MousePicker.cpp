#include "MousePicker.hpp"
#include "Camera.hpp"
#include "Model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

Ray MousePicker::screenToWorldRay(float mouseX, float mouseY, uint32_t screenWidth, uint32_t screenHeight,
                                    const glm::mat4& view, const glm::mat4& proj) {
    const float x = (2.0f * mouseX) / screenWidth - 1.0f;
    const float y = (proj[1][1] < 0.0f)
        ? (2.0f * mouseY) / screenHeight - 1.0f
        : 1.0f - (2.0f * mouseY) / screenHeight;
    float z = 1.0f;  // Far plane
    
    glm::vec3 rayNDS = glm::vec3(x, y, z);
    
    // Convert to clip space
    glm::vec4 rayClip = glm::vec4(rayNDS.x, rayNDS.y, -1.0f, 1.0f);
    
    // Convert to eye/view space
    glm::vec4 rayEye = glm::inverse(proj) * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
    
    // Convert to world space
    glm::vec3 rayWorld = glm::vec3(glm::inverse(view) * rayEye);
    rayWorld = glm::normalize(rayWorld);
    
    // Ray origin is camera position
    glm::vec3 cameraPos = glm::vec3(glm::inverse(view)[3]);
    
    return Ray{ cameraPos, rayWorld };
}

Ray MousePicker::screenToWorldRay(float mouseX, float mouseY, uint32_t screenWidth, uint32_t screenHeight,
                                  const Camera& camera) {
    return Ray{
        camera.screenToWorldRayOrigin(mouseX, mouseY, screenWidth, screenHeight),
        camera.screenToWorldRay(mouseX, mouseY, screenWidth, screenHeight)
    };
}

bool MousePicker::rayIntersectsAABB(const Ray& ray, const glm::vec3& aabbMin, const glm::vec3& aabbMax) {
    float tMin = 0.0f;
    float tMax = std::numeric_limits<float>::max();
    
    for (int i = 0; i < 3; i++) {
        if (std::abs(ray.direction[i]) < 0.0001f) {
            // Ray is parallel to slab
            if (ray.origin[i] < aabbMin[i] || ray.origin[i] > aabbMax[i]) {
                return false;
            }
        } else {
            float t1 = (aabbMin[i] - ray.origin[i]) / ray.direction[i];
            float t2 = (aabbMax[i] - ray.origin[i]) / ray.direction[i];
            
            if (t1 > t2) std::swap(t1, t2);
            
            tMin = std::max(tMin, t1);
            tMax = std::min(tMax, t2);
            
            if (tMin > tMax) {
                return false;
            }
        }
    }
    
    return true;
}

bool MousePicker::rayIntersectsSphere(const Ray& ray, const glm::vec3& center, float radius) {
    glm::vec3 oc = ray.origin - center;
    float a = glm::dot(ray.direction, ray.direction);
    float b = 2.0f * glm::dot(oc, ray.direction);
    float c = glm::dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    
    return discriminant >= 0.0f;
}

bool MousePicker::rayIntersectsTriangle(
    const Ray& ray,
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    float& outT, glm::vec3& outNormal,
    bool cullBackFace)
{
    constexpr float EPSILON = 1e-7f;
    const glm::vec3 edge1 = v1 - v0;
    const glm::vec3 edge2 = v2 - v0;
    const glm::vec3 pvec = glm::cross(ray.direction, edge2);
    const float det = glm::dot(edge1, pvec);

    if (cullBackFace) {
        if (det < EPSILON) return false;
    } else {
        if (std::abs(det) < EPSILON) return false;
    }

    const float invDet = 1.0f / det;
    const glm::vec3 tvec = ray.origin - v0;
    const float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    const glm::vec3 qvec = glm::cross(tvec, edge1);
    const float v = glm::dot(ray.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) return false;

    const float t = glm::dot(edge2, qvec) * invDet;
    if (t <= EPSILON) return false;

    outT = t;
    glm::vec3 geoNormal = glm::cross(edge1, edge2);
    const float len = glm::length(geoNormal);
    if (len > 0.0f) {
        geoNormal /= len;
    }
    if (glm::dot(geoNormal, ray.direction) > 0.0f) {
        geoNormal = -geoNormal;
    }
    outNormal = geoNormal;
    return true;
}

bool MousePicker::raycastModel(
    const Ray& ray,
    const Model& model,
    const glm::mat4& modelMatrix,
    RayHit& outHit,
    bool cullBackFace)
{
    const auto& vertices = !model.getRenderVertices().empty() ? model.getRenderVertices() : model.getVertices();
    const auto& indices = !model.getRenderIndices().empty() ? model.getRenderIndices() : model.getIndices();
    if (indices.empty() || vertices.empty()) return false;

    // Check AABB early-out in local space
    glm::vec3 localMin(0.0f);
    glm::vec3 localMax(0.0f);
    const bool hasBounds = model.getLocalBounds(localMin, localMax);

    const glm::mat4 invMatrix = glm::inverse(modelMatrix);
    const glm::vec3 localOrigin = glm::vec3(invMatrix * glm::vec4(ray.origin, 1.0f));
    const glm::vec3 unnormalizedDir = glm::vec3(invMatrix * glm::vec4(ray.direction, 0.0f));
    const float dirLength = glm::length(unnormalizedDir);
    if (dirLength < 1e-7f) return false;

    Ray localRay{ localOrigin, unnormalizedDir / dirLength };

    if (hasBounds) {
        constexpr float BoundsEpsilon = 1e-4f;
        if (!rayIntersectsAABB(localRay, localMin - BoundsEpsilon, localMax + BoundsEpsilon)) {
            return false;
        }
    }

    bool hitAny = false;
    float closestLocalT = std::numeric_limits<float>::max();
    glm::vec3 bestLocalNormal{0.0f};
    uint32_t bestTriIndex = 0;

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t i0 = indices[i];
        const uint32_t i1 = indices[i + 1];
        const uint32_t i2 = indices[i + 2];
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size()) continue;

        const glm::vec3& v0 = vertices[i0].pos;
        const glm::vec3& v1 = vertices[i1].pos;
        const glm::vec3& v2 = vertices[i2].pos;

        float t = 0.0f;
        glm::vec3 normal{0.0f};
        if (rayIntersectsTriangle(localRay, v0, v1, v2, t, normal, cullBackFace)) {
            if (t < closestLocalT) {
                closestLocalT = t;
                bestLocalNormal = normal;
                bestTriIndex = static_cast<uint32_t>(i / 3);
                hitAny = true;
            }
        }
    }

    if (!hitAny) return false;

    const glm::vec3 localHitPoint = localRay.origin + localRay.direction * closestLocalT;
    const glm::vec3 worldHitPoint = glm::vec3(modelMatrix * glm::vec4(localHitPoint, 1.0f));
    const float worldDistance = glm::distance(ray.origin, worldHitPoint);

    if (worldDistance < outHit.distance) {
        outHit.hit = true;
        outHit.distance = worldDistance;
        outHit.worldPosition = worldHitPoint;
        glm::vec3 worldNormal = glm::normalize(glm::vec3(glm::transpose(invMatrix) * glm::vec4(bestLocalNormal, 0.0f)));
        if (glm::dot(worldNormal, ray.direction) > 0.0f) {
            worldNormal = -worldNormal;
        }
        outHit.worldNormal = worldNormal;
        outHit.triangleIndex = bestTriIndex;
        return true;
    }

    return false;
}

bool MousePicker::raycastModel(
    const Ray& ray,
    const Model& model,
    RayHit& outHit,
    bool cullBackFace)
{
    return raycastModel(ray, model, model.getModelMatrix(), outHit, cullBackFace);
}
