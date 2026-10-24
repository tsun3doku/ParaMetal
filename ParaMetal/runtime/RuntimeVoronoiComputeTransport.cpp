#include "RuntimeVoronoiComputeTransport.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "util/GeometryUtils.hpp"

ProductHandle RuntimeVoronoiComputeTransport::apply(uint64_t socketKey, const VoronoiPackage& package) {
    if (!controller || !products || socketKey == 0) {
        return {};
    }

    if (!package.authored.active) {
        remove(socketKey);
        return {};
    }

    const uint64_t computeHash = package.hashes.simulation;

    VoronoiSystemComputeController::Config config{};
    config.active = true;
    config.cellSize = package.authored.cellSize;
    config.voxelResolution = package.authored.voxelResolution;
    config.contactRange = package.authored.contactRange;
    config.pointDomainCorners = package.pointDomainCorners;

    if (package.pointPositions.empty() ||
        package.globalRemeshProducts.size() != package.globalRemeshLocalToWorld.size() ||
        package.globalRemeshProducts.size() != package.globalModelProducts.size()) {
        remove(socketKey);
        return {};
    }
    config.volumePadding = package.authored.volumePadding;
    config.pointPositions = package.pointPositions;
    config.globalRemeshRuntimeModelIds.reserve(package.globalRemeshProducts.size());
    config.globalRemeshPositions.reserve(package.globalRemeshProducts.size());
    config.globalRemeshTriangleIndices.reserve(package.globalRemeshProducts.size());
    config.globalRemeshSurfacePositions.reserve(package.globalRemeshProducts.size());
    config.globalRemeshSurfaceTriangleIndices.reserve(package.globalRemeshProducts.size());
    for (size_t i = 0; i < package.globalRemeshProducts.size(); ++i) {
        const RemeshProduct* remeshProduct = products->resolve<RemeshProduct>(package.globalRemeshProducts[i]);
        if (!remeshProduct || remeshProduct->runtimeModelId == 0) {
            remove(socketKey);
            return {};
        }
        const glm::mat4 transform = toMat4(package.globalRemeshLocalToWorld[i]);
        std::vector<glm::vec3> positions;
        positions.reserve(remeshProduct->geometryPositions.size());
        for (const glm::vec3& position : remeshProduct->geometryPositions) {
            positions.push_back(glm::vec3(transform * glm::vec4(position, 1.0f)));
        }
        std::vector<glm::vec3> surfacePositions;
        surfacePositions.reserve(remeshProduct->surfacePositions.size());
        for (const glm::vec3& position : remeshProduct->surfacePositions) {
            surfacePositions.push_back(glm::vec3(transform * glm::vec4(position, 1.0f)));
        }
        config.globalRemeshRuntimeModelIds.push_back(remeshProduct->runtimeModelId);
        config.globalRemeshPositions.push_back(std::move(positions));
        config.globalRemeshTriangleIndices.push_back(remeshProduct->geometryTriangleIndices);
        config.globalRemeshSurfacePositions.push_back(std::move(surfacePositions));
        config.globalRemeshSurfaceTriangleIndices.push_back(remeshProduct->surfaceTriangleIndices);
    }

    config.computeHash = computeHash;

    if (!controller->apply(socketKey, config)) {
        return {};
    }

    VoronoiProduct product{};
    if (!controller->buildProduct(socketKey, product)) {
        return {};
    }

    return products->publish<VoronoiProduct>(socketKey, product);
}

void RuntimeVoronoiComputeTransport::remove(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->remove(socketKey);
}
