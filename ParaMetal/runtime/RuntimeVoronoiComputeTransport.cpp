#include "RuntimeVoronoiComputeTransport.hpp"
#include "hash/HashProduct.hpp"
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

    if (package.domainType == DomainType::Mesh) {
        if (package.modelMeshHandle.key == 0 || package.modelRemeshHandle.key == 0) {
            remove(socketKey);
            return {};
        }

        const RemeshProduct* remeshProduct = products->resolve<RemeshProduct>(package.remeshProduct);
        const ModelProduct* modelProduct = products->resolve<ModelProduct>(package.modelProduct);
        if (!remeshProduct || !modelProduct || remeshProduct->runtimeModelId == 0) {
            remove(socketKey);
            return {};
        }

        config.receiverRuntimeModelId = remeshProduct->runtimeModelId;
        config.receiverNodeModelId = 0;
        config.receiverGeometryPositions = remeshProduct->geometryPositions;
        config.receiverGeometryTriangleIndices = remeshProduct->geometryTriangleIndices;
        config.receiverIntrinsicMesh = remeshProduct->intrinsicMesh;
        config.receiverIntrinsicTriangleIndices = remeshProduct->intrinsicMesh.indices;
        config.receiverSurfaceVertices.reserve(remeshProduct->intrinsicMesh.vertices.size());
        for (const SupportingHalfedge::IntrinsicVertex& intrinsicVertex : remeshProduct->intrinsicMesh.vertices) {
            VoronoiModelRuntime::SurfaceVertex vertex{};
            vertex.position = glm::vec4(intrinsicVertex.position, 1.0f);
            vertex.normal = glm::vec4(intrinsicVertex.normal, 0.0f);
            config.receiverSurfaceVertices.push_back(vertex);
        }
        config.meshModelMatrix = toMat4(package.modelLocalToWorld);
        config.pointPositions = package.pointPositions;
    } else if (package.domainType == DomainType::Points) {
        if (package.pointPositions.empty()) {
            controller->remove(socketKey);
            return {};
        }
        config.isPointDomain = true;
        config.pointPositions = package.pointPositions;
    }

    config.computeHash = computeHash;

    controller->apply(socketKey, config);

    VoronoiProduct product{};
    if (!controller->buildProduct(socketKey, product)) {
        return {};
    }

    ProductHandle handle = products->publish<VoronoiProduct>(socketKey, product);
    return handle;
}

void RuntimeVoronoiComputeTransport::remove(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->remove(socketKey);
}
