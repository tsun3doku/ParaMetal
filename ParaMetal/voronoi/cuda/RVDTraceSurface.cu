#include "RVDTraceSurface.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <unordered_map>

namespace voronoi {

namespace {

glm::mat4 modelMatrixFromArray(const std::array<float, 16> *values) {
  if (!values) return glm::mat4(1.0f);
  const std::array<float, 16> &m = *values;
  return glm::mat4(
      m[0], m[1], m[2], m[3],
      m[4], m[5], m[6], m[7],
      m[8], m[9], m[10], m[11],
      m[12], m[13], m[14], m[15]);
}

} // namespace

void buildVemTraceSurfaceMeshes(
    const std::vector<ModelGeometry> *geometries,
    const std::vector<SurfaceRegion> &surfaceRegions,
    std::vector<VemTraceSurfaceMesh> &outMeshes) {
  outMeshes.clear();
  if (geometries == nullptr || geometries->empty() || surfaceRegions.empty()) {
    return;
  }

  std::unordered_map<uint32_t, const ModelGeometry *> geomLookup;
  for (const ModelGeometry &g : *geometries) {
    geomLookup[g.runtimeModelId] = &g;
  }

  std::unordered_map<uint32_t, size_t> meshIndexByModel;

  for (const SurfaceRegion &reg : surfaceRegions) {
    const size_t polySize = reg.polygon.size();
    if (polySize < 3 || reg.faceId == InvalidId) continue;

    const auto geomIt = geomLookup.find(reg.runtimeModelId);
    if (geomIt == geomLookup.end()) continue;

    const ModelGeometry &source = *geomIt->second;
    const glm::mat4 worldToModel = modelMatrixFromArray(source.worldToModel);
    const glm::mat3 normalToModel = glm::transpose(glm::mat3(worldToModel));
    glm::vec3 localNormal = normalToModel * glm::vec3(reg.normal);
    if (glm::length(localNormal) > 1e-12f) {
      localNormal = glm::normalize(localNormal);
    } else {
      localNormal = glm::vec3(0.0f, 0.0f, 1.0f);
    }

    auto meshIt = meshIndexByModel.find(reg.runtimeModelId);
    if (meshIt == meshIndexByModel.end()) {
      VemTraceSurfaceMesh newMesh;
      newMesh.runtimeModelId = reg.runtimeModelId;
      outMeshes.push_back(std::move(newMesh));
      meshIt = meshIndexByModel.emplace(
          reg.runtimeModelId,
          outMeshes.size() - 1).first;
    }
    VemTraceSurfaceMesh &meshData = outMeshes[meshIt->second];

    const auto toLocal = [&](const glm::dvec3 &world) {
      const glm::vec4 transformed = worldToModel * glm::vec4(
          float(world.x), float(world.y), float(world.z), 1.0f);
      return glm::vec3(transformed);
    };

    const glm::vec3 p0 = toLocal(reg.polygon[0]);
    for (size_t v = 1; v + 1 < polySize; ++v) {
      meshData.vertices.push_back({p0, localNormal, reg.faceId});
      meshData.vertices.push_back({toLocal(reg.polygon[v]), localNormal, reg.faceId});
      meshData.vertices.push_back({toLocal(reg.polygon[v + 1]), localNormal, reg.faceId});
    }
  }
}

} // namespace voronoi
