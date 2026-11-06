#pragma once

#include "RVD.hpp"
#include "RVDSurfacePartition.hpp"

#include <cstdint>
#include <vector>

namespace voronoi {

// build VEM trace surface meshes from CAD surface regions
void buildVemTraceSurfaceMeshes(
    const std::vector<ModelGeometry> *geometries,
    const std::vector<SurfaceRegion> &surfaceRegions,
    std::vector<VemTraceSurfaceMesh> &outMeshes);

} // namespace voronoi
