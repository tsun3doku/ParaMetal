#pragma once

#include <cuda_runtime.h>
#include "../../cuda/CudaBuffer.cuh"
#include "RVDGeometry.cuh"

namespace voronoi {

struct ContactManifoldMesh {
    cudaUtils::CudaBuffer<float3> vertices;
    cudaUtils::CudaBuffer<uint3> triangles;

    void reset() {
        vertices.reset();
        triangles.reset();
    }
};

bool extractContactManifold(
    cudaGeometry::DeviceModelGeometryView geomA,
    cudaGeometry::DeviceModelGeometryView geomB,
    float3 domainMin,
    float3 domainMax,
    float contactRange,
    float contactResolution,
    ContactManifoldMesh& outMesh,
    cudaStream_t stream);

} // namespace voronoi
