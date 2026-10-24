#include "VoxelGrid.hpp"
#include "TriangleHashGrid.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <unordered_map>
#include <omp.h>

glm::vec3 VoxelGrid::toCanonical(const glm::vec3& worldPos) const {
    return (worldPos - params.gridMin) * params.cellSize;
}

bool VoxelGrid::triBoxOverlap(const float boxcenter[3], const float boxhalfsize[3], const float triverts[3][3]) {
    float v0[3], v1[3], v2[3];
    float e0[3], e1[3], e2[3];
    float min, max, p0, p1, p2, rad;
    float fex, fey, fez;
    float normal[3];

    // Make boxcenter (0,0,0)
    for (int i = 0; i < 3; i++) {
        v0[i] = triverts[0][i] - boxcenter[i];
        v1[i] = triverts[1][i] - boxcenter[i];
        v2[i] = triverts[2][i] - boxcenter[i];
    }

    // Calculate edges
    for (int i = 0; i < 3; i++) {
        e0[i] = v1[i] - v0[i];
        e1[i] = v2[i] - v1[i];
        e2[i] = v0[i] - v2[i];
    }

    // 9 axis tests: cross products of triangle edges with coordinate axes
    fex = std::fabs(e0[0]);
    fey = std::fabs(e0[1]);
    fez = std::fabs(e0[2]);

    // AXISTEST_X01
    p0 = e0[2] * v0[1] - e0[1] * v0[2];
    p2 = e0[2] * v2[1] - e0[1] * v2[2];

    if (p0 < p2) { min = p0; max = p2; 
    } else { 
        min = p2; max = p0; 
    }

    rad = fez * boxhalfsize[1] + fey * boxhalfsize[2];
    if (min > rad || max < -rad) 
        return false;

    // AXISTEST_Y02
    p0 = -e0[2] * v0[0] + e0[0] * v0[2];
    p2 = -e0[2] * v2[0] + e0[0] * v2[2];

    if (p0 < p2) { min = p0; max = p2; 
    } else { 
        min = p2; max = p0; 
    }

    rad = fez * boxhalfsize[0] + fex * boxhalfsize[2];
    if (min > rad || max < -rad) 
        return false;

    // AXISTEST_Z12
    p1 = e0[1] * v1[0] - e0[0] * v1[1];
    p2 = e0[1] * v2[0] - e0[0] * v2[1];

    if (p2 < p1) { min = p2; max = p1; 
    } else { 
        min = p1; max = p2; 
    }

    rad = fey * boxhalfsize[0] + fex * boxhalfsize[1];
    if (min > rad || max < -rad) 
        return false;

    fex = std::fabs(e1[0]);
    fey = std::fabs(e1[1]);
    fez = std::fabs(e1[2]);

    // AXISTEST_X01
    p0 = e1[2] * v0[1] - e1[1] * v0[2];
    p2 = e1[2] * v2[1] - e1[1] * v2[2];

    if (p0 < p2) { min = p0; max = p2; 
    } else { 
        min = p2; max = p0; 
    }

    rad = fez * boxhalfsize[1] + fey * boxhalfsize[2];
    if (min > rad || max < -rad) 
        return false;

    // AXISTEST_Y02
    p0 = -e1[2] * v0[0] + e1[0] * v0[2];
    p2 = -e1[2] * v2[0] + e1[0] * v2[2];

    if (p0 < p2) { min = p0; max = p2; 
    } else { 
        min = p2; max = p0; 
    }

    rad = fez * boxhalfsize[0] + fex * boxhalfsize[2];
    if (min > rad || max < -rad) 
        return false;

    // AXISTEST_Z0
    p0 = e1[1] * v0[0] - e1[0] * v0[1];
    p1 = e1[1] * v1[0] - e1[0] * v1[1];

    if (p0 < p1) { min = p0; max = p1; 
    } else { 
        min = p1; max = p0; 
    }

    rad = fey * boxhalfsize[0] + fex * boxhalfsize[1];
    if (min > rad || max < -rad) 
        return false;

    fex = std::fabs(e2[0]);
    fey = std::fabs(e2[1]);
    fez = std::fabs(e2[2]);

    // AXISTEST_X2
    p0 = e2[2] * v0[1] - e2[1] * v0[2];
    p1 = e2[2] * v1[1] - e2[1] * v1[2];

    if (p0 < p1) { min = p0; max = p1; 
    } else { 
        min = p1; max = p0; 
    }

    rad = fez * boxhalfsize[1] + fey * boxhalfsize[2];
    if (min > rad || max < -rad) 
        return false;

    // AXISTEST_Y1
    p0 = -e2[2] * v0[0] + e2[0] * v0[2];
    p1 = -e2[2] * v1[0] + e2[0] * v1[2];

    if (p0 < p1) { min = p0; max = p1; 
    } else { 
        min = p1; max = p0; 
    }

    rad = fez * boxhalfsize[0] + fex * boxhalfsize[2];
    if (min > rad || max < -rad) 
        return false;

    // AXISTEST_Z12
    p1 = e2[1] * v1[0] - e2[0] * v1[1];
    p2 = e2[1] * v2[0] - e2[0] * v2[1];

    if (p2 < p1) { min = p2; max = p1; 
    } else { 
        min = p1; max = p2; 
    }

    rad = fey * boxhalfsize[0] + fex * boxhalfsize[1];
    if (min > rad || max < -rad) 
        return false;

    // Test overlap 
    min = std::min(std::min(v0[0], v1[0]), v2[0]);
    max = std::max(std::max(v0[0], v1[0]), v2[0]);
    if (min > boxhalfsize[0] || max < -boxhalfsize[0]) 
        return false;

    min = std::min(std::min(v0[1], v1[1]), v2[1]);
    max = std::max(std::max(v0[1], v1[1]), v2[1]);
    if (min > boxhalfsize[1] || max < -boxhalfsize[1]) 
        return false;

    min = std::min(std::min(v0[2], v1[2]), v2[2]);
    max = std::max(std::max(v0[2], v1[2]), v2[2]);
    if (min > boxhalfsize[2] || max < -boxhalfsize[2]) 
        return false;

    // Test if the box intersects the plane of the triangle
    normal[0] = e0[1] * e1[2] - e0[2] * e1[1];
    normal[1] = e0[2] * e1[0] - e0[0] * e1[2];
    normal[2] = e0[0] * e1[1] - e0[1] * e1[0];

    // Plane box overlap
    float vmin[3], vmax[3];
    for (int q = 0; q < 3; q++) {
        float v = v0[q];
        if (normal[q] > 0.0f) {
            vmin[q] = -boxhalfsize[q] - v;
            vmax[q] =  boxhalfsize[q] - v;
        } else {
            vmin[q] =  boxhalfsize[q] - v;
            vmax[q] = -boxhalfsize[q] - v;
        }
    }

    float dotmin = normal[0] * vmin[0] + normal[1] * vmin[1] + normal[2] * vmin[2];
    float dotmax = normal[0] * vmax[0] + normal[1] * vmax[1] + normal[2] * vmax[2];

    if (dotmin > 0.0f) 
        return false;
    if (dotmax >= 0.0f) 
        return true;

    return false;
}

VoxelGrid::VoxelGrid() {
    params.gridMin = glm::vec3(0.0f);
    params.cellSize = 1.0f;
    params.gridDim = glm::ivec3(1);
    params.totalCells = 1;
}

VoxelGrid::~VoxelGrid() {
}

void VoxelGrid::build(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices,
    const TriangleHashGrid& triangleGrid,
    int gridSize) {
    // Calculate grid parameters from mesh bounding box
    if (positions.empty()) {
        std::cerr << "[VoxelGrid] Error: Empty mesh" << std::endl;
        return;
    }

    glm::vec3 meshMin(FLT_MAX);
    glm::vec3 meshMax(-FLT_MAX);
    for (const glm::vec3& position : positions) {
        meshMin = glm::min(meshMin, position);
        meshMax = glm::max(meshMax, position);
    }

    glm::vec3 extent = meshMax - meshMin;
    float maxExtent = std::max(std::max(extent.x, extent.y), extent.z);
    if (!std::isfinite(maxExtent) || maxExtent <= 0.0f) {
        std::cerr << "[VoxelGrid] Error: mesh bbox degenerate" << std::endl;
        return;
    }

    float scale = CANONICAL_SCALE / maxExtent;
    params.cellSize = scale;
    params.gridMin = meshMin - glm::vec3(CANONICAL_BIAS / scale);
    params.gridDim = glm::ivec3(gridSize);
    params.totalCells = static_cast<uint32_t>(gridSize * gridSize * gridSize);

    buildTriangleLists(positions, indices, triangleGrid);
}

void VoxelGrid::buildTriangleLists(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices,
    const TriangleHashGrid& triangleGrid) {
    trianglesList.clear();
    offsets.clear();

    int gridSize = params.gridDim.x;
    int numVoxels = params.gridDim.x * params.gridDim.y * params.gridDim.z;
    offsets.resize(numVoxels + 1, 0);

    float canonicalVoxelSize = CANONICAL_DOMAIN_SIZE / float(gridSize);

    std::vector<std::vector<int32_t>> voxelTriangles(static_cast<size_t>(numVoxels));
    std::vector<std::unordered_map<int, std::vector<int32_t>>> threadLocalVoxelTriangles;

    size_t triCount = indices.size() / 3;

    #pragma omp parallel
    {
        #pragma omp single
        {
            threadLocalVoxelTriangles.resize(static_cast<size_t>(omp_get_num_threads()));
        }

        const int threadId = omp_get_thread_num();
        auto& localVoxelTriangles = threadLocalVoxelTriangles[static_cast<size_t>(threadId)];

        #pragma omp for schedule(static)
        for (int t = 0; t < static_cast<int>(triCount); t++) {
            uint32_t i0 = indices[3 * t + 0];
            uint32_t i1 = indices[3 * t + 1];
            uint32_t i2 = indices[3 * t + 2];
            if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) {
                continue;
            }

            glm::vec3 w0 = positions[i0];
            glm::vec3 w1 = positions[i1];
            glm::vec3 w2 = positions[i2];

            glm::vec3 c0 = toCanonical(w0);
            glm::vec3 c1 = toCanonical(w1);
            glm::vec3 c2 = toCanonical(w2);

            glm::vec3 bbMin(FLT_MAX), bbMax(-FLT_MAX);
            bbMin = glm::min(bbMin, c0); bbMax = glm::max(bbMax, c0);
            bbMin = glm::min(bbMin, c1); bbMax = glm::max(bbMax, c1);
            bbMin = glm::min(bbMin, c2); bbMax = glm::max(bbMax, c2);

            int vx0 = clampInt(static_cast<int>(std::floor(bbMin.x / canonicalVoxelSize)), 0, gridSize - 1);
            int vy0 = clampInt(static_cast<int>(std::floor(bbMin.y / canonicalVoxelSize)), 0, gridSize - 1);
            int vz0 = clampInt(static_cast<int>(std::floor(bbMin.z / canonicalVoxelSize)), 0, gridSize - 1);
            int vx1 = clampInt(static_cast<int>(std::ceil (bbMax.x / canonicalVoxelSize)), 0, gridSize - 1);
            int vy1 = clampInt(static_cast<int>(std::ceil (bbMax.y / canonicalVoxelSize)), 0, gridSize - 1);
            int vz1 = clampInt(static_cast<int>(std::ceil (bbMax.z / canonicalVoxelSize)), 0, gridSize - 1);

            float triverts[3][3] = {
                { c0.x, c0.y, c0.z },
                { c1.x, c1.y, c1.z },
                { c2.x, c2.y, c2.z },
            };

            for (int z = vz0; z <= vz1; z++) {
                for (int y = vy0; y <= vy1; y++) {
                    for (int x = vx0; x <= vx1; x++) {
                        float boxcenter[3] = {
                            (x + 0.5f) * canonicalVoxelSize,
                            (y + 0.5f) * canonicalVoxelSize,
                            (z + 0.5f) * canonicalVoxelSize,
                        };
                        float boxhalfsize[3] = {
                            0.5f * canonicalVoxelSize,
                            0.5f * canonicalVoxelSize,
                            0.5f * canonicalVoxelSize,
                        };

                        if (!triBoxOverlap(boxcenter, boxhalfsize, triverts)) {
                            continue;
                        }

                        int voxelIdx = x + y * gridSize + z * gridSize * gridSize;
                        localVoxelTriangles[voxelIdx].push_back(static_cast<int32_t>(t));
                    }
                }
            }
        }
    }

    for (const auto& localVoxelTriangles : threadLocalVoxelTriangles) {
        for (const auto& [voxelIdx, triIds] : localVoxelTriangles) {
            auto& dst = voxelTriangles[static_cast<size_t>(voxelIdx)];
            dst.insert(dst.end(), triIds.begin(), triIds.end());
        }
    }

    // Build flat triangle list and offsets
    int currentOffset = 0;
    for (int i = 0; i < numVoxels; i++) {
        offsets[i] = currentOffset;
        auto& list = voxelTriangles[static_cast<size_t>(i)];
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
        for (int32_t triID : list) {
            trianglesList.push_back(triID);
        }
        currentOffset += static_cast<int>(list.size());
    }
    offsets[numVoxels] = currentOffset;

}


