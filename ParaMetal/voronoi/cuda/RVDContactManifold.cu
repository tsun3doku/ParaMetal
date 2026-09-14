#include "RVDContactManifold.cuh"
#include "RVDContactPredicates.cuh"
#include <cub/cub.cuh>
#include <cmath>
#include <algorithm>

namespace voronoi {

static __host__ __device__ inline uint64_t encodeBrickKey(int3 coord) {
    const uint64_t ux = uint64_t(coord.x + (1 << 20)) & 0x1FFFFFull;
    const uint64_t uy = uint64_t(coord.y + (1 << 20)) & 0x1FFFFFull;
    const uint64_t uz = uint64_t(coord.z + (1 << 20)) & 0x1FFFFFull;
    return (ux << 42) | (uy << 21) | uz;
}

static __host__ __device__ inline int3 decodeBrickKey(uint64_t key) {
    const int x = int((key >> 42) & 0x1FFFFFull) - (1 << 20);
    const int y = int((key >> 21) & 0x1FFFFFull) - (1 << 20);
    const int z = int(key & 0x1FFFFFull) - (1 << 20);
    return make_int3(x, y, z);
}

using cudaGeometry::DeviceTriangle;
using cudaGeometry::TriangleBvhNode;
using cudaGeometry::ClosestSurfaceHit;
using cudaGeometry::closestPoint;
using cudaGeometry::aabbDistanceSquared;
using cudaGeometry::InvalidNodeId;

static constexpr uint32_t BlockSize = 256;
static constexpr uint32_t CellsPerBrick = 512;

template <typename T>
static bool ensureCapacity(cudaUtils::CudaBuffer<T>& buffer, size_t needed) {
    if (buffer.size() < needed) {
        return buffer.allocate(needed);
    }
    return true;
}

template<typename T>
static bool cubExclusiveSum(cudaUtils::CudaBuffer<uint8_t>& scratch, const T* in, T* out, int count, cudaStream_t stream) {
    size_t bytes = 0;
    if (cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, count, stream) != cudaSuccess) return false;
    if (!ensureCapacity(scratch, bytes)) return false;
    return cub::DeviceScan::ExclusiveSum(scratch.get(), bytes, in, out, count, stream) == cudaSuccess;
}

template<typename T>
static bool cubSortKeys(cudaUtils::CudaBuffer<uint8_t>& scratch, const T* in, T* out, int count, cudaStream_t stream) {
    size_t bytes = 0;
    if (cub::DeviceRadixSort::SortKeys(nullptr, bytes, in, out, count, 0, sizeof(T) * 8, stream) != cudaSuccess) return false;
    if (!ensureCapacity(scratch, bytes)) return false;
    return cub::DeviceRadixSort::SortKeys(scratch.get(), bytes, in, out, count, 0, sizeof(T) * 8, stream) == cudaSuccess;
}

template<typename T, typename FlagT>
static bool cubSelectFlagged(cudaUtils::CudaBuffer<uint8_t>& scratch, const T* in, const FlagT* flags, T* out, uint32_t* outCount, int count, cudaStream_t stream) {
    size_t bytes = 0;
    if (cub::DeviceSelect::Flagged(nullptr, bytes, in, flags, out, outCount, count, stream) != cudaSuccess) return false;
    if (!ensureCapacity(scratch, bytes)) return false;
    return cub::DeviceSelect::Flagged(scratch.get(), bytes, in, flags, out, outCount, count, stream) == cudaSuccess;
}

template<typename T>
static bool cubSelectUnique(cudaUtils::CudaBuffer<uint8_t>& scratch, const T* in, T* out, uint32_t* outCount, int count, cudaStream_t stream) {
    size_t bytes = 0;
    if (cub::DeviceSelect::Unique(nullptr, bytes, in, out, outCount, count, stream) != cudaSuccess) return false;
    if (!ensureCapacity(scratch, bytes)) return false;
    return cub::DeviceSelect::Unique(scratch.get(), bytes, in, out, outCount, count, stream) == cudaSuccess;
}

static bool readLastSum(const cudaUtils::CudaBuffer<uint32_t>& off, const cudaUtils::CudaBuffer<uint32_t>& cnt, uint32_t count, cudaStream_t stream, uint32_t& result) {
    if (count == 0) {
        result = 0;
        return true;
    }
    uint32_t o = 0, c = 0;
    if (cudaMemcpyAsync(&o, off.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&c, cnt.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return false;
    result = o + c;
    return true;
}

static __device__ inline float3 min3(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

static __device__ inline float3 max3(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

// Kuhn 6-tetrahedron decomposition table
// 8 cube vertices: (x, y, z) in {0, 1}^3
// 0: (0,0,0), 1: (1,0,0), 2: (0,1,0), 3: (1,1,0)
// 4: (0,0,1), 5: (1,0,1), 6: (0,1,1), 7: (1,1,1)
static __device__ __constant__ int KuhnTetrahedra[6][4] = {
    {0, 1, 3, 7},
    {0, 1, 5, 7},
    {0, 2, 3, 7},
    {0, 2, 6, 7},
    {0, 4, 5, 7},
    {0, 4, 6, 7}
};

// Marching Tetrahedra triangles lookup: for each code in [0..15], lists edge indices (-1 terminated)
// 6 edges of tetrahedron (v0, v1, v2, v3):
// 0: (0,1), 1: (0,2), 2: (0,3), 3: (1,2), 4: (1,3), 5: (2,3)
static __device__ __constant__ int TetEdgePairs[6][2] = {
    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
};

static __device__ __constant__ int TetTriangles[16][7] = {
    {-1, -1, -1, -1, -1, -1, -1}, // 0000
    { 0,  2,  1, -1, -1, -1, -1}, // 0001 (v0 positive)
    { 0,  3,  4, -1, -1, -1, -1}, // 0010 (v1 positive)
    { 1,  2,  3,  2,  4,  3, -1}, // 0011 (v0, v1 positive)
    { 1,  5,  3, -1, -1, -1, -1}, // 0100 (v2 positive)
    { 0,  5,  3,  0,  2,  5, -1}, // 0101 (v0, v2 positive)
    { 0,  1,  5,  0,  5,  4, -1}, // 0110 (v1, v2 positive)
    { 2,  5,  4, -1, -1, -1, -1}, // 0111 (v0, v1, v2 positive)
    { 2,  4,  5, -1, -1, -1, -1}, // 1000 (v3 positive)
    { 0,  4,  5,  0,  5,  1, -1}, // 1001
    { 0,  2,  5,  0,  5,  3, -1}, // 1010
    { 1,  3,  5, -1, -1, -1, -1}, // 1011
    { 1,  4,  2,  1,  3,  4, -1}, // 1100
    { 0,  4,  3, -1, -1, -1, -1}, // 1101
    { 0,  1,  2, -1, -1, -1, -1}, // 1110
    {-1, -1, -1, -1, -1, -1, -1}  // 1111
};

static __device__ inline int fieldSign(float phi, uint64_t vertexKey) {
    constexpr float PhiTolerance = 1e-7f;
    if (phi >  PhiTolerance) return 1;
    if (phi < -PhiTolerance) return -1;
    return (vertexKey & 1ull) ? 1 : -1;
}

// 1D Bracketed Regula-Falsi + Illinois root refinement along edge (xA, xB)
static __device__ float3 refineEdgeZeroCrossing(
    float3 xA, float3 xB,
    float phiA, float phiB,
    const DeviceTriangle* triA, const TriangleBvhNode* nodesA, const uint32_t* parentsA, uint32_t rootA,
    const DeviceTriangle* triB, const TriangleBvhNode* nodesB, const uint32_t* parentsB, uint32_t rootB,
    float contactResolution)
{
    float tLo = 0.0f, tHi = 1.0f;
    float pLo = phiA, pHi = phiB;
    float t = 0.5f;
    const float edgeLen = sqrtf((xB.x - xA.x) * (xB.x - xA.x) + (xB.y - xA.y) * (xB.y - xA.y) + (xB.z - xA.z) * (xB.z - xA.z));
    const float termLen = 0.1f * contactResolution;

    for (int iter = 0; iter < 24; ++iter) {
        if ((tHi - tLo) * edgeLen <= termLen) break;
        const float denom = pHi - pLo;
        if (fabsf(denom) < 1e-12f) {
            t = 0.5f * (tLo + tHi);
        } else {
            t = tLo - pLo * (tHi - tLo) / denom;
            if (t <= tLo + 1e-4f || t >= tHi - 1e-4f) {
                t = 0.5f * (tLo + tHi);
            }
        }

        const float3 xMid = make_float3(
            xA.x + t * (xB.x - xA.x),
            xA.y + t * (xB.y - xA.y),
            xA.z + t * (xB.z - xA.z));

        const ClosestSurfaceHit hA = closestPoint(xMid, triA, nodesA, parentsA, rootA);
        const ClosestSurfaceHit hB = closestPoint(xMid, triB, nodesB, parentsB, rootB);
        const float dMidA = sqrtf(hA.closest.distanceSquared);
        const float dMidB = sqrtf(hB.closest.distanceSquared);
        const float pMid = dMidA - dMidB;

        if (pLo * pMid <= 0.0f) {
            tHi = t;
            pHi = pMid;
            pLo *= 0.5f; // Illinois dampening
        } else {
            tLo = t;
            pLo = pMid;
            pHi *= 0.5f;
        }
    }

    return make_float3(
        xA.x + t * (xB.x - xA.x),
        xA.y + t * (xB.y - xA.y),
        xA.z + t * (xB.z - xA.z));
}

// Canonical Kuhn edge encoding
static __device__ inline bool gridNodeLess(int3 a, int3 b) {
    if (a.z != b.z) return a.z < b.z;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

static __device__ inline int getKuhnEdgeType(int3 d) {
    if (d.x == 1 && d.y == 0 && d.z == 0) return 0; // Axis X
    if (d.x == 0 && d.y == 1 && d.z == 0) return 1; // Axis Y
    if (d.x == 0 && d.y == 0 && d.z == 1) return 2; // Axis Z
    if (d.x == 1 && d.y == 1 && d.z == 0) return 3; // Face XY
    if (d.x == 1 && d.y == 0 && d.z == 1) return 4; // Face XZ
    if (d.x == 0 && d.y == 1 && d.z == 1) return 5; // Face YZ
    return 6;                                       // Main Diag XYZ
}

static __device__ inline int3 getKuhnEdgeOffset(int edgeType) {
    switch (edgeType) {
        case 0: return make_int3(1, 0, 0);
        case 1: return make_int3(0, 1, 0);
        case 2: return make_int3(0, 0, 1);
        case 3: return make_int3(1, 1, 0);
        case 4: return make_int3(1, 0, 1);
        case 5: return make_int3(0, 1, 1);
        default: return make_int3(1, 1, 1);
    }
}

static __device__ inline uint64_t encodeEdgeKey(int3 gMin, int edgeType) {
    const uint64_t ux = uint64_t(gMin.x + (1 << 19)) & 0x0FFFFFull;
    const uint64_t uy = uint64_t(gMin.y + (1 << 19)) & 0x0FFFFFull;
    const uint64_t uz = uint64_t(gMin.z + (1 << 19)) & 0x0FFFFFull;
    const uint64_t nodeKey = (ux << 40) | (uy << 20) | uz;
    return (nodeKey << 3) | uint64_t(edgeType & 7);
}

static __device__ inline void decodeEdgeKey(uint64_t key, int3& outGMin, int& outEdgeType) {
    outEdgeType = int(key & 7);
    const uint64_t nodeKey = key >> 3;
    outGMin.x = int((nodeKey >> 40) & 0x0FFFFFull) - (1 << 19);
    outGMin.y = int((nodeKey >> 20) & 0x0FFFFFull) - (1 << 19);
    outGMin.z = int(nodeKey & 0x0FFFFFull) - (1 << 19);
}

static __device__ inline uint64_t getCanonicalKuhnEdgeKey(int3 gA, int3 gB) {
    int3 gMin = gA;
    int3 gMax = gB;
    if (gridNodeLess(gB, gA)) {
        gMin = gB;
        gMax = gA;
    }
    const int3 diff = make_int3(gMax.x - gMin.x, gMax.y - gMin.y, gMax.z - gMin.z);
    const int edgeType = getKuhnEdgeType(diff);
    return encodeEdgeKey(gMin, edgeType);
}

// -----------------------------------------------------------------------------
// Dual-Tree BVH Frontier Kernels
// -----------------------------------------------------------------------------
static __global__ void countBvhChildrenKernel(
    const uint2* currentPairs,
    uint32_t pairCount,
    const TriangleBvhNode* nodesA,
    const TriangleBvhNode* nodesB,
    float rangeSq,
    uint32_t* outChildCounts,
    uint8_t* outIsLeafPair)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairCount) return;

    const uint2 pair = currentPairs[idx];
    const TriangleBvhNode nA = nodesA[pair.x];
    const TriangleBvhNode nB = nodesB[pair.y];

    const float d2 = aabbDistanceSquared(nA.minimum, nA.maximum, nB.minimum, nB.maximum);
    if (d2 > rangeSq) {
        outChildCounts[idx] = 0;
        outIsLeafPair[idx] = 0;
        return;
    }

    if (nA.isLeaf() && nB.isLeaf()) {
        outChildCounts[idx] = 0;
        outIsLeafPair[idx] = 1;
    } else {
        outChildCounts[idx] = 2;
        outIsLeafPair[idx] = 0;
    }
}

static __global__ void emitBvhChildrenKernel(
    const uint2* currentPairs,
    uint32_t pairCount,
    const TriangleBvhNode* nodesA,
    const TriangleBvhNode* nodesB,
    const uint32_t* childCounts,
    const uint32_t* childOffsets,
    uint2* nextPairs)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairCount) return;

    if (childCounts[idx] == 0) return;

    const uint2 pair = currentPairs[idx];
    const TriangleBvhNode nA = nodesA[pair.x];
    const TriangleBvhNode nB = nodesB[pair.y];

    const uint32_t baseDst = childOffsets[idx];

    if (nA.isLeaf()) {
        nextPairs[baseDst + 0] = make_uint2(pair.x, nB.leftChild());
        nextPairs[baseDst + 1] = make_uint2(pair.x, nB.rightChild());
    } else if (nB.isLeaf()) {
        nextPairs[baseDst + 0] = make_uint2(nA.leftChild(), pair.y);
        nextPairs[baseDst + 1] = make_uint2(nA.rightChild(), pair.y);
    } else {
        const float3 extA = make_float3(nA.maximum.x - nA.minimum.x, nA.maximum.y - nA.minimum.y, nA.maximum.z - nA.minimum.z);
        const float3 extB = make_float3(nB.maximum.x - nB.minimum.x, nB.maximum.y - nB.minimum.y, nB.maximum.z - nB.minimum.z);
        const float vA = extA.x * extA.y * extA.z;
        const float vB = extB.x * extB.y * extB.z;

        if (vA >= vB) {
            nextPairs[baseDst + 0] = make_uint2(nA.leftChild(), pair.y);
            nextPairs[baseDst + 1] = make_uint2(nA.rightChild(), pair.y);
        } else {
            nextPairs[baseDst + 0] = make_uint2(pair.x, nB.leftChild());
            nextPairs[baseDst + 1] = make_uint2(pair.x, nB.rightChild());
        }
    }
}

static __global__ void countLeafBricksKernel(
    const uint2* leafPairs,
    uint32_t leafPairCount,
    const TriangleBvhNode* nodesA,
    const TriangleBvhNode* nodesB,
    float3 domainMin, float3 domainMax,
    float invBrickDim, float contactRange,
    uint32_t* outBrickCounts)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leafPairCount) return;

    const uint2 pair = leafPairs[idx];
    const TriangleBvhNode nA = nodesA[pair.x];
    const TriangleBvhNode nB = nodesB[pair.y];

    const float3 pairMin = min3(nA.minimum, nB.minimum);
    const float3 pairMax = max3(nA.maximum, nB.maximum);

    const float3 boxMin = max3(domainMin, make_float3(pairMin.x - contactRange, pairMin.y - contactRange, pairMin.z - contactRange));
    const float3 boxMax = min3(domainMax, make_float3(pairMax.x + contactRange, pairMax.y + contactRange, pairMax.z + contactRange));

    if (boxMin.x <= boxMax.x && boxMin.y <= boxMax.y && boxMin.z <= boxMax.z) {
        const int3 minCoord = make_int3(
            int(floorf((boxMin.x - domainMin.x) * invBrickDim)),
            int(floorf((boxMin.y - domainMin.y) * invBrickDim)),
            int(floorf((boxMin.z - domainMin.z) * invBrickDim)));
        const int3 maxCoord = make_int3(
            int(floorf((boxMax.x - domainMin.x) * invBrickDim)),
            int(floorf((boxMax.y - domainMin.y) * invBrickDim)),
            int(floorf((boxMax.z - domainMin.z) * invBrickDim)));

        const int nx = max(0, maxCoord.x - minCoord.x + 1);
        const int ny = max(0, maxCoord.y - minCoord.y + 1);
        const int nz = max(0, maxCoord.z - minCoord.z + 1);
        outBrickCounts[idx] = uint32_t(nx * ny * nz);
    } else {
        outBrickCounts[idx] = 0;
    }
}

static __global__ void emitLeafBricksKernel(
    const uint2* leafPairs,
    uint32_t leafPairCount,
    const TriangleBvhNode* nodesA,
    const TriangleBvhNode* nodesB,
    float3 domainMin, float3 domainMax,
    float invBrickDim, float contactRange,
    const uint32_t* brickOffsets,
    uint64_t* outBrickKeys)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= leafPairCount) return;

    const uint2 pair = leafPairs[idx];
    const TriangleBvhNode nA = nodesA[pair.x];
    const TriangleBvhNode nB = nodesB[pair.y];

    const float3 pairMin = min3(nA.minimum, nB.minimum);
    const float3 pairMax = max3(nA.maximum, nB.maximum);

    const float3 boxMin = max3(domainMin, make_float3(pairMin.x - contactRange, pairMin.y - contactRange, pairMin.z - contactRange));
    const float3 boxMax = min3(domainMax, make_float3(pairMax.x + contactRange, pairMax.y + contactRange, pairMax.z + contactRange));

    if (boxMin.x <= boxMax.x && boxMin.y <= boxMax.y && boxMin.z <= boxMax.z) {
        const int3 minCoord = make_int3(
            int(floorf((boxMin.x - domainMin.x) * invBrickDim)),
            int(floorf((boxMin.y - domainMin.y) * invBrickDim)),
            int(floorf((boxMin.z - domainMin.z) * invBrickDim)));
        const int3 maxCoord = make_int3(
            int(floorf((boxMax.x - domainMin.x) * invBrickDim)),
            int(floorf((boxMax.y - domainMin.y) * invBrickDim)),
            int(floorf((boxMax.z - domainMin.z) * invBrickDim)));

        uint32_t dst = brickOffsets[idx];
        for (int bz = minCoord.z; bz <= maxCoord.z; ++bz) {
            for (int by = minCoord.y; by <= maxCoord.y; ++by) {
                for (int bx = minCoord.x; bx <= maxCoord.x; ++bx) {
                    outBrickKeys[dst++] = encodeBrickKey(make_int3(bx, by, bz));
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Kuhn Triangle Extraction Helpers & Kernels
// -----------------------------------------------------------------------------
static __device__ bool evaluateCellTetCodes(
    int3 brickCoord,
    float3 brickMin,
    uint32_t tid,
    float h,
    float rCell,
    float contactRange,
    float3 domainMin,
    const DeviceTriangle* triA, const TriangleBvhNode* nodesA, const uint32_t* parentsA, uint32_t rootA,
    const DeviceTriangle* triB, const TriangleBvhNode* nodesB, const uint32_t* parentsB, uint32_t rootB,
    int3& outCellG,
    uint8_t outTetCodes[6])
{
    const int cx = int(tid % 8);
    const int cy = int((tid / 8) % 8);
    const int cz = int(tid / 64);

    outCellG = make_int3(
        brickCoord.x * 8 + cx,
        brickCoord.y * 8 + cy,
        brickCoord.z * 8 + cz);

    const float3 cellMin = make_float3(
        brickMin.x + float(cx) * h,
        brickMin.y + float(cy) * h,
        brickMin.z + float(cz) * h);

    // Evaluate 8 corners
    float3 corners[8];
    float phi[8];
    float g[8];

    for (int k = 0; k < 8; ++k) {
        const float ox = (k & 1) ? h : 0.0f;
        const float oy = (k & 2) ? h : 0.0f;
        const float oz = (k & 4) ? h : 0.0f;
        corners[k] = make_float3(cellMin.x + ox, cellMin.y + oy, cellMin.z + oz);

        const ClosestSurfaceHit hitA = closestPoint(corners[k], triA, nodesA, parentsA, rootA);
        const ClosestSurfaceHit hitB = closestPoint(corners[k], triB, nodesB, parentsB, rootB);

        const float dA = sqrtf(hitA.closest.distanceSquared);
        const float dB = sqrtf(hitB.closest.distanceSquared);
        phi[k] = dA - dB;
        g[k] = dA + dB;
    }

    float minPhi = phi[0], maxPhi = phi[0];
    float minG = g[0];
    for (int k = 1; k < 8; ++k) {
        minPhi = fminf(minPhi, phi[k]);
        maxPhi = fmaxf(maxPhi, phi[k]);
        minG = fminf(minG, g[k]);
    }

    if (minPhi > 2.0f * rCell || maxPhi < -2.0f * rCell || minG > contactRange + 2.0f * rCell) {
        return false;
    }

    for (int tetIdx = 0; tetIdx < 6; ++tetIdx) {
        const int* tetVerts = KuhnTetrahedra[tetIdx];
        int signs[4];
        for (int vi = 0; vi < 4; ++vi) {
            const int cornerIdx = tetVerts[vi];
            const float3 p = corners[cornerIdx];
            const uint64_t vKey =
                (uint64_t(int(floorf((p.x - domainMin.x) / h))) << 40) ^
                (uint64_t(int(floorf((p.y - domainMin.y) / h))) << 20) ^
                 uint64_t(int(floorf((p.z - domainMin.z) / h)));
            signs[vi] = fieldSign(phi[cornerIdx], vKey);
        }

        const int code = (signs[0] > 0 ? 1 : 0) |
                         (signs[1] > 0 ? 2 : 0) |
                         (signs[2] > 0 ? 4 : 0) |
                         (signs[3] > 0 ? 8 : 0);

        outTetCodes[tetIdx] = uint8_t(code);
    }

    return true;
}

static __global__ void countRawTrianglesKernel(
    const uint64_t* uniqueBrickKeys,
    uint32_t activeBrickCount,
    const DeviceTriangle* triA, const TriangleBvhNode* nodesA, const uint32_t* parentsA, uint32_t rootA,
    const DeviceTriangle* triB, const TriangleBvhNode* nodesB, const uint32_t* parentsB, uint32_t rootB,
    float3 domainMin,
    float fieldCellSize, float contactRange,
    uint32_t* outCellTriangleCounts)
{
    const uint32_t brickIdx = blockIdx.x;
    if (brickIdx >= activeBrickCount) return;

    const int3 brickCoord = decodeBrickKey(uniqueBrickKeys[brickIdx]);
    const float brickDim = 8.0f * fieldCellSize;
    const float3 brickMin = make_float3(
        domainMin.x + float(brickCoord.x) * brickDim,
        domainMin.y + float(brickCoord.y) * brickDim,
        domainMin.z + float(brickCoord.z) * brickDim);

    const float h = fieldCellSize;
    const float rCell = 0.5f * sqrtf(3.0f) * h;
    const uint32_t tid = threadIdx.x;
    const uint32_t cellGlobalIdx = brickIdx * CellsPerBrick + tid;

    int3 cellG;
    uint8_t tetCodes[6];
    if (!evaluateCellTetCodes(
            brickCoord, brickMin, tid, h, rCell, contactRange, domainMin,
            triA, nodesA, parentsA, rootA,
            triB, nodesB, parentsB, rootB,
            cellG, tetCodes)) {
        outCellTriangleCounts[cellGlobalIdx] = 0;
        return;
    }

    uint32_t triCount = 0;
    for (int tetIdx = 0; tetIdx < 6; ++tetIdx) {
        const int code = tetCodes[tetIdx];
        if (code == 0 || code == 15) continue;

        const int* edges = TetTriangles[code];
        for (int e = 0; edges[e] != -1; e += 3) {
            ++triCount;
        }
    }

    outCellTriangleCounts[cellGlobalIdx] = triCount;
}

static __global__ void emitRawTrianglesKernel(
    const uint64_t* uniqueBrickKeys,
    uint32_t activeBrickCount,
    const DeviceTriangle* triA, const TriangleBvhNode* nodesA, const uint32_t* parentsA, uint32_t rootA,
    const DeviceTriangle* triB, const TriangleBvhNode* nodesB, const uint32_t* parentsB, uint32_t rootB,
    float3 domainMin,
    float fieldCellSize, float contactRange,
    const uint32_t* cellTriangleOffsets,
    uint64_t* outRawEdgeKeys)
{
    const uint32_t brickIdx = blockIdx.x;
    if (brickIdx >= activeBrickCount) return;

    const int3 brickCoord = decodeBrickKey(uniqueBrickKeys[brickIdx]);
    const float brickDim = 8.0f * fieldCellSize;
    const float3 brickMin = make_float3(
        domainMin.x + float(brickCoord.x) * brickDim,
        domainMin.y + float(brickCoord.y) * brickDim,
        domainMin.z + float(brickCoord.z) * brickDim);

    const float h = fieldCellSize;
    const float rCell = 0.5f * sqrtf(3.0f) * h;
    const uint32_t tid = threadIdx.x;
    const uint32_t cellGlobalIdx = brickIdx * CellsPerBrick + tid;

    int3 cellG;
    uint8_t tetCodes[6];
    if (!evaluateCellTetCodes(
            brickCoord, brickMin, tid, h, rCell, contactRange, domainMin,
            triA, nodesA, parentsA, rootA,
            triB, nodesB, parentsB, rootB,
            cellG, tetCodes)) {
        return;
    }

    uint32_t curTriOffset = cellTriangleOffsets[cellGlobalIdx];

    for (int tetIdx = 0; tetIdx < 6; ++tetIdx) {
        const int code = tetCodes[tetIdx];
        if (code == 0 || code == 15) continue;

        const int* tetVerts = KuhnTetrahedra[tetIdx];
        const int* edges = TetTriangles[code];
        for (int e = 0; edges[e] != -1; e += 3) {
            uint64_t triEdgeKeys[3];

            for (int corner = 0; corner < 3; ++corner) {
                const int edgeIdx = edges[e + corner];
                const int vA = tetVerts[TetEdgePairs[edgeIdx][0]];
                const int vB = tetVerts[TetEdgePairs[edgeIdx][1]];

                const int3 gA = make_int3(
                    cellG.x + (vA & 1),
                    cellG.y + ((vA >> 1) & 1),
                    cellG.z + ((vA >> 2) & 1));
                const int3 gB = make_int3(
                    cellG.x + (vB & 1),
                    cellG.y + ((vB >> 1) & 1),
                    cellG.z + ((vB >> 2) & 1));

                triEdgeKeys[corner] = getCanonicalKuhnEdgeKey(gA, gB);
            }

            const uint32_t baseIdx = curTriOffset * 3;
            outRawEdgeKeys[baseIdx + 0] = triEdgeKeys[0];
            outRawEdgeKeys[baseIdx + 1] = triEdgeKeys[1];
            outRawEdgeKeys[baseIdx + 2] = triEdgeKeys[2];
            ++curTriOffset;
        }
    }
}

// Generate shared manifold vertices by refining zero-crossings on unique edges
static __global__ void generateVerticesKernel(
    const uint64_t* uniqueEdgeKeys,
    uint32_t uniqueEdgeCount,
    const DeviceTriangle* triA, const TriangleBvhNode* nodesA, const uint32_t* parentsA, uint32_t rootA,
    const DeviceTriangle* triB, const TriangleBvhNode* nodesB, const uint32_t* parentsB, uint32_t rootB,
    float3 domainMin,
    float fieldCellSize,
    float contactResolution,
    float3* outVertices)
{
    const uint32_t vIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vIdx >= uniqueEdgeCount) return;

    int3 gMin;
    int edgeType;
    decodeEdgeKey(uniqueEdgeKeys[vIdx], gMin, edgeType);

    const float h = fieldCellSize;
    const float3 xA = make_float3(
        domainMin.x + float(gMin.x) * h,
        domainMin.y + float(gMin.y) * h,
        domainMin.z + float(gMin.z) * h);

    const int3 offset = getKuhnEdgeOffset(edgeType);
    const float3 xB = make_float3(
        xA.x + float(offset.x) * h,
        xA.y + float(offset.y) * h,
        xA.z + float(offset.z) * h);

    const ClosestSurfaceHit hA1 = closestPoint(xA, triA, nodesA, parentsA, rootA);
    const ClosestSurfaceHit hB1 = closestPoint(xA, triB, nodesB, parentsB, rootB);
    const float phiA = sqrtf(hA1.closest.distanceSquared) - sqrtf(hB1.closest.distanceSquared);

    const ClosestSurfaceHit hA2 = closestPoint(xB, triA, nodesA, parentsA, rootA);
    const ClosestSurfaceHit hB2 = closestPoint(xB, triB, nodesB, parentsB, rootB);
    const float phiB = sqrtf(hA2.closest.distanceSquared) - sqrtf(hB2.closest.distanceSquared);

    const float3 xFinal = refineEdgeZeroCrossing(
        xA, xB, phiA, phiB,
        triA, nodesA, parentsA, rootA, triB, nodesB, parentsB, rootB,
        contactResolution);

    outVertices[vIdx] = xFinal;
}

static __device__ inline int binarySearchEdgeKey(uint64_t key, const uint64_t* keys, int count) {
    int low = 0, high = count - 1;
    while (low <= high) {
        const int mid = low + ((high - low) >> 1);
        const uint64_t midVal = keys[mid];
        if (midVal == key) return mid;
        if (midVal < key) low = mid + 1;
        else high = mid - 1;
    }
    return -1;
}

// Assemble indexed triangles, orient winding from A to B, mark valid flags
static __global__ void indexTrianglesKernel(
    const uint64_t* rawTriangleEdgeKeys,
    uint32_t rawTriangleCount,
    const uint64_t* uniqueEdgeKeys,
    uint32_t uniqueEdgeCount,
    const float3* vertices,
    const DeviceTriangle* triA, const TriangleBvhNode* nodesA, const uint32_t* parentsA, uint32_t rootA,
    const DeviceTriangle* triB, const TriangleBvhNode* nodesB, const uint32_t* parentsB, uint32_t rootB,
    float contactRange,
    float tolerance,
    uint3* outCandidates,
    uint8_t* outValidFlags)
{
    const uint32_t triIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (triIdx >= rawTriangleCount) return;

    outValidFlags[triIdx] = 0;

    const uint32_t baseIdx = triIdx * 3;
    const uint64_t k0 = rawTriangleEdgeKeys[baseIdx + 0];
    const uint64_t k1 = rawTriangleEdgeKeys[baseIdx + 1];
    const uint64_t k2 = rawTriangleEdgeKeys[baseIdx + 2];

    const int idx0 = binarySearchEdgeKey(k0, uniqueEdgeKeys, int(uniqueEdgeCount));
    const int idx1 = binarySearchEdgeKey(k1, uniqueEdgeKeys, int(uniqueEdgeCount));
    const int idx2 = binarySearchEdgeKey(k2, uniqueEdgeKeys, int(uniqueEdgeCount));

    if (idx0 < 0 || idx1 < 0 || idx2 < 0) return;

    uint32_t v0 = uint32_t(idx0);
    uint32_t v1 = uint32_t(idx1);
    uint32_t v2 = uint32_t(idx2);

    if (v0 == v1 || v1 == v2 || v2 == v0) return;

    const float3 p0 = vertices[v0];
    const float3 p1 = vertices[v1];
    const float3 p2 = vertices[v2];

    const float3 cPos = make_float3(
        (p0.x + p1.x + p2.x) * (1.0f / 3.0f),
        (p0.y + p1.y + p2.y) * (1.0f / 3.0f),
        (p0.z + p1.z + p2.z) * (1.0f / 3.0f));

    const ClosestSurfaceHit hitA = closestPoint(cPos, triA, nodesA, parentsA, rootA);
    const ClosestSurfaceHit hitB = closestPoint(cPos, triB, nodesB, parentsB, rootB);

    const float dA = sqrtf(hitA.closest.distanceSquared);
    const float dB = sqrtf(hitB.closest.distanceSquared);
    const float g = dA + dB;

    if (!contactActive(g, contactRange, tolerance)) return;

    const float3 connector = make_float3(
        hitB.closest.point.x - hitA.closest.point.x,
        hitB.closest.point.y - hitA.closest.point.y,
        hitB.closest.point.z - hitA.closest.point.z);

    // Orient winding consistently from model A towards model B
    const float3 e1 = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    const float3 e2 = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
    const float3 n = make_float3(
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x);

    const float dotNorm = n.x * connector.x + n.y * connector.y + n.z * connector.z;
    if (dotNorm < 0.0f) {
        const uint32_t tmp = v1;
        v1 = v2;
        v2 = tmp;
    }

    outCandidates[triIdx] = make_uint3(v0, v1, v2);
    outValidFlags[triIdx] = 1;
}

bool extractContactManifold(
    cudaGeometry::DeviceModelGeometryView geomA,
    cudaGeometry::DeviceModelGeometryView geomB,
    float3 domainMin,
    float3 domainMax,
    float contactRange,
    float contactResolution,
    ContactManifoldMesh& outMesh,
    cudaStream_t stream)
{
    outMesh.reset();

    if (geomA.root == InvalidNodeId || geomB.root == InvalidNodeId)
        return true; // legitimately empty BVH

    if (!geomA.triangles || !geomA.nodes || !geomA.parents ||
        !geomB.triangles || !geomB.nodes || !geomB.parents ||
        !std::isfinite(contactRange) ||
        !std::isfinite(contactResolution) ||
        !(contactRange > 0.0f) ||
        !(contactResolution > 0.0f)) {
        return false;
    }

    const float3 domainSize = make_float3(
        domainMax.x - domainMin.x,
        domainMax.y - domainMin.y,
        domainMax.z - domainMin.z);

    if (!std::isfinite(domainSize.x) ||
        !std::isfinite(domainSize.y) ||
        !std::isfinite(domainSize.z) ||
        !(domainSize.x > 0.0f) ||
        !(domainSize.y > 0.0f) ||
        !(domainSize.z > 0.0f)) {
        return false;
    }

    const float fieldCellSize = contactResolution;
    const float brickDim = 8.0f * fieldCellSize;
    const float invBrickDim = 1.0f / brickDim;
    const float rangeSq = contactRange * contactRange;
    const float coordinateScale = fmaxf(domainSize.x, fmaxf(domainSize.y, domainSize.z));
    const float lengthTol = contactGeometryTolerance(contactResolution, coordinateScale);

    // -------------------------------------------------------------------------
    // Step 1: Dynamic Dual-Tree BVH Frontier to collect leaf-leaf pairs
    // -------------------------------------------------------------------------
    cudaUtils::CudaBuffer<uint2> currentPairs;
    cudaUtils::CudaBuffer<uint2> nextPairs;
    if (!currentPairs.allocate(1)) return false;
    const uint2 initialPair = make_uint2(geomA.root, geomB.root);
    if (!currentPairs.uploadAsync(&initialPair, 1, stream)) return false;

    cudaUtils::CudaBuffer<uint32_t> devChildCounts;
    cudaUtils::CudaBuffer<uint8_t> devIsLeafPair;
    cudaUtils::CudaBuffer<uint2> devLeafBatch;
    cudaUtils::CudaBuffer<uint32_t> devLeafCount;
    if (!devLeafCount.allocate(1)) return false;
    cudaUtils::CudaBuffer<uint32_t> devChildOffsets;

    cudaUtils::CudaBuffer<uint2> devLeafPairs;
    uint32_t totalLeafPairs = 0;
    uint32_t currentPairCount = 1;

    cudaUtils::CudaBuffer<uint8_t> cubScratch;

    while (currentPairCount > 0) {
        const uint32_t pairCount = currentPairCount;
        const uint32_t grid = (pairCount + BlockSize - 1) / BlockSize;

        if (!ensureCapacity(devChildCounts, pairCount) ||
            !ensureCapacity(devIsLeafPair, pairCount) ||
            !ensureCapacity(devLeafBatch, pairCount) ||
            !ensureCapacity(devChildOffsets, pairCount)) {
            return false;
        }

        countBvhChildrenKernel<<<grid, BlockSize, 0, stream>>>(
            currentPairs.get(), pairCount,
            geomA.nodes, geomB.nodes,
            rangeSq,
            devChildCounts.get(), devIsLeafPair.get());
        if (cudaGetLastError() != cudaSuccess) return false;

        // Extract leaf pairs in current frontier
        if (!cubSelectFlagged(cubScratch, currentPairs.get(), devIsLeafPair.get(), devLeafBatch.get(), devLeafCount.get(), int(pairCount), stream)) return false;

        // Scan child counts to produce nextPairs
        if (!cubExclusiveSum(cubScratch, devChildCounts.get(), devChildOffsets.get(), int(pairCount), stream)) return false;

        uint32_t leafBatchCount = 0;
        uint32_t lastOffset = 0;
        uint32_t lastCount = 0;

        if (cudaMemcpyAsync(&leafBatchCount, devLeafCount.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaMemcpyAsync(&lastOffset, devChildOffsets.get() + (pairCount - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaMemcpyAsync(&lastCount, devChildCounts.get() + (pairCount - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            return false;
        }

        if (leafBatchCount > 0) {
            if (devLeafPairs.size() < totalLeafPairs + leafBatchCount) {
                const size_t newCap = std::max(size_t(totalLeafPairs + leafBatchCount), devLeafPairs.size() * 2);
                cudaUtils::CudaBuffer<uint2> grownBuffer;
                if (!grownBuffer.allocate(newCap)) return false;
                if (totalLeafPairs > 0) {
                    if (cudaMemcpyAsync(grownBuffer.get(), devLeafPairs.get(), totalLeafPairs * sizeof(uint2), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                        return false;
                    }
                }
                devLeafPairs.swap(grownBuffer);
            }
            if (cudaMemcpyAsync(devLeafPairs.get() + totalLeafPairs, devLeafBatch.get(), leafBatchCount * sizeof(uint2), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                return false;
            }
            totalLeafPairs += leafBatchCount;
        }

        const uint32_t nextPairCount = lastOffset + lastCount;
        if (nextPairCount > 0) {
            if (!ensureCapacity(nextPairs, nextPairCount)) return false;

            emitBvhChildrenKernel<<<grid, BlockSize, 0, stream>>>(
                currentPairs.get(), pairCount,
                geomA.nodes, geomB.nodes,
                devChildCounts.get(),
                devChildOffsets.get(),
                nextPairs.get());
            if (cudaGetLastError() != cudaSuccess) return false;
        }

        currentPairs.swap(nextPairs);
        currentPairCount = nextPairCount;
    }

    if (totalLeafPairs == 0) return true;

    // -------------------------------------------------------------------------
    // Step 2: From leaf pairs, count touched bricks -> CUB scan -> emit brick keys
    // -------------------------------------------------------------------------
    const uint32_t leafPairCount = totalLeafPairs;

    const uint32_t gridLeaf = (leafPairCount + BlockSize - 1) / BlockSize;

    cudaUtils::CudaBuffer<uint32_t> devLeafBrickCounts;
    cudaUtils::CudaBuffer<uint32_t> devLeafBrickOffsets;
    if (!devLeafBrickCounts.allocate(leafPairCount) || !devLeafBrickOffsets.allocate(leafPairCount)) return false;

    countLeafBricksKernel<<<gridLeaf, BlockSize, 0, stream>>>(
        devLeafPairs.get(), leafPairCount,
        geomA.nodes, geomB.nodes,
        domainMin, domainMax,
        invBrickDim, contactRange,
        devLeafBrickCounts.get());
    if (cudaGetLastError() != cudaSuccess) return false;

    if (!cubExclusiveSum(cubScratch, devLeafBrickCounts.get(), devLeafBrickOffsets.get(), int(leafPairCount), stream)) return false;
    uint32_t totalRawBricks = 0;
    if (!readLastSum(devLeafBrickOffsets, devLeafBrickCounts, leafPairCount, stream, totalRawBricks)) return false;
    if (totalRawBricks == 0) return true;

    cudaUtils::CudaBuffer<uint64_t> devRawBrickKeysIn;
    cudaUtils::CudaBuffer<uint64_t> devRawBrickKeysOut;
    if (!devRawBrickKeysIn.allocate(totalRawBricks) || !devRawBrickKeysOut.allocate(totalRawBricks)) return false;

    emitLeafBricksKernel<<<gridLeaf, BlockSize, 0, stream>>>(
        devLeafPairs.get(), leafPairCount,
        geomA.nodes, geomB.nodes,
        domainMin, domainMax,
        invBrickDim, contactRange,
        devLeafBrickOffsets.get(),
        devRawBrickKeysIn.get());
    if (cudaGetLastError() != cudaSuccess) return false;

    // Radix sort + unique brick keys
    if (!cubSortKeys(cubScratch, devRawBrickKeysIn.get(), devRawBrickKeysOut.get(), int(totalRawBricks), stream)) return false;

    cudaUtils::CudaBuffer<uint32_t> devUniqueBrickCount;
    if (!devUniqueBrickCount.allocate(1) ||
        !cubSelectUnique(cubScratch, devRawBrickKeysOut.get(), devRawBrickKeysIn.get(), devUniqueBrickCount.get(), int(totalRawBricks), stream)) return false;

    uint32_t uniqueBrickCount = 0;
    if (cudaMemcpyAsync(&uniqueBrickCount, devUniqueBrickCount.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return false;

    if (uniqueBrickCount == 0) return true;

    // -------------------------------------------------------------------------
    // Step 3: Count Kuhn triangles per cell -> CUB scan -> exact emit edge keys
    // -------------------------------------------------------------------------
    const uint32_t totalFieldCells = uniqueBrickCount * CellsPerBrick;
    cudaUtils::CudaBuffer<uint32_t> devCellTriCounts;
    cudaUtils::CudaBuffer<uint32_t> devCellTriOffsets;
    if (!devCellTriCounts.allocate(totalFieldCells) || !devCellTriOffsets.allocate(totalFieldCells)) return false;

    countRawTrianglesKernel<<<uniqueBrickCount, CellsPerBrick, 0, stream>>>(
        devRawBrickKeysIn.get(), uniqueBrickCount,
        geomA.triangles, geomA.nodes, geomA.parents, geomA.root,
        geomB.triangles, geomB.nodes, geomB.parents, geomB.root,
        domainMin,
        fieldCellSize, contactRange,
        devCellTriCounts.get());
    if (cudaGetLastError() != cudaSuccess) return false;

    if (!cubExclusiveSum(cubScratch, devCellTriCounts.get(), devCellTriOffsets.get(), int(totalFieldCells), stream)) return false;
    uint32_t rawTriangles = 0;
    if (!readLastSum(devCellTriOffsets, devCellTriCounts, totalFieldCells, stream, rawTriangles)) return false;
    if (rawTriangles == 0) return true;

    const uint32_t totalEdges = rawTriangles * 3;
    cudaUtils::CudaBuffer<uint64_t> devRawEdgeKeysIn;
    cudaUtils::CudaBuffer<uint64_t> devRawEdgeKeysOut;
    if (!devRawEdgeKeysIn.allocate(totalEdges) || !devRawEdgeKeysOut.allocate(totalEdges)) return false;

    emitRawTrianglesKernel<<<uniqueBrickCount, CellsPerBrick, 0, stream>>>(
        devRawBrickKeysIn.get(), uniqueBrickCount,
        geomA.triangles, geomA.nodes, geomA.parents, geomA.root,
        geomB.triangles, geomB.nodes, geomB.parents, geomB.root,
        domainMin,
        fieldCellSize, contactRange,
        devCellTriOffsets.get(),
        devRawEdgeKeysIn.get());
    if (cudaGetLastError() != cudaSuccess) return false;

    // Radix sort + Unique edge keys
    if (!cubSortKeys(cubScratch, devRawEdgeKeysIn.get(), devRawEdgeKeysOut.get(), int(totalEdges), stream)) return false;

    cudaUtils::CudaBuffer<uint64_t> devUniqueEdgeKeys;
    cudaUtils::CudaBuffer<uint32_t> devUniqueEdgeCount;
    if (!devUniqueEdgeKeys.allocate(totalEdges) || !devUniqueEdgeCount.allocate(1) ||
        !cubSelectUnique(cubScratch, devRawEdgeKeysOut.get(), devUniqueEdgeKeys.get(), devUniqueEdgeCount.get(), int(totalEdges), stream)) return false;

    uint32_t uniqueEdgeCount = 0;
    if (cudaMemcpyAsync(&uniqueEdgeCount, devUniqueEdgeCount.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return false;

    if (uniqueEdgeCount == 0) return true;

    // -------------------------------------------------------------------------
    // Step 4: Refine zero-crossing vertices on unique edges
    // -------------------------------------------------------------------------
    if (!outMesh.vertices.allocate(uniqueEdgeCount)) return false;

    const uint32_t gridV = (uniqueEdgeCount + BlockSize - 1) / BlockSize;
    generateVerticesKernel<<<gridV, BlockSize, 0, stream>>>(
        devUniqueEdgeKeys.get(), uniqueEdgeCount,
        geomA.triangles, geomA.nodes, geomA.parents, geomA.root,
        geomB.triangles, geomB.nodes, geomB.parents, geomB.root,
        domainMin, fieldCellSize, contactResolution,
        outMesh.vertices.get());
    if (cudaGetLastError() != cudaSuccess) return false;

    // -------------------------------------------------------------------------
    // Step 5: Index triangles with Flagged compaction
    // -------------------------------------------------------------------------
    cudaUtils::CudaBuffer<uint3> devCandidates;
    cudaUtils::CudaBuffer<uint8_t> devValidFlags;
    if (!devCandidates.allocate(rawTriangles) || !devValidFlags.allocate(rawTriangles)) return false;

    const uint32_t gridT = (rawTriangles + BlockSize - 1) / BlockSize;
    indexTrianglesKernel<<<gridT, BlockSize, 0, stream>>>(
        devRawEdgeKeysIn.get(), rawTriangles,
        devUniqueEdgeKeys.get(), uniqueEdgeCount,
        outMesh.vertices.get(),
        geomA.triangles, geomA.nodes, geomA.parents, geomA.root,
        geomB.triangles, geomB.nodes, geomB.parents, geomB.root,
        contactRange, lengthTol,
        devCandidates.get(),
        devValidFlags.get());
    if (cudaGetLastError() != cudaSuccess) return false;

    if (!outMesh.triangles.allocate(rawTriangles)) return false;

    cudaUtils::CudaBuffer<uint32_t> devValidTriCount;
    if (!devValidTriCount.allocate(1) ||
        !cubSelectFlagged(cubScratch, devCandidates.get(), devValidFlags.get(), outMesh.triangles.get(), devValidTriCount.get(), int(rawTriangles), stream)) return false;

    uint32_t validTriangles = 0;
    if (cudaMemcpyAsync(&validTriangles, devValidTriCount.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return false;

    if (validTriangles < rawTriangles) {
        cudaUtils::CudaBuffer<uint3> compacted;
        if (validTriangles > 0) {
            if (!compacted.allocate(validTriangles)) return false;
            if (cudaMemcpyAsync(compacted.get(), outMesh.triangles.get(), validTriangles * sizeof(uint3), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                return false;
            }
        }
        outMesh.triangles.swap(compacted);
    }

    return true;
}

} // namespace voronoi
