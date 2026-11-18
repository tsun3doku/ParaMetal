#include "SupportingHalfedge.hpp"
#include <glm/gtc/constants.hpp>
#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_set>

SupportingHalfedge::SupportingHalfedge(const SignpostMesh& inputMesh, SignpostMesh& intrinsicMesh)
    : inputMesh(inputMesh)
    , intrinsicMesh(intrinsicMesh)
{
}

SupportingHalfedge::~SupportingHalfedge() = default;

void SupportingHalfedge::initialize() {
    const auto& inputConn = inputMesh.getConnectivity();
    const auto& inputFaces = inputConn.getFaces();
    const auto& inputHalfedges = inputConn.getHalfEdges();

    const auto& intrinsicConn = intrinsicMesh.getConnectivity();
    const auto& intrinsicFaces = intrinsicConn.getFaces();
    const auto& intrinsicHalfedges = intrinsicConn.getHalfEdges();

    supportingInfoPerHalfedge.resize(inputHalfedges.size());
    for (auto& info : supportingInfoPerHalfedge) {
        info.supportingHE = INVALID_INDEX;
        info.supportingAngle = 0.0;
    }

    for (size_t faceIdx = 0; faceIdx < inputFaces.size(); ++faceIdx) {
        if (faceIdx >= intrinsicFaces.size()) {
            continue;
        }

        uint32_t inputHe = inputFaces[faceIdx].halfEdgeIdx;
        uint32_t intrinsicHe = intrinsicFaces[faceIdx].halfEdgeIdx;
        for (int i = 0; i < 3; ++i) {
            if (inputHe == INVALID_INDEX || inputHe >= inputHalfedges.size() ||
                intrinsicHe == INVALID_INDEX || intrinsicHe >= intrinsicHalfedges.size()) {
                break;
            }

            supportingInfoPerHalfedge[inputHe].supportingHE = intrinsicHe;
            supportingInfoPerHalfedge[inputHe].supportingAngle = 0.0;

            inputHe = inputHalfedges[inputHe].next;
            intrinsicHe = intrinsicHalfedges[intrinsicHe].next;
        }
    }

}

double SupportingHalfedge::cornerAngleAtHalfedge(uint32_t he) const {
    const auto& intrinsicConn = intrinsicMesh.getConnectivity();
    const auto& intrinsicHalfedges = intrinsicConn.getHalfEdges();
    const auto& intrinsicFaces = intrinsicConn.getFaces();

    if (he == INVALID_INDEX || he >= intrinsicHalfedges.size()) {
        return 0.0;
    }

    const uint32_t faceIdx = intrinsicHalfedges[he].face;
    if (faceIdx == INVALID_INDEX || faceIdx >= intrinsicFaces.size()) {
        return 0.0;
    }

    const uint32_t he0 = intrinsicFaces[faceIdx].halfEdgeIdx;
    if (he0 == INVALID_INDEX || he0 >= intrinsicHalfedges.size()) {
        return 0.0;
    }

    const uint32_t he1 = intrinsicHalfedges[he0].next;
    if (he1 == INVALID_INDEX || he1 >= intrinsicHalfedges.size()) {
        return 0.0;
    }

    const uint32_t he2 = intrinsicHalfedges[he1].next;
    if (he2 == INVALID_INDEX || he2 >= intrinsicHalfedges.size()) {
        return 0.0;
    }

    const auto angles = intrinsicMesh.computeCornerAngles(faceIdx);
    if (he == he0) {
        return angles[0];
    }
    if (he == he1) {
        return angles[1];
    }
    if (he == he2) {
        return angles[2];
    }

    return 0.0;
}

void SupportingHalfedge::prepareRemoval(uint32_t intrinsicHE, RemovalPlan& plan) const {
    const auto& intrinsicConn = intrinsicMesh.getConnectivity();
    const auto& intrinsicHalfedges = intrinsicConn.getHalfEdges();

    if (intrinsicHE >= intrinsicHalfedges.size()) {
        return;
    }

    uint32_t v = intrinsicHalfedges[intrinsicHE].origin;

    const auto& inputConn = inputMesh.getConnectivity();
    const auto& inputVertices = inputConn.getVertices();
    const auto& inputHalfedges = inputConn.getHalfEdges();

    if (v >= inputVertices.size()) {
        return;
    }

    std::vector<uint32_t> helist = inputConn.getVertexHalfEdges(v);

    auto effectiveInfo = [&](uint32_t idx) -> SupportingInfo {
        auto it = plan.changes.find(idx);
        if (it != plan.changes.end()) {
            return it->second;
        }
        return supportingInfoPerHalfedge[idx];
    };

    for (uint32_t he : helist) {
        if (he >= inputHalfedges.size() || he >= supportingInfoPerHalfedge.size()) {
            continue;
        }

        SupportingInfo info = effectiveInfo(he);
        if (info.supportingHE != intrinsicHE) {
            continue;
        }

        uint32_t mate_h = intrinsicHalfedges[intrinsicHE].opposite;
        if (mate_h == INVALID_INDEX || mate_h >= intrinsicHalfedges.size()) {
            continue;
        }
        uint32_t new_ref = intrinsicHalfedges[mate_h].next;
        if (new_ref == INVALID_INDEX || new_ref >= intrinsicHalfedges.size()) {
            continue;
        }

        info.supportingHE = new_ref;

        double theta = cornerAngleAtHalfedge(new_ref);
        info.supportingAngle -= theta;
        clampSupportingAngle(info);

        plan.changes[he] = info;
    }
}

void SupportingHalfedge::commitRemoval(const RemovalPlan& plan) {
    for (const auto& entry : plan.changes) {
        if (entry.first < supportingInfoPerHalfedge.size()) {
            supportingInfoPerHalfedge[entry.first] = entry.second;
        }
    }
}

bool SupportingHalfedge::planReferencesInvalidated(const RemovalPlan& plan,
                                                   const std::unordered_set<uint32_t>& invalidated) const {
    for (uint32_t idx = 0; idx < supportingInfoPerHalfedge.size(); ++idx) {
        const SupportingInfo& info = [&]() -> const SupportingInfo& {
            auto it = plan.changes.find(idx);
            if (it != plan.changes.end()) {
                return it->second;
            }
            return supportingInfoPerHalfedge[idx];
        }();

        if (info.supportingHE != INVALID_INDEX && invalidated.count(info.supportingHE) != 0) {
            return true;
        }
    }
    return false;
}

void SupportingHalfedge::updateRemoval(uint32_t intrinsicHE) {
    RemovalPlan plan;
    prepareRemoval(intrinsicHE, plan);
    commitRemoval(plan);
}

void SupportingHalfedge::updateInsertion(uint32_t h) {
    auto& intrinsicConn = intrinsicMesh.getConnectivity();
    const auto& intrinsicHalfedges = intrinsicConn.getHalfEdges();

    if (h >= intrinsicHalfedges.size()) {
        return;
    }

    uint32_t v = intrinsicHalfedges[h].origin;

    const auto& inputConn = inputMesh.getConnectivity();
    const auto& inputHalfedges = inputConn.getHalfEdges();
    const auto& inputVertices = inputConn.getVertices();

    if (v >= inputVertices.size()) {
        return;
    }

    std::vector<uint32_t> helist = inputConn.getVertexHalfEdges(v);

    for (uint32_t he : helist) {
        if (he >= inputHalfedges.size()) {
            continue;
        }

        if (he >= supportingInfoPerHalfedge.size()) {
            continue;
        }

        SupportingInfo& info = supportingInfoPerHalfedge[he];
        uint32_t ref = info.supportingHE;
        if (ref == INVALID_INDEX || ref >= intrinsicHalfedges.size()) {
            continue;
        }

        uint32_t prevRef = intrinsicHalfedges[ref].prev;
        if (prevRef == INVALID_INDEX || prevRef >= intrinsicHalfedges.size()) {
            continue;
        }

        uint32_t matePrev = intrinsicHalfedges[prevRef].opposite;

        if (matePrev == h) {
            double theta = cornerAngleAtHalfedge(ref);

            double newAngle = info.supportingAngle + theta;

            if (newAngle <= 0.0) {
                info.supportingHE = h;

                info.supportingAngle = newAngle;
                clampSupportingAngle(info);
            }
        }
    }
}

bool SupportingHalfedge::flipEdge(uint32_t edgeIdx) {
    auto& intrinsicConn = intrinsicMesh.getConnectivity();
    const auto& edges = intrinsicConn.getEdges();

    if (edgeIdx >= edges.size()) {
        return false;
    }

    uint32_t ha1 = edges[edgeIdx].halfEdgeIdx;
    if (ha1 == INVALID_INDEX) {
        return false;
    }

    const auto& halfedges = intrinsicConn.getHalfEdges();
    uint32_t hb1 = halfedges[ha1].opposite;
    if (hb1 == INVALID_INDEX) {
        return false;
    }

    std::array<glm::dvec2,4> positions;
    if (!intrinsicConn.isFlippableEdge(edgeIdx, &positions)) {
        return false;
    }

    // Update before flip
    updateRemoval(ha1);
    updateRemoval(hb1);

    // Call flip on the intrinsic mesh
    if (!intrinsicConn.flipEdge(edgeIdx, &positions)) {
        return false;
    }

    // Update after flip
    updateInsertion(ha1);
    updateInsertion(hb1);

    return true;
}

int SupportingHalfedge::makeDelaunay(std::vector<uint32_t>* flippedEdges) {
    auto& intrinsicConn = intrinsicMesh.getConnectivity();
    const auto& edges = intrinsicConn.getEdges();
    std::vector<uint32_t> allEdges;
    allEdges.reserve(edges.size());
    for (uint32_t edgeIdx = 0; edgeIdx < edges.size(); ++edgeIdx) {
        if (edges[edgeIdx].halfEdgeIdx != INVALID_INDEX) {
            allEdges.push_back(edgeIdx);
        }
    }

    return makeDelaunayLocal(allEdges, flippedEdges);
}

int SupportingHalfedge::makeDelaunayLocal(const std::vector<uint32_t>& seedEdges, std::vector<uint32_t>* flippedEdges,
                                          const std::function<void(uint32_t)>& onFlip) {
    auto& intrinsicConn = intrinsicMesh.getConnectivity();
    int totalFlips = 0;
    std::queue<uint32_t> queueEdges;
    std::unordered_set<uint32_t> inQueueEdges;

    const auto& edges = intrinsicConn.getEdges();
    const auto& faces = intrinsicConn.getFaces();
    const auto& halfEdges = intrinsicConn.getHalfEdges();

    for (uint32_t edgeIdx : seedEdges) {
        if (edgeIdx >= edges.size()) {
            continue;
        }

        uint32_t he = edges[edgeIdx].halfEdgeIdx;
        if (he != INVALID_INDEX && !inQueueEdges.count(edgeIdx)) {
            queueEdges.push(edgeIdx);
            inQueueEdges.insert(edgeIdx);
        }
    }

    while (!queueEdges.empty()) {
        uint32_t edgeIdx = queueEdges.front();
        queueEdges.pop();
        inQueueEdges.erase(edgeIdx);

        if (edgeIdx >= edges.size()) {
            continue;
        }

        uint32_t he = edges[edgeIdx].halfEdgeIdx;
        if (he == INVALID_INDEX || he >= halfEdges.size() || intrinsicConn.isDelaunayEdge(he)) {
            continue;
        }

        if (flipEdge(edgeIdx)) {
            totalFlips++;
            if (flippedEdges) {
                flippedEdges->push_back(edgeIdx);
            }
            if (onFlip) {
                onFlip(edgeIdx);
            }

            he = edges[edgeIdx].halfEdgeIdx;
            if (he == INVALID_INDEX || he >= halfEdges.size()) {
                continue;
            }

            const uint32_t opp = halfEdges[he].opposite;
            const uint32_t faceA = halfEdges[he].face;
            const uint32_t faceB = (opp != INVALID_INDEX && opp < halfEdges.size()) ? halfEdges[opp].face : INVALID_INDEX;
            const uint32_t affectedFaces[2] = { faceA, faceB };

            for (uint32_t faceIdx : affectedFaces) {
                if (faceIdx == INVALID_INDEX || faceIdx >= faces.size()) {
                    continue;
                }

                uint32_t faceHe = faces[faceIdx].halfEdgeIdx;
                if (faceHe == INVALID_INDEX || faceHe >= halfEdges.size()) {
                    continue;
                }

                uint32_t curr = faceHe;
                for (int i = 0; i < 3; ++i) {
                    if (curr == INVALID_INDEX || curr >= halfEdges.size()) {
                        break;
                    }

                    uint32_t nhe = curr;
                    uint32_t neighEdgeIdx = intrinsicConn.getEdgeFromHalfEdge(nhe);
                    if (neighEdgeIdx != INVALID_INDEX && !inQueueEdges.count(neighEdgeIdx)) {
                        queueEdges.push(neighEdgeIdx);
                        inQueueEdges.insert(neighEdgeIdx);
                    }

                    curr = halfEdges[curr].next;
                }
            }
        }
    }

    return totalFlips;
}

void SupportingHalfedge::clampSupportingAngle(SupportingInfo& info) const {
    const double pi = glm::pi<double>();
    const double eps = 1e-12;

    if (info.supportingAngle > 0.0 && info.supportingAngle <= eps) {
        info.supportingAngle = 0.0;
    }
    if (info.supportingAngle <= -pi && info.supportingAngle > -pi - eps) {
        info.supportingAngle = -pi + eps;
    }
}

const SupportingHalfedge::SupportingInfo& SupportingHalfedge::getSupportingInfo(uint32_t inputHalfedgeIdx) const {
    static SupportingInfo invalid;
    if (inputHalfedgeIdx >= supportingInfoPerHalfedge.size()) {
        return invalid;
    }
    return supportingInfoPerHalfedge[inputHalfedgeIdx];
}

void SupportingHalfedge::trackInsertedVertex(uint32_t vertexIdx, const GeodesicTracer::SurfacePoint& surfacePoint) {
    InsertedVertexLocation loc;

    switch (surfacePoint.type) {
    case GeodesicTracer::SurfacePoint::Type::VERTEX:
        loc.locationType = 0;
        break;
    case GeodesicTracer::SurfacePoint::Type::EDGE:
        loc.locationType = 1;
        break;
    case GeodesicTracer::SurfacePoint::Type::FACE:
        loc.locationType = 2;
        break;
    }

    loc.elementId = surfacePoint.elementId;

    if (surfacePoint.type == GeodesicTracer::SurfacePoint::Type::EDGE) {
        double t = surfacePoint.split;
        loc.baryCoords = glm::vec3(1.0f - static_cast<float>(t), static_cast<float>(t), 0.0f);
    }
    else {
        loc.baryCoords = glm::vec3(
            static_cast<float>(surfacePoint.baryCoords.x),
            static_cast<float>(surfacePoint.baryCoords.y),
            static_cast<float>(surfacePoint.baryCoords.z)
        );
    }

    insertedVertexLocations[vertexIdx] = loc;
}

void SupportingHalfedge::untrackInsertedVertex(uint32_t vertexIdx) {
    insertedVertexLocations.erase(vertexIdx);
}

SupportingHalfedge::IntrinsicMesh SupportingHalfedge::buildIntrinsicMesh() const {
    IntrinsicMesh meshData;

    auto& intrinsicConn = intrinsicMesh.getConnectivity();
    const auto& intrinsicVertices = intrinsicConn.getVertices();
    const auto& intrinsicFaces = intrinsicConn.getFaces();
    const auto& intrinsicHalfedges = intrinsicConn.getHalfEdges();
    const auto& inputVertices = inputMesh.getConnectivity().getVertices();

    // Build deduplicated vertex buffer
    std::vector<uint32_t> vertexRemap(intrinsicVertices.size(), INVALID_INDEX);
    meshData.vertices.reserve(intrinsicVertices.size());

    for (uint32_t vertIdx = 0; vertIdx < intrinsicVertices.size(); ++vertIdx) {
        if (intrinsicVertices[vertIdx].halfEdgeIdx == INVALID_INDEX) {
            continue;
        }

        IntrinsicVertex vertex;
        vertex.intrinsicVertexId = vertIdx;
        vertex.position = intrinsicVertices[vertIdx].position;

        if (vertIdx < inputVertices.size()) {
            // Original vertex from input mesh
            vertex.inputLocationType = 0; // VERTEX
            vertex.inputElementId = vertIdx;
            vertex.inputBaryCoords = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        else {
            // Inserted vertex
            auto it = insertedVertexLocations.find(vertIdx);
            if (it == insertedVertexLocations.end()) {
                std::cerr << "[SupportingHalfedge] Inserted vertex " << vertIdx
                          << " has no tracked input location" << std::endl;
                return {};
            }
            const auto& loc = it->second;
            vertex.inputLocationType = loc.locationType;
            vertex.inputElementId = loc.elementId;
            vertex.inputBaryCoords = loc.baryCoords;
        }
        vertexRemap[vertIdx] = static_cast<uint32_t>(meshData.vertices.size());
        meshData.vertices.push_back(vertex);
    }

    // Build index buffer and face ID mapping
    meshData.indices.reserve(intrinsicFaces.size() * 3);
    meshData.faceIds.reserve(intrinsicFaces.size());
    meshData.triangles.reserve(intrinsicFaces.size());

    for (uint32_t faceIdx = 0; faceIdx < intrinsicFaces.size(); ++faceIdx) {
        const auto& face = intrinsicFaces[faceIdx];
        if (face.halfEdgeIdx == INVALID_INDEX) {
            continue;
        }

        // Get the three vertices of this face
        uint32_t he0 = face.halfEdgeIdx;
        uint32_t he1 = intrinsicHalfedges[he0].next;
        uint32_t he2 = intrinsicHalfedges[he1].next;

        uint32_t rawV0 = intrinsicHalfedges[he0].origin;
        uint32_t rawV1 = intrinsicHalfedges[he1].origin;
        uint32_t rawV2 = intrinsicHalfedges[he2].origin;

        if (rawV0 >= vertexRemap.size() ||
            rawV1 >= vertexRemap.size() ||
            rawV2 >= vertexRemap.size() ||
            vertexRemap[rawV0] == INVALID_INDEX ||
            vertexRemap[rawV1] == INVALID_INDEX ||
            vertexRemap[rawV2] == INVALID_INDEX) {
            std::cerr << "[SupportingHalfedge] Invalid live face "
                      << faceIdx << std::endl;
            return {};
        }

        uint32_t v0 = vertexRemap[rawV0];
        uint32_t v1 = vertexRemap[rawV1];
        uint32_t v2 = vertexRemap[rawV2];

        // Add indices for this triangle
        meshData.indices.push_back(v0);
        meshData.indices.push_back(v1);
        meshData.indices.push_back(v2);

        // Store intrinsic face ID
        meshData.faceIds.push_back(faceIdx);

        // Calculate triangle geometric data
        glm::vec3 p0 = meshData.vertices[v0].position;
        glm::vec3 p1 = meshData.vertices[v1].position;
        glm::vec3 p2 = meshData.vertices[v2].position;

        // Calculate edges
        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;

        // Calculate normal
        glm::vec3 crossProd = glm::cross(edge1, edge2);
        float crossLength = glm::length(crossProd);

        IntrinsicTriangle tri;
        tri.vertexIndices[0] = v0;
        tri.vertexIndices[1] = v1;
        tri.vertexIndices[2] = v2;
        tri.faceId = faceIdx;
        tri.center = (p0 + p1 + p2) / 3.0f;
        tri.area = crossLength * 0.5f;
        tri.normal = (crossLength > 1e-8f) ? (crossProd / crossLength) : glm::vec3(0.0f, 1.0f, 0.0f);

        meshData.triangles.push_back(tri);
    }

    // Calculate area weighted vertex normals from neighboring triangles
    calculateVertexNormals(meshData);

    return meshData;
}

void SupportingHalfedge::calculateVertexNormals(IntrinsicMesh& meshData) const {
    for (IntrinsicVertex& vertex : meshData.vertices) {
        vertex.normal = glm::vec3(0.0f);
    }

    for (const IntrinsicTriangle& tri : meshData.triangles) {
        const glm::vec3 weighted = tri.normal * tri.area;
        for (uint32_t vertexIdx : tri.vertexIndices) {
            meshData.vertices[vertexIdx].normal += weighted;
        }
    }

    for (IntrinsicVertex& vertex : meshData.vertices) {
        const float length = glm::length(vertex.normal);
        vertex.normal = (length > 0.0f)
            ? (vertex.normal / length)
            : glm::vec3(0.0f, 1.0f, 0.0f);
    }
}

SupportingHalfedge::GPUBuffers SupportingHalfedge::buildGPUBuffers(const IntrinsicMesh& exportedMesh) const {
    GPUBuffers buffers;

    const auto& inputConn = inputMesh.getConnectivity();
    auto& intrinsicConn = intrinsicMesh.getConnectivity();
    const auto& inputHalfedges = inputConn.getHalfEdges();
    const auto& inputFaces = inputConn.getFaces();
    const auto& inputEdges = inputConn.getEdges();
    const auto& intrinsicVertices = intrinsicConn.getVertices();
    const auto& intrinsicHalfedges = intrinsicConn.getHalfEdges();
    const auto& intrinsicEdges = intrinsicConn.getEdges();
    const auto& intrinsicFaces = intrinsicConn.getFaces();

    buffers.supportingHalfedges.resize(inputHalfedges.size(), -1);
    buffers.supportingAngles.resize(inputHalfedges.size(), 0.0f);
    for (size_t inputHe = 0; inputHe < inputHalfedges.size() && inputHe < supportingInfoPerHalfedge.size(); ++inputHe) {
        const SupportingInfo& info = supportingInfoPerHalfedge[inputHe];
        buffers.supportingHalfedges[inputHe] = static_cast<int32_t>(info.supportingHE);
        buffers.supportingAngles[inputHe] = static_cast<float>(info.supportingAngle);
    }

    std::vector<int32_t> vertexRemap(intrinsicVertices.size(), -1);
    for (uint32_t exportedIdx = 0; exportedIdx < exportedMesh.vertices.size(); ++exportedIdx) {
        const uint32_t rawIdx = exportedMesh.vertices[exportedIdx].intrinsicVertexId;
        if (rawIdx < vertexRemap.size()) {
            vertexRemap[rawIdx] = static_cast<int32_t>(exportedIdx);
        }
    }

    std::vector<int32_t> faceRemap(intrinsicFaces.size(), -1);
    for (uint32_t exportedIdx = 0; exportedIdx < exportedMesh.faceIds.size(); ++exportedIdx) {
        const uint32_t rawIdx = exportedMesh.faceIds[exportedIdx];
        if (rawIdx < faceRemap.size()) {
            faceRemap[rawIdx] = static_cast<int32_t>(exportedIdx);
        }
    }

    buffers.intrinsicHalfedgeData.reserve(intrinsicHalfedges.size() * 4);
    for (const auto& he : intrinsicHalfedges) {
        int32_t compactFace = (he.face != INVALID_INDEX &&
                               he.face < static_cast<uint32_t>(faceRemap.size()))
            ? faceRemap[he.face]
            : -1;
        int32_t compactOrigin = (he.origin < vertexRemap.size())
            ? vertexRemap[he.origin]
            : -1;
        buffers.intrinsicHalfedgeData.push_back(compactOrigin);
        buffers.intrinsicHalfedgeData.push_back(static_cast<int32_t>(he.edgeIdx));
        buffers.intrinsicHalfedgeData.push_back(compactFace);
        buffers.intrinsicHalfedgeData.push_back(static_cast<int32_t>(he.next));
    }

    buffers.intrinsicEdgeData.reserve(intrinsicEdges.size() * 2);
    for (const auto& edge : intrinsicEdges) {
        buffers.intrinsicEdgeData.push_back(static_cast<int32_t>(edge.halfEdgeIdx));
        if (edge.halfEdgeIdx != INVALID_INDEX && edge.halfEdgeIdx < intrinsicHalfedges.size()) {
            buffers.intrinsicEdgeData.push_back(static_cast<int32_t>(intrinsicHalfedges[edge.halfEdgeIdx].opposite));
        } else {
            buffers.intrinsicEdgeData.push_back(-1);
        }
    }

    buffers.intrinsicTriangleData.resize(exportedMesh.faceIds.size(), -1);
    for (uint32_t faceIdx = 0; faceIdx < intrinsicFaces.size(); ++faceIdx) {
        if (faceRemap[faceIdx] >= 0) {
            buffers.intrinsicTriangleData[faceRemap[faceIdx]] =
                static_cast<int32_t>(intrinsicFaces[faceIdx].halfEdgeIdx);
        }
    }

    buffers.intrinsicLengths.reserve(intrinsicEdges.size());
    for (const auto& edge : intrinsicEdges) {
        buffers.intrinsicLengths.push_back(static_cast<float>(edge.intrinsicLength));
    }


    buffers.inputHalfedgeData.reserve(inputHalfedges.size() * 4);
    for (const auto& he : inputHalfedges) {
        buffers.inputHalfedgeData.push_back(static_cast<int32_t>(he.origin));
        buffers.inputHalfedgeData.push_back(static_cast<int32_t>(he.edgeIdx));
        buffers.inputHalfedgeData.push_back(static_cast<int32_t>(he.face));
        buffers.inputHalfedgeData.push_back(static_cast<int32_t>(he.next));
    }

    buffers.inputEdgeData.reserve(inputEdges.size() * 2);
    for (const auto& edge : inputEdges) {
        buffers.inputEdgeData.push_back(static_cast<int32_t>(edge.halfEdgeIdx));
        if (edge.halfEdgeIdx != INVALID_INDEX && edge.halfEdgeIdx < inputHalfedges.size()) {
            buffers.inputEdgeData.push_back(static_cast<int32_t>(inputHalfedges[edge.halfEdgeIdx].opposite));
        } else {
            buffers.inputEdgeData.push_back(-1);
        }
    }

    buffers.inputTriangleData.reserve(inputFaces.size());
    for (const auto& face : inputFaces) {
        buffers.inputTriangleData.push_back(static_cast<int32_t>(face.halfEdgeIdx));
    }

    buffers.inputLengths.reserve(inputEdges.size());
    for (const auto& edge : inputEdges) {
        buffers.inputLengths.push_back(static_cast<float>(edge.intrinsicLength));
    }

    return buffers;
}

