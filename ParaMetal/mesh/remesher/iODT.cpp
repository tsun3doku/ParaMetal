#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/constants.hpp>

#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <algorithm>
#include <limits>
#include <string>
#include <deque>
#include <stdexcept>

#include "iODT.hpp"

iODT::iODT(const std::vector<float>& pointPositions, const std::vector<uint32_t>& triangleIndices)
    : tracer(intrinsicMesh), tracerInput(inputMesh) {

    inputMesh.buildSignposts(pointPositions, triangleIndices);
    inputMesh.updateAllSignposts();
    inputMesh.computeVertexAngleScales();
    inputMesh.buildHalfedgeVectorsInVertex();
    inputMesh.buildHalfedgeVectorsInFace();

    intrinsicMesh.buildSignposts(pointPositions, triangleIndices);
    auto& conn = intrinsicMesh.getConnectivity();
    intrinsicMesh.updateAllSignposts();
    intrinsicMesh.computeVertexAngleScales();
    intrinsicMesh.buildHalfedgeVectorsInVertex();
    intrinsicMesh.buildHalfedgeVectorsInFace();

    initializeVertexLocations();

    const auto& edges = conn.getEdges();
    for (size_t i = 0; i < edges.size(); ++i) {
        if (edges[i].halfEdgeIdx != INVALID_INDEX) {
            conn.getEdges()[i].isOriginal = true;
        }
    }

    supportingHalfedge = std::make_unique<SupportingHalfedge>(inputMesh, intrinsicMesh);
    supportingHalfedge->initialize();
}

iODT::~iODT() {
}

void iODT::refreshIntrinsicDirectionalData() {
    intrinsicMesh.rebuildVertexSums();
    intrinsicMesh.computeVertexAngleScales();
    intrinsicMesh.buildHalfedgeVectorsInVertex();
    intrinsicMesh.buildHalfedgeVectorsInFace();
}

size_t iODT::countInvalidIntrinsicFaces() const {
    const auto& conn = intrinsicMesh.getConnectivity();
    const auto& faces = conn.getFaces();
    size_t invalidFaces = 0;

    for (uint32_t f = 0; f < faces.size(); ++f) {
        if (!isValidFace(f)) {
            continue;
        }

        auto fhes = conn.getFaceHalfEdges(f);
        if (fhes.size() != 3) {
            continue;
        }

        double intrinsic[3] = {
            conn.getIntrinsicLengthFromHalfEdge(fhes[0]),
            conn.getIntrinsicLengthFromHalfEdge(fhes[1]),
            conn.getIntrinsicLengthFromHalfEdge(fhes[2])
        };
        std::sort(intrinsic, intrinsic + 3);
        if (!(intrinsic[2] < intrinsic[0] + intrinsic[1])) {
            ++invalidFaces;
        }
    }

    return invalidFaces;
}

bool iODT::optimalDelaunayTriangulation(int maxIterations, double minAngleDegrees, double maxEdgeLength, double stepSize) {
    insertedVertices.clear();

    // Delaunay phase
    supportingHalfedge->makeDelaunay();
    
    refreshIntrinsicDirectionalData();

    delaunayRefinement(minAngleDegrees);

    if (maxIterations > 0) {
        optimalReposition(maxIterations, 1e-4, maxEdgeLength, stepSize);
    }

    return countInvalidIntrinsicFaces() == 0;
}

int iODT::splitLongEdges(double maxEdgeLength, int maxSplits) {
    auto& conn = intrinsicMesh.getConnectivity();
    auto& edges = conn.getEdges();
    auto& halfEdges = conn.getHalfEdges();
    int splitCount = 0;

    if (maxSplits <= 0) {
        maxSplits = std::numeric_limits<int>::max();
    }

    std::vector<std::pair<uint32_t, uint32_t>> edgesToSplit;
    edgesToSplit.reserve(edges.size());
    for (uint32_t e = 0; e < edges.size(); ++e) {
        if (edges[e].halfEdgeIdx == INVALID_INDEX) {
            continue;
        }
        if (!(edges[e].intrinsicLength > maxEdgeLength)) {
            continue;
        }
        edgesToSplit.push_back({ e, edges[e].halfEdgeIdx });
    }

    std::vector<uint32_t> newlyInsertedVertices;
    newlyInsertedVertices.reserve(edgesToSplit.size());

    for (const auto& [edgeIdx, heIdx] : edgesToSplit) {
        if (splitCount >= maxSplits) {
            break;
        }
        if (edgeIdx >= edges.size() || heIdx == INVALID_INDEX || heIdx >= halfEdges.size()) {
            continue;
        }

        uint32_t newVertex = INVALID_INDEX;
        uint32_t diagFront = INVALID_INDEX;
        uint32_t diagBack = INVALID_INDEX;
        if (splitEdge(edgeIdx, newVertex, diagFront, diagBack, heIdx, 0.5)) {
            splitCount++;
            newlyInsertedVertices.push_back(newVertex);
        }
    }

    if (splitCount > 0) {
        std::unordered_set<uint32_t> patchEdges;

        for (uint32_t newV : newlyInsertedVertices) {
            if (newV >= conn.getVertices().size()) {
                continue;
            }

            std::vector<uint32_t> localEdges = collectLocalDelaunayEdges(conn, newV);
            std::vector<uint32_t> flippedEdges;
            supportingHalfedge->makeDelaunayLocal(localEdges, &flippedEdges);

            for (uint32_t edgeIdx : localEdges) {
                patchEdges.insert(edgeIdx);
            }
            for (uint32_t edgeIdx : flippedEdges) {
                patchEdges.insert(edgeIdx);
            }
        }

        updateLocalEdges(conn, std::vector<uint32_t>(patchEdges.begin(), patchEdges.end()));
    }

    return splitCount;
}

double iODT::repositionInsertedVertices(double stepSize, std::vector<uint32_t>& outMovedEdges) {
    auto& conn = intrinsicMesh.getConnectivity();
    auto& verts = conn.getVertices();
    auto& edges = conn.getEdges();
    auto& halfEdges = conn.getHalfEdges();

    const double EPS_LEN = 1e-12;
    double maxMove = 0.0;

    if (insertedVertices.empty()) {
        return maxMove;
    }

    // Process each vertex
    for (uint32_t vIdx : insertedVertices) {
        if (vIdx >= verts.size())
            continue;
        if (conn.isBoundaryVertex(vIdx))
            continue;

        // Build 2D ring with vertex at origin
        auto ring = conn.buildVertexRing2D(vIdx);
        if (ring.neighborVertexIndices.empty()) {
            continue;
        }

        // Calculate weighted circumcenter displacement
        glm::dvec2 avgVec(0.0);
        double avgLen = 0.0;
        bool ok = computeWeightedCircumcenter(ring, avgVec, avgLen);

        if (!ok || avgLen <= EPS_LEN)
            continue;

        // Record movement for convergence tracking
        double moveLen = avgLen * stepSize;

        // Trace from vertex along displacement vector on intrinsic mesh
        if (moveLen < 1e-12) {
            continue;
        }

        // New position in ring coords is the displacement from origin
        glm::dvec2 newPos2D = avgVec * stepSize;

        // Check for triangle inversion
        bool wouldInvert = false;
        double areaEPS = 1e-9 * avgLen * avgLen;

        for (size_t i = 0; i < ring.neighborVertexIndices.size(); ++i) {
            size_t nextI = (i + 1) % ring.neighborVertexIndices.size();

            glm::dvec2 p_i = ring.neighborPositions2D[i];
            glm::dvec2 p_next = ring.neighborPositions2D[nextI];

            // Calculate signed area
            double det = (p_i.x - newPos2D.x) * (p_next.y - newPos2D.y) - (p_i.y - newPos2D.y) * (p_next.x - newPos2D.x);

            // If area is negative or zero, dont move
            if (det <= areaEPS) {
                wouldInvert = true;
                break;
            }
        }

        if (wouldInvert) {
            continue; 
        }         

        if (moveLen > maxMove) {
            maxMove = moveLen;
        }

        auto outgoingHEs = conn.getVertexHalfEdges(vIdx);
        
        // Call updateRemoval for all incident halfedges before changing lengths
        for (uint32_t he : outgoingHEs) {
            uint32_t mate = halfEdges[he].opposite;
            if (mate != INVALID_INDEX) {
                supportingHalfedge->updateRemoval(mate);
            }
        }
        
        for (size_t i = 0; i < ring.neighborVertexIndices.size(); ++i) {
            uint32_t edgeIdx = ring.edgeIndices[i];
            glm::dvec2 neighborPos2D = ring.neighborPositions2D[i];

            edges[edgeIdx].intrinsicLength = glm::length(neighborPos2D - newPos2D);
        }
        
        for (uint32_t fIdx : ring.faceIndices) {
            intrinsicMesh.updateCornerAnglesForFace(fIdx);
        }

        // Call updateInsertion for all incident halfedges after updating corner angles
        for (uint32_t he : outgoingHEs) {
            uint32_t mate = halfEdges[he].opposite;
            if (mate != INVALID_INDEX) {
                supportingHalfedge->updateInsertion(mate);
            }
        }

        std::vector<uint32_t> ringVertices;
        ringVertices.reserve(ring.neighborVertexIndices.size() + 1);
        ringVertices.push_back(vIdx);
        for (uint32_t neighborIdx : ring.neighborVertexIndices) {
            ringVertices.push_back(neighborIdx);
        }

        intrinsicMesh.updateVertexSums(ringVertices);
        intrinsicMesh.updateVertexScales(ringVertices);

        if (!resolveVertex(vIdx)) {
            throw std::runtime_error("iODT: repositioned vertex could not be resolved");
        }

        for (uint32_t faceIdx : ring.faceIndices) {
            for (uint32_t he : conn.getFaceHalfEdges(faceIdx)) {
                uint32_t edgeIdx = conn.getEdgeFromHalfEdge(he);
                if (edgeIdx != INVALID_INDEX) {
                    outMovedEdges.push_back(edgeIdx);
                }
            }
        }

        intrinsicMesh.updateVertexVectors(ringVertices);
        intrinsicMesh.updateFaceVectors(ring.faceIndices);
    }

    return maxMove;
}

void iODT::optimalReposition(int maxIters, double tol, double maxEdgeLength, double stepSize)
{
    auto& conn = intrinsicMesh.getConnectivity();

    for (int iter = 0; iter < maxIters; ++iter) {
        if (insertedVertices.empty()) {
            break;
        }

        int splitCount = 0;
        if (maxEdgeLength > 0.0) {
            splitCount = splitLongEdges(maxEdgeLength, 0);
        }

        std::vector<uint32_t> movedEdges;
        movedEdges.reserve(insertedVertices.size() * 6);
        double maxMove = repositionInsertedVertices(stepSize, movedEdges);

        if (!movedEdges.empty()) {
            std::vector<uint32_t> flippedEdges;
            supportingHalfedge->makeDelaunayLocal(movedEdges, &flippedEdges);
            if (!flippedEdges.empty()) {
                updateLocalEdges(conn, flippedEdges);
            }
        }

        if (maxMove < tol && splitCount == 0) {
            break;
        }
    }
}

iODT::RefinementStatus iODT::delaunayRefinement(float minAngleDegrees) {
    const float MIN_ANGLE = minAngleDegrees * glm::pi<float>() / 180.0f;
    auto& conn = intrinsicMesh.getConnectivity();
    const auto& halfEdges = conn.getHalfEdges();

    std::priority_queue<FaceCandidate> faceQueue;
    std::deque<uint32_t> delaunayQueue;
    std::vector<uint8_t> inDelaunayQueue(conn.getEdges().size(), 0);

    RefinementCallbacks callbacks;
    callbacks.onFlip = [&](uint32_t edgeIdx) {
        std::vector<uint32_t> flipFaces;
        std::vector<uint32_t> flipNeighborEdges;
        collectFlipNeighborhood(edgeIdx, flipFaces, flipNeighborEdges);
        for (uint32_t faceIdx : flipFaces) {
            queueRefineFace(faceIdx, MIN_ANGLE, faceQueue);
        }
        for (uint32_t neighEdgeIdx : flipNeighborEdges) {
            queueDelaunayEdge(neighEdgeIdx, delaunayQueue, inDelaunayQueue);
        }
    };
    callbacks.onFace = [&](uint32_t faceIdx) {
        if (!isValidFace(faceIdx)) {
            return;
        }
        queueRefineFace(faceIdx, MIN_ANGLE, faceQueue);
        for (uint32_t he : conn.getFaceHalfEdges(faceIdx)) {
            queueDelaunayEdge(conn.getEdgeFromHalfEdge(he), delaunayQueue, inDelaunayQueue);
        }
    };

    const auto& faces = conn.getFaces();
    for (uint32_t faceIdx = 0; faceIdx < faces.size(); ++faceIdx) {
        queueRefineFace(faceIdx, MIN_ANGLE, faceQueue);
    }

    auto drainDelaunayQueue = [&]() -> bool {
        std::vector<uint32_t> flippedEdges;
        while (!delaunayQueue.empty()) {
            uint32_t edgeIdx = delaunayQueue.front();
            delaunayQueue.pop_front();
            if (edgeIdx < inDelaunayQueue.size()) {
                inDelaunayQueue[edgeIdx] = 0;
            }

            if (!isValidEdge(edgeIdx)) {
                continue;
            }

            const auto& edgesNow = conn.getEdges();
            uint32_t he = edgesNow[edgeIdx].halfEdgeIdx;
            if (he == INVALID_INDEX || he >= halfEdges.size()) {
                continue;
            }
            if (conn.isDelaunayEdge(he)) {
                continue;
            }

            bool flipped = supportingHalfedge->flipEdge(edgeIdx);
            if (!flipped) {
                continue;
            }

            flippedEdges.push_back(edgeIdx);
            if (flippedEdges.size() == 100000) {
                std::cerr << "[iODT] delaunay drain passed 100000 flips in one pass at edge="
                          << edgeIdx << std::endl;
            }
            if (callbacks.onFlip) {
                callbacks.onFlip(edgeIdx);
            }
        }

        if (!flippedEdges.empty()) {
            updateLocalEdges(conn, flippedEdges);
            return true;
        }
        return false;
    };

    const auto& edges = conn.getEdges();
    for (uint32_t edgeIdx = 0; edgeIdx < edges.size(); ++edgeIdx) {
        queueDelaunayEdge(edgeIdx, delaunayQueue, inDelaunayQueue);
    }

    int recheckCount = 0;
    constexpr int MAX_RECHECK_COUNT = 5;

    while (true) {
        drainDelaunayQueue();

        bool inserted = false;
        while (!faceQueue.empty()) {
            FaceCandidate candidate = faceQueue.top();
            faceQueue.pop();

            if (!isValidFace(candidate.faceIdx)) {
                continue;
            }

            float currentArea = intrinsicMesh.computeFaceArea(candidate.faceIdx);
            if (currentArea != candidate.area) {
                continue;
            }
            if (!needsRefinement(candidate.faceIdx, MIN_ANGLE)) {
                continue;
            }

            uint32_t newV = INVALID_INDEX;
            if (!insertCircumcenter(candidate.faceIdx, newV, &callbacks)) {
                continue;
            }
            if (newV == INVALID_INDEX || newV >= conn.getVertices().size()) {
                continue;
            }

            if (inDelaunayQueue.size() < conn.getEdges().size()) {
                inDelaunayQueue.resize(conn.getEdges().size(), 0);
            }

            for (uint32_t faceIdx : conn.getVertexFaces(newV)) {
                queueRefineFace(faceIdx, MIN_ANGLE, faceQueue);
                for (uint32_t he : conn.getFaceHalfEdges(faceIdx)) {
                    queueDelaunayEdge(conn.getEdgeFromHalfEdge(he), delaunayQueue, inDelaunayQueue);
                }
            }

            inserted = true;
            break;
        }

        if (inserted) {
            continue;
        }

        bool flipped = drainDelaunayQueue();
        if (flipped) {
            continue;
        }

        bool anyFound = false;
        if (recheckCount < MAX_RECHECK_COUNT) {
            ++recheckCount;

            for (uint32_t faceIdx = 0; faceIdx < conn.getFaces().size(); ++faceIdx) {
                const size_t before = faceQueue.size();
                queueRefineFace(faceIdx, MIN_ANGLE, faceQueue);
                anyFound = anyFound || faceQueue.size() != before;
            }

            for (uint32_t edgeIdx = 0; edgeIdx < conn.getEdges().size(); ++edgeIdx) {
                if (!isValidEdge(edgeIdx)) {
                    continue;
                }

                uint32_t he = conn.getEdges()[edgeIdx].halfEdgeIdx;
                if (he == INVALID_INDEX || he >= conn.getHalfEdges().size()) {
                    continue;
                }

                if (!conn.isDelaunayEdge(he)) {
                    queueDelaunayEdge(edgeIdx, delaunayQueue, inDelaunayQueue);
                    anyFound = true;
                }
            }

            if (anyFound) {
                continue;
            }
        }

        bool anyEligible = false;
        for (uint32_t faceIdx = 0; faceIdx < conn.getFaces().size(); ++faceIdx) {
            if (needsRefinement(faceIdx, MIN_ANGLE)) {
                anyEligible = true;
                break;
            }
        }

        if (anyEligible) {
            return RefinementStatus::Partial;
        }

        return RefinementStatus::Complete;
    }
}

bool iODT::isValidFace(uint32_t faceIdx) const {
    const auto& conn = intrinsicMesh.getConnectivity();
    const auto& faces = conn.getFaces();
    const auto& halfEdges = conn.getHalfEdges();
    if (faceIdx >= faces.size() || faces[faceIdx].halfEdgeIdx == HalfEdgeMesh::INVALID_INDEX) {
        return false;
    }

    uint32_t he0 = faces[faceIdx].halfEdgeIdx;
    if (he0 >= halfEdges.size()) {
        return false;
    }
    uint32_t he1 = halfEdges[he0].next;
    if (he1 >= halfEdges.size()) {
        return false;
    }
    uint32_t he2 = halfEdges[he1].next;
    if (he2 >= halfEdges.size()) {
        return false;
    }

    return halfEdges[he2].next == he0;
}

bool iODT::isValidEdge(uint32_t edgeIdx) const {
    const auto& conn = intrinsicMesh.getConnectivity();
    const auto& edges = conn.getEdges();
    const auto& halfEdges = conn.getHalfEdges();
    if (edgeIdx >= edges.size()) {
        return false;
    }

    uint32_t he = edges[edgeIdx].halfEdgeIdx;
    return he != HalfEdgeMesh::INVALID_INDEX && he < halfEdges.size();
}

size_t iODT::countSharpCorners(SignpostMesh& mesh, const uint32_t* faceHEs) {
    auto& conn = mesh.getConnectivity();
    const auto& halfEdges = conn.getHalfEdges();
    size_t sharpCorners = 0;

    for (int i = 0; i < 3; ++i) {
        const uint32_t he = faceHEs[i];
        if (he == HalfEdgeMesh::INVALID_INDEX || he >= halfEdges.size()) {
            continue;
        }

        double angleSum = 0.0;
        uint32_t curr = he;
        bool validWalk = true;

        for (size_t step = 0; step < halfEdges.size(); ++step) {
            if (curr == HalfEdgeMesh::INVALID_INDEX || curr >= halfEdges.size()) {
                validWalk = false;
                break;
            }

            angleSum += mesh.getCornerAngle(curr);

            uint32_t next = conn.getNextAroundVertex(curr);
            if (next == HalfEdgeMesh::INVALID_INDEX || next >= halfEdges.size()) {
                validWalk = false;
                break;
            }

            curr = next;
            if (curr == he || conn.isFixedEdge(conn.getEdgeFromHalfEdge(curr))) {
                break;
            }
        }

        if (!validWalk) {
            continue;
        }

        if (curr != he && !conn.isFixedEdge(conn.getEdgeFromHalfEdge(he))) {
            curr = he;
            for (size_t step = 0; step < halfEdges.size(); ++step) {
                curr = conn.getPrevAroundVertex(curr);
                if (curr == HalfEdgeMesh::INVALID_INDEX || curr >= halfEdges.size()) {
                    validWalk = false;
                    break;
                }

                angleSum += mesh.getCornerAngle(curr);

                if (conn.isFixedEdge(conn.getEdgeFromHalfEdge(curr))) {
                    break;
                }
            }
        }

        if (validWalk && angleSum < glm::pi<double>() / 3.0) {
            ++sharpCorners;
        }
    }

    return sharpCorners;
}

uint32_t iODT::getParentInputFace(uint32_t faceIdx) const {
    if (!isValidFace(faceIdx)) {
        return INVALID_INDEX;
    }

    const auto& conn = intrinsicMesh.getConnectivity();
    const auto& inputConn = inputMesh.getConnectivity();
    const auto& inputFaces = inputConn.getFaces();
    const auto& inputHalfEdges = inputConn.getHalfEdges();
    const auto& inputEdges = inputConn.getEdges();

    std::vector<uint32_t> intrinsicVerts = conn.getFaceVertices(faceIdx);
    if (intrinsicVerts.size() != 3) {
        return INVALID_INDEX;
    }

    GeodesicTracer::SurfacePoint locations[3];
    for (int i = 0; i < 3; ++i) {
        auto it = intrinsicVertexLocations.find(intrinsicVerts[i]);
        if (it == intrinsicVertexLocations.end()) {
            return INVALID_INDEX;
        }
        locations[i] = it->second;
    }

    auto locationOnFace = [&](const GeodesicTracer::SurfacePoint& loc, uint32_t inputFace) -> bool {
        if (inputFace >= inputFaces.size() || inputFaces[inputFace].halfEdgeIdx == INVALID_INDEX) {
            return false;
        }

        switch (loc.type) {
        case GeodesicTracer::SurfacePoint::Type::FACE:
            return loc.elementId == inputFace;

        case GeodesicTracer::SurfacePoint::Type::EDGE: {
            if (loc.elementId >= inputEdges.size()) {
                return false;
            }
            uint32_t edgeHE = inputEdges[loc.elementId].halfEdgeIdx;
            if (edgeHE == INVALID_INDEX || edgeHE >= inputHalfEdges.size()) {
                return false;
            }
            if (inputHalfEdges[edgeHE].face == inputFace) {
                return true;
            }
            uint32_t opp = inputHalfEdges[edgeHE].opposite;
            return opp != INVALID_INDEX && opp < inputHalfEdges.size() &&
                   inputHalfEdges[opp].face == inputFace;
        }

        case GeodesicTracer::SurfacePoint::Type::VERTEX: {
            for (uint32_t f : inputConn.getVertexFaces(loc.elementId)) {
                if (f == inputFace) {
                    return true;
                }
            }
            return false;
        }
        }

        return false;
    };

    std::vector<uint32_t> candidates;
    const GeodesicTracer::SurfacePoint& first = locations[0];
    if (first.type == GeodesicTracer::SurfacePoint::Type::FACE) {
        candidates.push_back(first.elementId);
    }
    else if (first.type == GeodesicTracer::SurfacePoint::Type::EDGE) {
        if (first.elementId < inputEdges.size()) {
            uint32_t edgeHE = inputEdges[first.elementId].halfEdgeIdx;
            if (edgeHE != INVALID_INDEX && edgeHE < inputHalfEdges.size()) {
                candidates.push_back(inputHalfEdges[edgeHE].face);
                uint32_t opp = inputHalfEdges[edgeHE].opposite;
                if (opp != INVALID_INDEX && opp < inputHalfEdges.size()) {
                    candidates.push_back(inputHalfEdges[opp].face);
                }
            }
        }
    }
    else {
        candidates = inputConn.getVertexFaces(first.elementId);
    }

    for (uint32_t candidate : candidates) {
        if (candidate == INVALID_INDEX) {
            continue;
        }
        if (locationOnFace(locations[0], candidate) &&
            locationOnFace(locations[1], candidate) &&
            locationOnFace(locations[2], candidate)) {
            return candidate;
        }
    }

    return INVALID_INDEX;
}

bool iODT::needsRefinement(uint32_t faceIdx, float minAngleThreshold) {
    if (minAngleThreshold <= 0.0f) {
        return false;
    }

    const auto& conn = intrinsicMesh.getConnectivity();
    const auto& faces = conn.getFaces();
    const auto& halfEdges = conn.getHalfEdges();

    if (faceIdx >= faces.size() || faces[faceIdx].halfEdgeIdx == HalfEdgeMesh::INVALID_INDEX) {
        return false;
    }

    const uint32_t he0 = faces[faceIdx].halfEdgeIdx;
    if (he0 >= halfEdges.size()) {
        return false;
    }
    const uint32_t he1 = halfEdges[he0].next;
    if (he1 >= halfEdges.size()) {
        return false;
    }
    const uint32_t he2 = halfEdges[he1].next;
    if (he2 >= halfEdges.size() || halfEdges[he2].next != he0) {
        return false;
    }
    const uint32_t faceHEs[3] = { he0, he1, he2 };

    bool hasRefinableSmallAngle = false;
    for (uint32_t he : faceHEs) {
        double angle = intrinsicMesh.getCornerAngle(he);
        if (!(angle > 0.0 && angle < minAngleThreshold)) {
            continue;
        }

        uint32_t heNext = halfEdges[he].next;
        if (heNext == HalfEdgeMesh::INVALID_INDEX || heNext >= halfEdges.size()) {
            continue;
        }

        uint32_t hePrev = halfEdges[heNext].next;
        bool isDegreeOneVertex = (hePrev != HalfEdgeMesh::INVALID_INDEX &&
                                  hePrev < halfEdges.size() &&
                                  halfEdges[he].opposite == hePrev);
        if (isDegreeOneVertex) {
            continue;
        }

        if (conn.isFixedEdge(conn.getEdgeFromHalfEdge(he)) &&
            conn.isFixedEdge(conn.getEdgeFromHalfEdge(hePrev))) {
            continue;
        }

        hasRefinableSmallAngle = true;
        break;
    }

    if (!hasRefinableSmallAngle) {
        return false;
    }

    size_t needleCorners = countSharpCorners(intrinsicMesh, faceHEs);

    if (needleCorners == 1) {
        return false;
    }

    uint32_t parentFace = getParentInputFace(faceIdx);
    if (parentFace != INVALID_INDEX) {
        const auto& inputConn = inputMesh.getConnectivity();
        for (uint32_t inputVertex : inputConn.getFaceVertices(parentFace)) {
            if (inputMesh.getVertexAngleSum(inputVertex) < glm::pi<double>() / 3.0) {
                return false;
            }
        }
    }

    return true;
}

float iODT::refinementPriority(uint32_t faceIdx, float area) {
    const auto& conn = intrinsicMesh.getConnectivity();
    const auto& faces = conn.getFaces();
    const auto& halfEdges = conn.getHalfEdges();

    if (faceIdx >= faces.size()) {
        return area;
    }

    uint32_t he = faces[faceIdx].halfEdgeIdx;
    for (int i = 0; i < 3 && he < halfEdges.size(); ++i) {
        if (conn.isFixedEdge(conn.getEdgeFromHalfEdge(he))) {
            return std::numeric_limits<float>::infinity();
        }
        he = halfEdges[he].next;
    }

    return area;
}

void iODT::queueRefineFace(uint32_t faceIdx, float minAngleThreshold, std::priority_queue<FaceCandidate>& faceQueue) {
    if (!needsRefinement(faceIdx, minAngleThreshold)) {
        return;
    }

    FaceCandidate candidate;
    candidate.faceIdx = faceIdx;
    candidate.area = intrinsicMesh.computeFaceArea(faceIdx);
    candidate.priority = refinementPriority(faceIdx, candidate.area);
    faceQueue.push(candidate);
}

void iODT::queueDelaunayEdge(uint32_t edgeIdx, std::deque<uint32_t>& edgeQueue, std::vector<uint8_t>& inQueue) {
    if (!isValidEdge(edgeIdx)) {
        return;
    }
    if (edgeIdx >= inQueue.size()) {
        inQueue.resize(edgeIdx + 1, 0);
    }
    if (inQueue[edgeIdx]) {
        return;
    }
    edgeQueue.push_back(edgeIdx);
    inQueue[edgeIdx] = 1;
}

std::vector<uint32_t> iODT::collectLocalDelaunayEdges(HalfEdgeMesh& conn, uint32_t vertexIdx) {
    std::unordered_set<uint32_t> patchFaces;
    std::unordered_set<uint32_t> patchEdges;

    if (vertexIdx >= conn.getVertices().size()) {
        return {};
    }

    const auto& halfEdges = conn.getHalfEdges();
    for (uint32_t faceIdx : conn.getVertexFaces(vertexIdx)) {
        if (faceIdx == HalfEdgeMesh::INVALID_INDEX) {
            continue;
        }

        patchFaces.insert(faceIdx);
        for (uint32_t he : conn.getFaceHalfEdges(faceIdx)) {
            if (he == HalfEdgeMesh::INVALID_INDEX || he >= halfEdges.size()) {
                continue;
            }

            uint32_t opp = halfEdges[he].opposite;
            if (opp != HalfEdgeMesh::INVALID_INDEX) {
                uint32_t oppFace = halfEdges[opp].face;
                if (oppFace != HalfEdgeMesh::INVALID_INDEX) {
                    patchFaces.insert(oppFace);
                }
            }
        }
    }

    for (uint32_t faceIdx : patchFaces) {
        for (uint32_t he : conn.getFaceHalfEdges(faceIdx)) {
            uint32_t edgeIdx = conn.getEdgeFromHalfEdge(he);
            if (edgeIdx != HalfEdgeMesh::INVALID_INDEX) {
                patchEdges.insert(edgeIdx);
            }
        }
    }

    return std::vector<uint32_t>(patchEdges.begin(), patchEdges.end());
}

void iODT::updateLocalEdges(HalfEdgeMesh& conn, const std::vector<uint32_t>& edgePatch) {
    std::unordered_set<uint32_t> patchFaces;
    std::unordered_set<uint32_t> patchVertices;
    const auto& edges = conn.getEdges();
    const auto& halfEdges = conn.getHalfEdges();

    for (uint32_t edgeIdx : edgePatch) {
        if (edgeIdx >= edges.size()) {
            continue;
        }

        uint32_t he = edges[edgeIdx].halfEdgeIdx;
        if (he == HalfEdgeMesh::INVALID_INDEX || he >= halfEdges.size()) {
            continue;
        }

        uint32_t faceA = halfEdges[he].face;
        if (faceA != HalfEdgeMesh::INVALID_INDEX) {
            patchFaces.insert(faceA);
        }

        uint32_t opp = halfEdges[he].opposite;
        if (opp != HalfEdgeMesh::INVALID_INDEX && opp < halfEdges.size()) {
            uint32_t faceB = halfEdges[opp].face;
            if (faceB != HalfEdgeMesh::INVALID_INDEX) {
                patchFaces.insert(faceB);
            }
        }
    }

    for (uint32_t faceIdx : patchFaces) {
        for (uint32_t he : conn.getFaceHalfEdges(faceIdx)) {
            if (he != HalfEdgeMesh::INVALID_INDEX && he < halfEdges.size()) {
                patchVertices.insert(halfEdges[he].origin);
            }
        }
    }

    std::vector<uint32_t> affectedVertices(patchVertices.begin(), patchVertices.end());
    std::vector<uint32_t> affectedFaces(patchFaces.begin(), patchFaces.end());

    intrinsicMesh.updateVertexSums(affectedVertices);
    intrinsicMesh.updateVertexScales(affectedVertices);
    intrinsicMesh.updateVertexVectors(affectedVertices);
    intrinsicMesh.updateFaceVectors(affectedFaces);
}

bool iODT::insertCircumcenter(uint32_t faceIdx, uint32_t& outNewVertex, const RefinementCallbacks* callbacks) {
    auto& conn = intrinsicMesh.getConnectivity();
    const auto& faces = conn.getFaces();

    // Validate face index
    if (faceIdx >= faces.size()) {
        return false;
    }
    if (faces[faceIdx].halfEdgeIdx == INVALID_INDEX) {
        return false;
    }

    uint32_t he0 = faces[faceIdx].halfEdgeIdx;
    const auto& halfEdges = conn.getHalfEdges();
    if (he0 == INVALID_INDEX || he0 >= halfEdges.size()) {
        return false;
    }
    uint32_t he1 = halfEdges[he0].next;
    if (he1 == INVALID_INDEX || he1 >= halfEdges.size()) {
        return false;
    }
    uint32_t he2 = halfEdges[he1].next;
    if (he2 == INVALID_INDEX || he2 >= halfEdges.size() || halfEdges[he2].next != he0) {
        return false;
    }

    double a = conn.getIntrinsicLengthFromHalfEdge(he1);
    double b = conn.getIntrinsicLengthFromHalfEdge(he2);
    double c = conn.getIntrinsicLengthFromHalfEdge(he0);
    if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(c)) ||
        a <= 1e-12 || b <= 1e-12 || c <= 1e-12) {
        return false;
    }

    double a2 = a * a;
    double b2 = b * b;
    double c2 = c * c;
    glm::dvec3 circumcenterBary(
        a2 * (b2 + c2 - a2),
        b2 * (c2 + a2 - b2),
        c2 * (a2 + b2 - c2)
    );
    double barySum = circumcenterBary.x + circumcenterBary.y + circumcenterBary.z;
    if (std::abs(barySum) <= 1e-30) {
        return false;
    }
    circumcenterBary /= barySum;

    if (!std::isfinite(circumcenterBary.x) ||
        !std::isfinite(circumcenterBary.y) ||
        !std::isfinite(circumcenterBary.z)) {
        return false;
    }

    glm::dvec3 startBary(1.0 / 3.0);

    // Convert bary start and target to cartesian for traceFromFace
    auto tri = intrinsicMesh.layoutTriangle(faceIdx);
    glm::dvec2 start2D = tri.vertices[0] * startBary.x + tri.vertices[1] * startBary.y + tri.vertices[2] * startBary.z;
    glm::dvec2 circ2D  = tri.vertices[0] * circumcenterBary.x + tri.vertices[1] * circumcenterBary.y + tri.vertices[2] * circumcenterBary.z;
    glm::dvec2 traceDir = circ2D - start2D;
    double traceLen = glm::length(traceDir);

    if (traceLen < 1e-12) {
        return false;
    }

    GeodesicTracer::GeodesicTraceResult intrinsicRes = tracer.traceFromFace(
        faceIdx,
        startBary,
        traceDir,
        traceLen
    );

    if (!intrinsicRes.success) {
        return false;
    }

    // Pick insertion type
    GeodesicTracer::SurfacePoint& surfacePoint = intrinsicRes.exitPoint;

    if (surfacePoint.type == GeodesicTracer::SurfacePoint::Type::EDGE) {
        uint32_t edgeIdx = surfacePoint.elementId;
        if (edgeIdx == INVALID_INDEX || edgeIdx >= conn.getEdges().size()) {
            return false;
        }

        uint32_t heSplit = conn.getEdges()[edgeIdx].halfEdgeIdx;
        double splitLength = conn.getEdges()[edgeIdx].intrinsicLength;
        uint32_t diagFront = INVALID_INDEX;
        uint32_t diagBack = INVALID_INDEX;
        if (!splitEdge(edgeIdx, outNewVertex, diagFront, diagBack, heSplit, 0.5, callbacks)) {
            return false;
        }

        deleteInsertedVerticesNear(outNewVertex, 0.5 * splitLength, callbacks);

        return true;
    }

    if (surfacePoint.type == GeodesicTracer::SurfacePoint::Type::FACE) {
        bool wasInserted = false;
        if (!insertPoint(surfacePoint.elementId, surfacePoint.baryCoords, outNewVertex, &wasInserted, callbacks)) {
            return false;
        }
        if (!wasInserted) {
            return false;
        }
        return true;
    }

    return false;
}

bool iODT::insertPoint(uint32_t faceIdx, const glm::dvec3& baryCoords, uint32_t& outVertex, bool* outWasInserted, const RefinementCallbacks* callbacks) {
	if (outWasInserted) {
		*outWasInserted = false;
	}

	auto& conn = intrinsicMesh.getConnectivity();
	auto& halfEdges = conn.getHalfEdges();
	const auto& faces = conn.getFaces();

	if (faceIdx >= faces.size() || faces[faceIdx].halfEdgeIdx == INVALID_INDEX) {
		return false;
	}

	const double EPS_BARY = 1e-8;

	glm::dvec3 b = baryCoords;
	if (!std::isfinite(b.x) || !std::isfinite(b.y) || !std::isfinite(b.z)) {
		return false;
	}

    b.x = std::max(0.0, b.x);
    b.y = std::max(0.0, b.y);
    b.z = std::max(0.0, b.z);
    double sum = b.x + b.y + b.z;
    if (sum <= 1e-16) {
        return false;
    }
    b /= sum;

    std::vector<uint32_t> faceHEs = conn.getFaceHalfEdges(faceIdx);
    if (faceHEs.size() != 3) {
        return false;
    }
    uint32_t he0 = faceHEs[0]; // v0 -> v1
    uint32_t he1 = faceHEs[1]; // v1 -> v2
    uint32_t he2 = faceHEs[2]; // v2 -> v0

    uint32_t v0 = halfEdges[he0].origin;
    uint32_t v1 = halfEdges[he1].origin;
    uint32_t v2 = halfEdges[he2].origin;

    // If the point is on an existing point then return it
    if (b.x >= 1.0 - EPS_BARY) {
        outVertex = v0;
        return true;
    }
    if (b.y >= 1.0 - EPS_BARY) {
        outVertex = v1;
        return true;
    }
    if (b.z >= 1.0 - EPS_BARY) {
        outVertex = v2;
        return true;
    }

    // If the point is on an edge then split it
    if (b.z <= EPS_BARY) {
        double denom = b.x + b.y;
        if (denom <= 1e-16) {
            return false;
        }
        double t = b.y / denom; // along v0->v1
        return splitEdge(he0, t, outVertex, outWasInserted, callbacks);
    }

    if (b.x <= EPS_BARY) {
        double denom = b.y + b.z;
        if (denom <= 1e-16) {
            return false;
        }
        double t = b.z / denom; // along v1->v2
        return splitEdge(he1, t, outVertex, outWasInserted, callbacks);
    }

    if (b.y <= EPS_BARY) {
        double denom = b.z + b.x;
        if (denom <= 1e-16) {
            return false;
        }
        double t = b.x / denom; // along v2->v0
        return splitEdge(he2, t, outVertex, outWasInserted, callbacks);
    }

    // Else, insert into the face 
    auto tri2D = intrinsicMesh.layoutTriangle(faceIdx);
    glm::dvec2 P0 = tri2D.vertices[0];
    glm::dvec2 P1 = tri2D.vertices[1];
    glm::dvec2 P2 = tri2D.vertices[2];

    glm::dvec2 s2D = P0 * b.x + P1 * b.y + P2 * b.z;
    double R0 = glm::length(s2D - P0);
    double R1 = glm::length(s2D - P1);
    double R2 = glm::length(s2D - P2);

    // Call updateRemoval on boundary halfedges before topology changes
    for (uint32_t he : faceHEs) {
        supportingHalfedge->updateRemoval(he);
    }

    HalfEdgeMesh::TriangleSplit split = conn.splitTriangleIntrinsic(faceIdx, R0, R1, R2);
    uint32_t newV = split.newV;
    if (newV == HalfEdgeMesh::INVALID_INDEX) {
        return false;
    }

    uint32_t he_to_v0 = split.heV0ToNewV;
    uint32_t he_to_v1 = split.heV1ToNewV;
    uint32_t he_to_v2 = split.heV2ToNewV;

    // Resize halfedge vectors buffer
    intrinsicMesh.getHalfedgeVectorsInVertex().resize(halfEdges.size(), glm::dvec2(0.0));

    // Update corner angles for only new faces around the inserted vertex
    auto newFaces = conn.getVertexFaces(newV);
    for (uint32_t fIdx : newFaces) {
        if (fIdx != HalfEdgeMesh::INVALID_INDEX) {
            intrinsicMesh.updateCornerAnglesForFace(fIdx);
        }
    }

    // Update supporting halfedges after corner angles are set
    if (he_to_v0 < halfEdges.size()) {
        supportingHalfedge->updateInsertion(he_to_v0);
    }
    if (he_to_v1 < halfEdges.size()) {
        supportingHalfedge->updateInsertion(he_to_v1);
    }
    if (he_to_v2 < halfEdges.size()) {
        supportingHalfedge->updateInsertion(he_to_v2);
    }

    // Update the original boundary halfedges that were reused
    for (uint32_t he : faceHEs) {
        supportingHalfedge->updateInsertion(he);
    }

    std::vector<uint32_t> scaleVertices;
    scaleVertices.reserve(faceHEs.size() + 1);
    for (uint32_t he : faceHEs) {
        if (he < halfEdges.size()) {
            scaleVertices.push_back(halfEdges[he].origin);
        }
    }
    scaleVertices.push_back(newV);

    intrinsicMesh.updateVertexSums(scaleVertices);
    intrinsicMesh.updateVertexScales(scaleVertices);

    insertedVertices.insert(newV);

    if (!resolveVertex(newV)) {
        return false;
    }

    intrinsicMesh.updateVertexVectors(scaleVertices);
    intrinsicMesh.updateFaceVectors(newFaces);

    outVertex = newV;

    if (outWasInserted) {
        *outWasInserted = true;
    }

    return true;
}

bool iODT::splitEdge(uint32_t heEdge, double tParam, uint32_t& outNewV, bool* outWasInserted, const RefinementCallbacks* callbacks) {
    if (outWasInserted) {
        *outWasInserted = false;
    }

    auto& conn = intrinsicMesh.getConnectivity();
    auto& halfEdges = conn.getHalfEdges();

    const double EPS_T = 1e-8;

    tParam = std::clamp(tParam, 0.0, 1.0);
    if (tParam <= EPS_T) {
        outNewV = halfEdges[heEdge].origin;
        return true;
    }
    uint32_t heNext = halfEdges[heEdge].next;
    if (heNext == HalfEdgeMesh::INVALID_INDEX) {
        return false;
    }
    if (1.0 - tParam <= EPS_T) {
        outNewV = halfEdges[heNext].origin;
        return true;
    }

    uint32_t edgeIdx = conn.getEdgeFromHalfEdge(heEdge);
    if (edgeIdx == HalfEdgeMesh::INVALID_INDEX) {
        return false;
    }

    double originalLength = conn.getIntrinsicLengthFromHalfEdge(heEdge);

    uint32_t diagF = HalfEdgeMesh::INVALID_INDEX;
    uint32_t diagB = HalfEdgeMesh::INVALID_INDEX;
    uint32_t newV = HalfEdgeMesh::INVALID_INDEX;
    if (!splitEdge(edgeIdx, newV, diagF, diagB, heEdge, tParam, callbacks)) {
        return false;
    }

    deleteInsertedVerticesNear(newV, std::max(tParam, 1.0 - tParam) * originalLength, callbacks);

    if (outWasInserted) {
        *outWasInserted = true;
    }

    outNewV = newV;
    return true;
}

bool iODT::splitEdge(uint32_t edgeIdx, uint32_t& outNewVertex, uint32_t& outDiagFront, uint32_t& outDiagBack, uint32_t HESplit, double t, const RefinementCallbacks* callbacks) {
    auto& conn = intrinsicMesh.getConnectivity();
    auto& edges = conn.getEdges();
    auto& halfEdges = conn.getHalfEdges();

    if (edgeIdx >= edges.size()) {
        return false;
    }

    // Use the passed in halfedge 
    uint32_t parentHE = HESplit;
    uint32_t oppHE = halfEdges[parentHE].opposite;

    // Verify the halfedge actually belongs to this edge
    uint32_t edgeFromHE = conn.getEdgeFromHalfEdge(HESplit);
    if (edgeFromHE != edgeIdx) {
        return false;
    }

    uint32_t originalVA = halfEdges[parentHE].origin;
    uint32_t originalVB = halfEdges[halfEdges[parentHE].next].origin;

    // Set split fraction
    double splitFraction = t;

    // Get original edge length
    double originalLength = conn.getIntrinsicLengthFromHalfEdge(parentHE);

    if (originalLength <= 1e-12) {
        return false;
    }

    // Precalculate diagonal lengths before split
    std::vector<double> precomputedDiagonals;
    uint32_t face1 = halfEdges[parentHE].face;
    uint32_t face2 = (oppHE != HalfEdgeMesh::INVALID_INDEX)
        ? halfEdges[oppHE].face : HalfEdgeMesh::INVALID_INDEX;

    double diagLen1 = 0.0;
    if (face1 != HalfEdgeMesh::INVALID_INDEX) {
        diagLen1 = intrinsicMesh.computeSplitDiagonalLength(face1, originalVA, originalVB, splitFraction);
    }

    double diagLen2 = 0.0;
    if (face2 != HalfEdgeMesh::INVALID_INDEX) {
        diagLen2 = intrinsicMesh.computeSplitDiagonalLength(face2, originalVA, originalVB, splitFraction);
    }

    if (!(std::isfinite(diagLen1) && diagLen1 > 1e-12)) {
        return false;
    }

    if (face2 != HalfEdgeMesh::INVALID_INDEX && !(std::isfinite(diagLen2) && diagLen2 > 1e-12)) {
        return false;
    }

    precomputedDiagonals = { diagLen1, diagLen2 };

    // Resolve the new vertex on the input surface before changing any topology
    uint32_t endpointCandidates[2] = { originalVA, originalVB };
    uint32_t preferredStart = INVALID_INDEX;
    chooseTraceStart(endpointCandidates, 2, preferredStart);

    GeodesicTracer::GeodesicTraceResult newLocation;
    uint32_t traceStart = INVALID_INDEX;
    for (int attempt = 0; attempt < 2 && traceStart == INVALID_INDEX; ++attempt) {
        uint32_t endpoint = (attempt == 0)
            ? preferredStart
            : (preferredStart == originalVA ? originalVB : originalVA);
        if (endpoint == INVALID_INDEX) {
            continue;
        }

        auto locIt = intrinsicVertexLocations.find(endpoint);
        if (locIt == intrinsicVertexLocations.end()) {
            continue;
        }

        bool fromVA = (endpoint == originalVA);
        uint32_t dirHE = fromVA ? parentHE : oppHE;
        if (dirHE == INVALID_INDEX) {
            continue;
        }

        glm::dvec2 edgeVec = intrinsicMesh.halfedgeVector(dirHE);
        double edgeVecLen = glm::length(edgeVec);
        if (!(edgeVecLen > 1e-12)) {
            continue;
        }

        double traceLen = fromVA
            ? splitFraction * originalLength
            : (1.0 - splitFraction) * originalLength;
        if (traceFromLocation(endpoint, locIt->second, edgeVec / edgeVecLen, traceLen, newLocation)) {
            traceStart = endpoint;
        }
    }

    if (traceStart == INVALID_INDEX) {
        return false;
    }

    if (newLocation.steps.empty()) {
        return false;
    }
    const glm::dvec2& incomingDir = newLocation.steps.back().dir2D;
    double incomingLen = glm::length(incomingDir);
    if (!(incomingLen > 1e-12) || !std::isfinite(incomingDir.x) || !std::isfinite(incomingDir.y)) {
        return false;
    }
    glm::dvec2 outgoingVec = -incomingDir / incomingLen;

    double splitAngleFromVA = halfEdges[parentHE].signpostAngle;

    // Call updateRemoval on the original edge halfedges before topology changes
    supportingHalfedge->updateRemoval(parentHE);
    if (oppHE != HalfEdgeMesh::INVALID_INDEX) {
        supportingHalfedge->updateRemoval(oppHE);
    }

    // 5) Topology split
    edges[edgeIdx].halfEdgeIdx = parentHE;
    auto R = conn.splitEdgeTopo(edgeIdx, splitFraction);
    if (R.newV == HalfEdgeMesh::INVALID_INDEX) {
        return false;
    }

    uint32_t newV = R.newV;
    uint32_t diagFront = R.diagFront;
    uint32_t diagBack = R.diagBack;
    conn.getVertices()[newV].halfEdgeIdx = R.heB;
    halfEdges[R.heA].signpostAngle = splitAngleFromVA;

    uint32_t anchorHE = R.heB;
    if (traceStart == originalVA) {
        uint32_t heBack = halfEdges[R.heA].opposite;
        if (heBack != HalfEdgeMesh::INVALID_INDEX) {
            anchorHE = heBack;
        }
    }

    // Set diagonal lengths before corner angle calculations
    for (int i = 0; i < 2; ++i) {
        uint32_t diagHE = (i == 0 ? R.diagFront : R.diagBack);
        double L = precomputedDiagonals[i];
        if (diagHE == HalfEdgeMesh::INVALID_INDEX) {
            continue;
        }

        uint32_t diagEdgeIdx = conn.getEdgeFromHalfEdge(diagHE);
        if (diagEdgeIdx != HalfEdgeMesh::INVALID_INDEX && diagEdgeIdx < edges.size()) {
            edges[diagEdgeIdx].intrinsicLength = L;
        }
    }

    // Update the corner angles only for affected faces
    auto adjacentFaces = conn.getVertexFaces(newV);
    for (uint32_t fIdx : adjacentFaces) {
        if (fIdx != HalfEdgeMesh::INVALID_INDEX) {
            intrinsicMesh.updateCornerAnglesForFace(fIdx);
        }
    }

    // Update supporting halfedges
    auto newVertexHEs = conn.getVertexHalfEdges(newV);
    for (uint32_t he : newVertexHEs) {
        supportingHalfedge->updateInsertion(he);
        uint32_t opp = halfEdges[he].opposite;
        if (opp != HalfEdgeMesh::INVALID_INDEX) {
            supportingHalfedge->updateInsertion(opp);
        }
    }

    std::unordered_set<uint32_t> patchVertexSet;
    for (uint32_t fIdx : adjacentFaces) {
        for (uint32_t vIdx : conn.getFaceVertices(fIdx)) {
            patchVertexSet.insert(vIdx);
        }
    }
    std::vector<uint32_t> patchVertices(patchVertexSet.begin(), patchVertexSet.end());

    intrinsicMesh.getHalfedgeVectorsInVertex().resize(
        halfEdges.size(), glm::dvec2(std::numeric_limits<double>::quiet_NaN()));

    intrinsicMesh.updateVertexSums(patchVertices);
    intrinsicMesh.updateVertexScales(patchVertices);

    storeVertexLocation(newV, newLocation);
    insertedVertices.insert(newV);
    anchorFanToArrival(newV, anchorHE, outgoingVec);

    // Call updateAngleFromCWNeighbor on all incoming halfedges first
    for (uint32_t heOut : conn.getVertexHalfEdges(newV)) {
        uint32_t twinHe = halfEdges[heOut].opposite;
        if (twinHe != HalfEdgeMesh::INVALID_INDEX) {
            intrinsicMesh.updateAngleFromCWNeighbor(twinHe);
        }
    }

    intrinsicMesh.updateVertexVectors(patchVertices);
    intrinsicMesh.updateFaceVectors(adjacentFaces);

    outNewVertex = newV;
    outDiagFront = diagFront;
    outDiagBack = diagBack;

    return true;
}

void iODT::collectFlipNeighborhood(uint32_t edgeIdx, std::vector<uint32_t>& outFaces,
                                   std::vector<uint32_t>& outEdges) const {
    outFaces.clear();
    outEdges.clear();

    const auto& conn = intrinsicMesh.getConnectivity();
    const auto& edges = conn.getEdges();
    const auto& halfEdges = conn.getHalfEdges();
    if (edgeIdx >= edges.size()) {
        return;
    }

    uint32_t he = edges[edgeIdx].halfEdgeIdx;
    if (he == INVALID_INDEX || he >= halfEdges.size()) {
        return;
    }
    uint32_t heT = halfEdges[he].opposite;
    if (heT == INVALID_INDEX || heT >= halfEdges.size()) {
        return;
    }

    uint32_t faceA = halfEdges[he].face;
    uint32_t faceB = halfEdges[heT].face;
    if (faceA != INVALID_INDEX) {
        outFaces.push_back(faceA);
    }
    if (faceB != INVALID_INDEX) {
        outFaces.push_back(faceB);
    }

    uint32_t heN = halfEdges[he].next;
    uint32_t heTN = halfEdges[heT].next;
    if (heN == INVALID_INDEX || heN >= halfEdges.size() ||
        heTN == INVALID_INDEX || heTN >= halfEdges.size()) {
        return;
    }

    uint32_t neighHalfedges[4] = {
        heN,
        halfEdges[heN].next,
        heTN,
        halfEdges[heTN].next
    };
    for (uint32_t nhe : neighHalfedges) {
        if (nhe == INVALID_INDEX || nhe >= halfEdges.size()) {
            continue;
        }
        outEdges.push_back(conn.getEdgeFromHalfEdge(nhe));
    }
}

uint32_t iODT::removeInsertedVertex(uint32_t vertexIdx, const RefinementCallbacks* callbacks) {
    auto& conn = intrinsicMesh.getConnectivity();
    const auto& verts = conn.getVertices();
    const auto& halfEdges = conn.getHalfEdges();

    if (vertexIdx >= verts.size() || verts[vertexIdx].halfEdgeIdx == INVALID_INDEX) {
        return INVALID_INDEX;
    }
    if (insertedVertices.find(vertexIdx) == insertedVertices.end()) {
        return INVALID_INDEX;
    }

    auto locIt = intrinsicVertexLocations.find(vertexIdx);
    if (locIt == intrinsicVertexLocations.end() ||
        locIt->second.type != GeodesicTracer::SurfacePoint::Type::FACE) {
        return INVALID_INDEX;
    }
    if (conn.isBoundaryVertex(vertexIdx)) {
        return INVALID_INDEX;
    }

    std::vector<uint32_t> flippedEdges;
    size_t guard = 0;
    size_t degree = intrinsicMesh.getVertexDegree(vertexIdx);
    const size_t guardLimit = 10 * degree;

    while (degree != 3) {
        bool flipped = false;
        for (uint32_t heOut : conn.getVertexHalfEdges(vertexIdx)) {
            uint32_t edgeIdx = conn.getEdgeFromHalfEdge(heOut);
            if (edgeIdx == INVALID_INDEX) {
                continue;
            }
            bool flippedNow = supportingHalfedge->flipEdge(edgeIdx);
            if (flippedNow) {
                flippedEdges.push_back(edgeIdx);
                if (callbacks && callbacks->onFlip) {
                    callbacks->onFlip(edgeIdx);
                }
                flipped = true;
                break;
            }
        }
        if (!flipped || guard >= guardLimit) {
            break;
        }
        ++guard;
        degree = intrinsicMesh.getVertexDegree(vertexIdx);
    }

    if (!flippedEdges.empty()) {
        updateLocalEdges(conn, flippedEdges);
    }
    if (degree != 3) {
        return INVALID_INDEX;
    }

    std::vector<uint32_t> outgoing = conn.getVertexHalfEdges(vertexIdx);
    if (outgoing.size() != 3) {
        return INVALID_INDEX;
    }

    const double MERGE_EPS = 1e-12;
    double l0 = conn.getIntrinsicLengthFromHalfEdge(halfEdges[outgoing[0]].next);
    double l1 = conn.getIntrinsicLengthFromHalfEdge(halfEdges[outgoing[1]].next);
    double l2 = conn.getIntrinsicLengthFromHalfEdge(halfEdges[outgoing[2]].next);
    if (!(l0 + l1 > l2 + MERGE_EPS && l1 + l2 > l0 + MERGE_EPS && l2 + l0 > l1 + MERGE_EPS)) {
        return INVALID_INDEX;
    }

    uint32_t twins[3] = {
        halfEdges[outgoing[0]].opposite,
        halfEdges[outgoing[1]].opposite,
        halfEdges[outgoing[2]].opposite
    };
    for (int i = 0; i < 3; ++i) {
        if (twins[i] == INVALID_INDEX || twins[i] >= halfEdges.size()) {
            return INVALID_INDEX;
        }
    }

    std::unordered_set<uint32_t> invalidatedHalfedges;
    SupportingHalfedge::RemovalPlan removalPlan;
    for (int i = 0; i < 3; ++i) {
        invalidatedHalfedges.insert(outgoing[i]);
        invalidatedHalfedges.insert(twins[i]);
    }

    for (uint32_t he : invalidatedHalfedges) {
        supportingHalfedge->prepareRemoval(he, removalPlan);
    }

    if (supportingHalfedge->planReferencesInvalidated(removalPlan, invalidatedHalfedges)) {
        std::cerr << "[iODT] removal would invalidate a supporting halfedge reference"
                  << " for vertex " << vertexIdx << std::endl;
        return INVALID_INDEX;
    }

    std::vector<uint32_t> ringVertices = {
        halfEdges[twins[0]].origin,
        halfEdges[twins[1]].origin,
        halfEdges[twins[2]].origin
    };

    uint32_t newFace = conn.removeVertex(vertexIdx);
    if (newFace == INVALID_INDEX) {
        return INVALID_INDEX;
    }

    supportingHalfedge->commitRemoval(removalPlan);

    intrinsicMesh.updateCornerAnglesForFace(newFace);
    intrinsicMesh.updateVertexSums(ringVertices);
    intrinsicMesh.updateVertexScales(ringVertices);
    intrinsicMesh.updateVertexVectors(ringVertices);
    intrinsicMesh.updateFaceVectors(std::vector<uint32_t>{ newFace });

    insertedVertices.erase(vertexIdx);
    intrinsicVertexLocations.erase(vertexIdx);
    vertexResolutionFaces.erase(vertexIdx);
    supportingHalfedge->untrackInsertedVertex(vertexIdx);
    if (callbacks && callbacks->onFace) {
        callbacks->onFace(newFace);
    }

    return newFace;
}

void iODT::collectVerticesWithinBall(uint32_t startVertex, double radius,
                                     std::vector<uint32_t>& outVertices) const {
    outVertices.clear();
    if (!(radius > 0.0)) {
        return;
    }

    const auto& conn = intrinsicMesh.getConnectivity();
    const auto& edges = conn.getEdges();
    const auto& verts = conn.getVertices();

    using Entry = std::pair<double, uint32_t>;
    std::unordered_map<uint32_t, double> settled;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    queue.emplace(0.0, startVertex);

    while (!queue.empty()) {
        Entry top = queue.top();
        queue.pop();
        uint32_t v = top.second;
        if (settled.find(v) != settled.end()) {
            continue;
        }
        settled.emplace(v, top.first);
        if (v != startVertex) {
            outVertices.push_back(v);
        }

        for (uint32_t he : conn.getVertexHalfEdges(v)) {
            uint32_t edgeIdx = conn.getEdgeFromHalfEdge(he);
            if (edgeIdx >= edges.size()) {
                continue;
            }
            double length = edges[edgeIdx].intrinsicLength;
            if (!(length > 0.0)) {
                continue;
            }
            auto edgeVerts = conn.getEdgeVertices(edgeIdx);
            uint32_t other = (edgeVerts.first == v) ? edgeVerts.second : edgeVerts.first;
            if (other >= verts.size()) {
                continue;
            }
            double nextDist = top.first + length;
            if (nextDist <= radius && settled.find(other) == settled.end()) {
                queue.emplace(nextDist, other);
            }
        }
    }
}

void iODT::deleteInsertedVerticesNear(uint32_t insertedVertex, double ballRadius, const RefinementCallbacks* callbacks) {
    if (!(ballRadius > 0.0)) {
        return;
    }

    auto& conn = intrinsicMesh.getConnectivity();
    std::vector<uint32_t> localEdges = collectLocalDelaunayEdges(conn, insertedVertex);
    std::vector<uint32_t> flippedEdges;
    std::function<void(uint32_t)> onFlip;
    if (callbacks) {
        onFlip = callbacks->onFlip;
    }
    supportingHalfedge->makeDelaunayLocal(localEdges, &flippedEdges, onFlip);
    if (!flippedEdges.empty()) {
        updateLocalEdges(conn, flippedEdges);
    }

    std::vector<uint32_t> nearby;
    collectVerticesWithinBall(insertedVertex, 2.0 * ballRadius, nearby);

    for (uint32_t v : nearby) {
        if (v == insertedVertex) {
            continue;
        }
        removeInsertedVertex(v, callbacks);
    }
}

bool iODT::computeWeightedCircumcenter(const HalfEdgeMesh::VertexRing2D& ring, glm::dvec2& outAvgVec, double& outAvgLen) {
    const double EPS_LEN = 1e-12;

    outAvgVec = glm::dvec2(0.0);
    outAvgLen = 0.0;

    if (ring.neighborVertexIndices.empty()) {
        return false;
    }

    // Calculate area weighted average of vectors to circumcenters in ring coordinates
    glm::dvec2 accum(0.0);
    double accumW = 0.0;
    // Calculate circumcenter for each neighbor pair
    for (size_t i = 0; i < ring.neighborVertexIndices.size(); ++i) {
        size_t nextI = (i + 1) % ring.neighborVertexIndices.size();

        // Find which face this corresponds to 
        uint32_t f = HalfEdgeMesh::INVALID_INDEX;
        if (i < ring.faceIndices.size()) {
            f = ring.faceIndices[i];
        }

        if (f == HalfEdgeMesh::INVALID_INDEX)
            continue;

        double area = intrinsicMesh.computeFaceArea(f);
        if (!(area > 0.0))
            continue;

        // Get the face's vertices in ring coordinates
        glm::dvec2 v0 = glm::dvec2(0.0, 0.0); 
        glm::dvec2 v1 = ring.neighborPositions2D[i];
        glm::dvec2 v2 = ring.neighborPositions2D[nextI];

        // Calculate circumcenter in ring coordinates
        glm::dvec2 cc2d = intrinsicMesh.computeCircumcenter2D(v0, v1, v2);
        if (!std::isfinite(cc2d.x) || !std::isfinite(cc2d.y)) {
            continue;
        }

        // Vector from vertex (at origin) to circumcenter
        glm::dvec2 vectorToCircumcenter = cc2d;

        double L = glm::length(vectorToCircumcenter);
        if (L <= EPS_LEN)
            continue;

        // Area weighted accumulation
        accum += area * vectorToCircumcenter;
        accumW += area;
    }

    if (!(accumW > 0.0)) {
        return false;
    }

    outAvgVec = accum / accumW;
    outAvgLen = glm::length(outAvgVec);

    return true;
}

bool iODT::chooseTraceStart(const uint32_t* candidates, size_t count, uint32_t& outStart) const {
    const auto& verts = intrinsicMesh.getConnectivity().getVertices();

    int bestPriority = 9999;
    for (size_t i = 0; i < count; ++i) {
        uint32_t v = candidates[i];
        if (v >= verts.size()) {
            continue;
        }
        if (intrinsicVertexLocations.find(v) == intrinsicVertexLocations.end()) {
            continue;
        }

        int priority = 2;
        if (verts[v].originalIndex != HalfEdgeMesh::INVALID_INDEX) {
            priority = 1;
        }
        if (priority < bestPriority) {
            bestPriority = priority;
            outStart = v;
        }
    }

    return bestPriority != 9999;
}

bool iODT::traceFromLocation(uint32_t startVertex, const GeodesicTracer::SurfacePoint& start,
                             const glm::dvec2& dir2D, double length,
                             GeodesicTracer::GeodesicTraceResult& outTrace) {
    outTrace = GeodesicTracer::GeodesicTraceResult();

    if (!(length > 1e-12) || !std::isfinite(dir2D.x) || !std::isfinite(dir2D.y)) {
        return false;
    }

    const auto& inputConn = inputMesh.getConnectivity();

    if (start.type == GeodesicTracer::SurfacePoint::Type::VERTEX) {
        uint32_t refFace = INVALID_INDEX;
        for (uint32_t he : inputConn.getVertexHalfEdges(start.elementId)) {
            uint32_t face = inputConn.getHalfEdges()[he].face;
            if (face != INVALID_INDEX) {
                refFace = face;
                break;
            }
        }
        if (refFace == INVALID_INDEX) {
            return false;
        }

        GeodesicTracer::GeodesicTraceResult baseResult;
        baseResult.success = true;
        baseResult.distance = 0.0;
        outTrace = tracerInput.traceFromVertex(start.elementId, dir2D, length, baseResult, length);
    }
    else if (start.type == GeodesicTracer::SurfacePoint::Type::EDGE) {
        uint32_t refFace = INVALID_INDEX;
        auto rfIt = vertexResolutionFaces.find(startVertex);
        if (rfIt != vertexResolutionFaces.end()) {
            refFace = rfIt->second;
        }
        outTrace = tracerInput.traceFromEdge(start.elementId, start.split, dir2D, length, refFace);
    }
    else if (start.type == GeodesicTracer::SurfacePoint::Type::FACE) {
        outTrace = tracerInput.traceFromFace(start.elementId, start.baryCoords, dir2D, length);
    }

    return outTrace.success;
}

void iODT::storeVertexLocation(uint32_t vertexIdx, const GeodesicTracer::GeodesicTraceResult& trace) {
    auto& verts = intrinsicMesh.getConnectivity().getVertices();

    if (vertexIdx >= verts.size()) {
        return;
    }

    intrinsicVertexLocations[vertexIdx] = trace.exitPoint;
    if (trace.finalFaceIdx != INVALID_INDEX) {
        vertexResolutionFaces[vertexIdx] = trace.finalFaceIdx;
    }
    verts[vertexIdx].position = trace.position3D;

    supportingHalfedge->trackInsertedVertex(vertexIdx, trace.exitPoint);
}

bool iODT::resolveVertex(uint32_t vertexIdx) {
    auto& conn = intrinsicMesh.getConnectivity();
    auto& verts = conn.getVertices();
    auto& halfEdges = conn.getHalfEdges();

    std::vector<uint32_t> incomingHEs;
    for (uint32_t heOut : conn.getVertexHalfEdges(vertexIdx)) {
        uint32_t heIn = halfEdges[heOut].opposite;
        if (heIn != INVALID_INDEX) {
            incomingHEs.push_back(heIn);
        }
    }
    if (incomingHEs.empty()) {
        return false;
    }

    for (uint32_t heIn : incomingHEs) {
        intrinsicMesh.updateAngleFromCWNeighbor(heIn);
    }

    uint32_t inputTraceHe = INVALID_INDEX;
    int bestPriority = 9999;
    double bestLength = std::numeric_limits<double>::max();

    for (uint32_t heIn : incomingHEs) {
        uint32_t neighbor = halfEdges[heIn].origin;
        if (neighbor >= verts.size()) {
            continue;
        }

        if (intrinsicVertexLocations.find(neighbor) == intrinsicVertexLocations.end()) {
            if (verts[neighbor].originalIndex == INVALID_INDEX) {
                continue;
            }
            intrinsicVertexLocations[neighbor] =
                GeodesicTracer::SurfacePoint(verts[neighbor].originalIndex);
        }

        int priority = (verts[neighbor].originalIndex != INVALID_INDEX) ? 1 : 2;
        uint32_t opp = halfEdges[heIn].opposite;
        if (opp == INVALID_INDEX || halfEdges[opp].face == INVALID_INDEX) {
            priority = 3;
        }

        double edgeLength = conn.getIntrinsicLengthFromHalfEdge(heIn);
        if (priority < bestPriority || (priority == bestPriority && edgeLength < bestLength)) {
            bestPriority = priority;
            bestLength = edgeLength;
            inputTraceHe = heIn;
        }
    }

    if (inputTraceHe == INVALID_INDEX) {
        return false;
    }

    uint32_t anchorHE = halfEdges[inputTraceHe].opposite;
    if (anchorHE == INVALID_INDEX || halfEdges[anchorHE].origin != vertexIdx) {
        return false;
    }

    uint32_t traceStart = halfEdges[inputTraceHe].origin;
    glm::dvec2 traceVec = intrinsicMesh.halfedgeVector(inputTraceHe);
    double traceLength = conn.getIntrinsicLengthFromHalfEdge(inputTraceHe);

    GeodesicTracer::GeodesicTraceResult resolved;
    if (!traceFromLocation(traceStart, intrinsicVertexLocations[traceStart],
                           traceVec, traceLength, resolved)) {
        return false;
    }
    if (resolved.steps.empty()) {
        return false;
    }

    glm::dvec2 outgoingVec = -resolved.steps.back().dir2D;
    if (!(glm::length(outgoingVec) > 1e-12) ||
        !std::isfinite(outgoingVec.x) || !std::isfinite(outgoingVec.y)) {
        return false;
    }

    storeVertexLocation(vertexIdx, resolved);
    anchorFanToArrival(vertexIdx, anchorHE, outgoingVec);
    return true;
}

void iODT::anchorFanToArrival(uint32_t vertexIdx, uint32_t anchorHE,
                              const glm::dvec2& outgoingVec) {    auto& conn = intrinsicMesh.getConnectivity();
    auto& verts = conn.getVertices();
    auto& halfEdges = conn.getHalfEdges();

    if (vertexIdx >= verts.size()) {
        return;
    }
    if (anchorHE == INVALID_INDEX || anchorHE >= halfEdges.size() ||
        halfEdges[anchorHE].origin != vertexIdx) {
        std::cerr << "[iODT] invalid anchor halfedge for inserted vertex " << vertexIdx << std::endl;
        return;
    }
    if (!std::isfinite(outgoingVec.x) || !std::isfinite(outgoingVec.y) ||
        glm::length(outgoingVec) <= 1e-12) {
        std::cerr << "[iODT] invalid arrival direction for inserted vertex " << vertexIdx << std::endl;
        return;
    }

    if (conn.isBoundaryVertex(vertexIdx)) {
        if (conn.isInteriorHalfEdge(anchorHE)) {
            std::cerr << "[iODT] boundary anchor is not a boundary halfedge for vertex " << vertexIdx << std::endl;
            return;
        }

        verts[vertexIdx].halfEdgeIdx = anchorHE;
        halfEdges[anchorHE].signpostAngle = 0.0;
    } else {
        halfEdges[anchorHE].signpostAngle = intrinsicMesh.standardizeAngleForVertex(
            vertexIdx, std::atan2(outgoingVec.y, outgoingVec.x));
    }

    uint32_t currHe = conn.getNextAroundVertex(anchorHE);
    while (currHe != INVALID_INDEX && currHe != anchorHE) {
        intrinsicMesh.updateAngleFromCWNeighbor(currHe);
        if (!conn.isInteriorHalfEdge(currHe)) {
            break;
        }
        currHe = conn.getNextAroundVertex(currHe);
    }
}

bool iODT::isEdgeOriginal(uint32_t edgeIdx) const {
    auto& conn = intrinsicMesh.getConnectivity();
    const auto& edges = conn.getEdges();
    return edgeIdx < edges.size() ? edges[edgeIdx].isOriginal : false;
}

void iODT::initializeVertexLocations() {
    auto& conn = intrinsicMesh.getConnectivity();
    const auto& vertices = conn.getVertices();

    // Initialize all original vertices to map 1:1 to input model vertices
    for (uint32_t vIdx = 0; vIdx < vertices.size(); ++vIdx) {
        if (vertices[vIdx].halfEdgeIdx == HalfEdgeMesh::INVALID_INDEX) 
            continue;

        // Initialize all original vertices 
        if (vertices[vIdx].originalIndex != HalfEdgeMesh::INVALID_INDEX) {
            GeodesicTracer::SurfacePoint location;
            location.type = GeodesicTracer::SurfacePoint::Type::VERTEX;
            location.elementId = vertices[vIdx].originalIndex;
            location.baryCoords = glm::dvec3(1.0, 0.0, 0.0);
            location.split = 0.0;

            intrinsicVertexLocations[vIdx] = location;
        }
    }
}

GeodesicTracer::GeodesicTraceResult iODT::traceIntrinsicHalfedgeAlongInput(uint32_t intrinsicHalfedgeIdx) {
    auto& conn = intrinsicMesh.getConnectivity();
    const auto& HEs = conn.getHalfEdges();
    const auto& inputVertices = inputMesh.getConnectivity().getVertices();

    if (intrinsicHalfedgeIdx >= HEs.size())
        return {};

    const auto& intrinsicHE = HEs[intrinsicHalfedgeIdx];
    uint32_t intrinsicStartV = intrinsicHE.origin;
    uint32_t intrinsicEndV = HEs[intrinsicHE.next].origin;
    uint32_t intrinsicEdgeIdx = conn.getEdgeFromHalfEdge(intrinsicHalfedgeIdx);

    // Short circuit for original edges
    if (isEdgeOriginal(intrinsicEdgeIdx)) {
        auto itA = intrinsicVertexLocations.find(intrinsicStartV);
        auto itB = intrinsicVertexLocations.find(intrinsicEndV);

        if (itA == intrinsicVertexLocations.end() || itB == intrinsicVertexLocations.end()) {
            return {};
        }

        GeodesicTracer::GeodesicTraceResult result;
        result.success = true;
        result.pathPoints = { itA->second, itB->second };
        result.distance = glm::length(intrinsicMesh.halfedgeVector(intrinsicHalfedgeIdx));
        
        // Add a single step
        GeodesicTracer::FaceStepResult step;
        step.success = true;
        step.distanceTraveled = result.distance;
        result.steps.push_back(step);
        
        return result;
    }

    // Lookup mapped endpoints
    auto itStart = intrinsicVertexLocations.find(intrinsicStartV);
    auto itEnd = intrinsicVertexLocations.find(intrinsicEndV);

    if (itStart == intrinsicVertexLocations.end()) {
        return {};
    }

    if (itEnd == intrinsicVertexLocations.end()) {
        return {};
    }

    // Check if the mapped vertices exist in input mesh
    GeodesicTracer::SurfacePoint startSP = itStart->second;
    if (startSP.type == GeodesicTracer::SurfacePoint::Type::VERTEX &&
        startSP.elementId >= inputVertices.size()) {
        return {};
    }

    GeodesicTracer::SurfacePoint endSP = itEnd->second;

    // Get intrinsic vector
    glm::dvec2 traceVec = intrinsicMesh.halfedgeVector(intrinsicHalfedgeIdx);
    double traceLen = glm::length(traceVec);

    if (traceLen < 1e-12) {
        GeodesicTracer::GeodesicTraceResult result;
        result.success = true;
        result.pathPoints = { startSP, endSP };
        result.distance = 0.0;
        
        // Add a step with zero distance
        GeodesicTracer::FaceStepResult step;
        step.success = true;
        step.distanceTraveled = 0.0;
        result.steps.push_back(step);
        
        return result;
    }

    glm::dvec2 traceDir = traceVec / traceLen;

    GeodesicTracer::GeodesicTraceResult base;
    base.success = true;
    base.pathPoints.clear();
    base.pathPoints.push_back(startSP);
    base.distance = 0.0;
    
    uint32_t startVertex = startSP.elementId;
    if (startSP.type == GeodesicTracer::SurfacePoint::Type::EDGE) {
        // For edge tracing, get the start vertex from the intrinsic halfedge
        auto& meshConn = intrinsicMesh.getConnectivity();
        const auto& halfEdge = meshConn.getHalfEdges()[intrinsicHalfedgeIdx];
        startVertex = halfEdge.origin;
    }

    // Trace on input mesh based on start point type
    GeodesicTracer::GeodesicTraceResult result;

    if (startSP.type == GeodesicTracer::SurfacePoint::Type::VERTEX) {
        result = tracerInput.traceFromVertex(startSP.elementId, traceDir, traceLen, base, traceLen);
    }
    else if (startSP.type == GeodesicTracer::SurfacePoint::Type::FACE) {
        result = tracerInput.traceFromFace(startSP.elementId, startSP.baryCoords, traceDir, traceLen);
    }
    else if (startSP.type == GeodesicTracer::SurfacePoint::Type::EDGE) {

        // Get the resolution face for the start vertex (which face's chart the direction is expressed in)
        uint32_t resolutionFace = HalfEdgeMesh::INVALID_INDEX;
        auto it = vertexResolutionFaces.find(startVertex);
        if (it != vertexResolutionFaces.end()) {
            resolutionFace = it->second;
        }

        result = tracerInput.traceFromEdge(startSP.elementId, startSP.split, traceDir, traceLen, resolutionFace);
    }

    if (result.success) {
        return result;
    }
    else {
        GeodesicTracer::GeodesicTraceResult fallback;
        fallback.success = false;
        fallback.pathPoints = { startSP, endSP };
        fallback.distance = traceLen;
        return fallback;
    }
}

std::vector<glm::vec3> iODT::getCommonSubdivision(uint32_t intrinsicHalfedgeIdx) const {
    std::vector<glm::vec3> polyline;

    // Trace the intrinsic halfedge to get surface points
    auto traceResult = const_cast<iODT*>(this)->traceIntrinsicHalfedgeAlongInput(intrinsicHalfedgeIdx);

    // Convert surface points to 3D positions
    for (const auto& sp : traceResult.pathPoints) {
        glm::dvec3 pos3D = tracerInput.evaluateSurfacePoint(sp);
        polyline.push_back(glm::vec3(pos3D));
    }

    return polyline;
}

void iODT::saveCommonSubdivisionOBJ(const std::string& filename, const Model& overlayModel) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[saveIntrinsicOverlayOBJ] Failed to open file: " << filename << std::endl;
        return;
    }

    // Extract vertices from the overlay model
    const auto& vertices = overlayModel.getVertices();
    const auto& indices = overlayModel.getIndices();

    // Write vertices
    for (const auto& vertex : vertices) {
        file << "v " << vertex.pos.x << " " << vertex.pos.y << " " << vertex.pos.z << "\n";
    }

    // Write vertex colors
    for (const auto& vertex : vertices) {
        file << "vc " << vertex.color.r << " " << vertex.color.g << " " << vertex.color.b << "\n";
    }

    // Write face elements 
    uint32_t faceCount = 0;
    for (size_t i = 0; i < indices.size(); i += 3) {
        if (i + 2 < indices.size()) {
            // OBJ uses 1-based indexing
            file << "f " << (indices[i] + 1) << " " << (indices[i + 1] + 1) << " " << (indices[i + 2] + 1) << "\n";
            faceCount++;
        }
    }

    file.close();
}

void iODT::createCommonSubdivision(Model& overlayModel, std::vector<CommonSubdivision::IntrinsicTriangle>& outIntrinsicTriangles) {
    commonSubdivision = std::make_unique<CommonSubdivision>(
        intrinsicMesh,
        inputMesh,
        tracerInput,
        intrinsicVertexLocations,
        *this
    );
    
    commonSubdivision->build();  
    outIntrinsicTriangles = commonSubdivision->getIntrinsicTriangles();  
    commonSubdivision->exportToModel(overlayModel);
}