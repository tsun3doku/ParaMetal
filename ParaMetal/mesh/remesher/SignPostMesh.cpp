#include <glm/gtc/constants.hpp>

#include <algorithm>

#include "SignPostMesh.hpp"

void SignpostMesh::buildSignposts(const std::vector<float>& pointPositions, const std::vector<uint32_t>& triangleIndices) {
    conn.buildMesh(pointPositions, triangleIndices);
    updateAllCornerAngles(std::unordered_set<uint32_t>());
}

SignpostMesh::Triangle2D SignpostMesh::layoutTriangle(uint32_t faceIdx) const {
    return conn.layoutTriangle(faceIdx);
}

void SignpostMesh::updateAllSignposts() {
    const auto& V = conn.getVertices();
    rebuildVertexSums();

    // Walk around vertices assigning angular directions 
    for (uint32_t vid = 0; vid < V.size(); ++vid) {
        uint32_t firstOutgoing = V[vid].halfEdgeIdx;  
        if (firstOutgoing == HalfEdgeMesh::INVALID_INDEX)
            continue;

        double runningAngle = 0.0;
        uint32_t currHe = firstOutgoing;
        int safety = 0;
        do {
            // Store signpost angle at origin vertex 
            conn.getHalfEdges()[currHe].signpostAngle = runningAngle;

            // Add corner angle for next iteration 
            double cornerAngleVal = conn.getHalfEdges()[currHe].cornerAngle;
            runningAngle += cornerAngleVal;

            // Break at boundary 
            if (conn.getHalfEdges()[currHe].opposite == HalfEdgeMesh::INVALID_INDEX) {
                break;
            }

            uint32_t nextHe = conn.getNextAroundVertex(currHe);
            if (nextHe == HalfEdgeMesh::INVALID_INDEX)
                break;
            currHe = nextHe;
            
            if (++safety > 10000)
                break;
        } while (currHe != firstOutgoing);
    }
}

void SignpostMesh::rebuildVertexSums() {
    const auto& V = conn.getVertices();
    const auto& HEs = conn.getHalfEdges();

    vertexAngleSums.clear();
    vertexAngleSums.resize(V.size(), 0.0);

    for (uint32_t heIdx = 0; heIdx < HEs.size(); ++heIdx) {
        uint32_t vertexIdx = HEs[heIdx].origin;
        if (vertexIdx < vertexAngleSums.size()) {
            vertexAngleSums[vertexIdx] += HEs[heIdx].cornerAngle;
        }
    }
}

void SignpostMesh::updateVertexSums(const std::vector<uint32_t>& vertexIndices) {
    const auto& V = conn.getVertices();
    const auto& HEs = conn.getHalfEdges();

    if (vertexAngleSums.size() != V.size()) {
        vertexAngleSums.resize(V.size(), 0.0);
    }

    std::unordered_set<uint32_t> uniqueVertices;
    for (uint32_t vertexIdx : vertexIndices) {
        if (vertexIdx < V.size()) {
            uniqueVertices.insert(vertexIdx);
        }
    }

    for (uint32_t vertexIdx : uniqueVertices) {
        vertexAngleSums[vertexIdx] = 0.0;
        for (uint32_t heIdx : conn.getVertexHalfEdges(vertexIdx)) {
            if (heIdx < HEs.size()) {
                vertexAngleSums[vertexIdx] += HEs[heIdx].cornerAngle;
            }
        }
    }
}

glm::dvec2 SignpostMesh::halfedgeVector(uint32_t heIdx) const {
    const auto& HEs = conn.getHalfEdges();
    if (heIdx >= HEs.size()) 
        return glm::dvec2(0.0);

    const auto& he = HEs[heIdx];

    // Get raw signpost angle and apply vertex angle scaling
    double rawAngle = he.signpostAngle;
    // Use origin vertex 
    uint32_t originVertexIdx = he.origin;
    
    // Apply vertex angle scaling: scaleFac = target_angle_sum / intrinsic_angle_sum
    double scaleFac = 1.0;
    if (originVertexIdx < vertexAngleScales.size()) {
        scaleFac = vertexAngleScales[originVertexIdx];
    }
    
    double scaledAngle = rawAngle * scaleFac;
    double length = conn.getIntrinsicLengthFromHalfEdge(heIdx);
    return glm::dvec2(std::cos(scaledAngle) * length, std::sin(scaledAngle) * length);
}

void SignpostMesh::buildHalfedgeVectorsInVertex() {
    auto& HEs = conn.getHalfEdges();
    const auto& V = conn.getVertices();
    
    halfedgeVectorsInVertex.clear();
    halfedgeVectorsInVertex.resize(HEs.size(), glm::dvec2(std::numeric_limits<double>::quiet_NaN()));

    for (uint32_t vid = 0; vid < V.size(); ++vid) {
        uint32_t firstOutgoing = V[vid].halfEdgeIdx;  
        if (firstOutgoing == HalfEdgeMesh::INVALID_INDEX) 
            continue;

        uint32_t currHe = firstOutgoing;
        int safety = 0;
        
        do {
            halfedgeVectorsInVertex[currHe] = halfedgeVector(currHe);
            
            // Break at boundary 
            if (HEs[currHe].opposite == HalfEdgeMesh::INVALID_INDEX) {
                break;
            }
            
            uint32_t nextHe = conn.getNextAroundVertex(currHe);
            if (nextHe == HalfEdgeMesh::INVALID_INDEX)
                break;
            currHe = nextHe;
            
            if (++safety > 2000) 
                break;
        } while (currHe != firstOutgoing);
    }
}

void SignpostMesh::updateVertexVectors(const std::vector<uint32_t>& vertexIndices) {
    auto& HEs = conn.getHalfEdges();
    const auto& V = conn.getVertices();

    if (halfedgeVectorsInVertex.size() != HEs.size()) {
        halfedgeVectorsInVertex.resize(HEs.size(), glm::dvec2(std::numeric_limits<double>::quiet_NaN()));
    }

    std::unordered_set<uint32_t> uniqueVertices;
    for (uint32_t vertexIdx : vertexIndices) {
        if (vertexIdx < V.size()) {
            uniqueVertices.insert(vertexIdx);
        }
    }

    for (uint32_t vid : uniqueVertices) {
        uint32_t firstOutgoing = V[vid].halfEdgeIdx;
        if (firstOutgoing == HalfEdgeMesh::INVALID_INDEX) {
            continue;
        }

        uint32_t currHe = firstOutgoing;
        int safety = 0;
        do {
            halfedgeVectorsInVertex[currHe] = halfedgeVector(currHe);

            if (HEs[currHe].opposite == HalfEdgeMesh::INVALID_INDEX) {
                break;
            }

            uint32_t nextHe = conn.getNextAroundVertex(currHe);
            if (nextHe == HalfEdgeMesh::INVALID_INDEX) {
                break;
            }
            currHe = nextHe;

            if (++safety > 2000) {
                break;
            }
        } while (currHe != firstOutgoing);
    }
}

void SignpostMesh::buildHalfedgeVectorsInFace() {
    const auto& HEs = conn.getHalfEdges();
    halfedgeVectorsInFace.clear();
    halfedgeVectorsInFace.resize(HEs.size(), glm::dvec2(0.0));

    const uint32_t faceCount = conn.getFaces().size();
    for (uint32_t f = 0; f < faceCount; ++f) {
        auto faceHEs = conn.getFaceHalfEdges(f);
        if (faceHEs.size() != 3) {
            continue;
        }

        // Use the canonical triangle layout in this face
        Triangle2D tri = layoutTriangle(f);

        // faceHEs[i] has origin = tri.indices[i], and tri.vertices[i] is position of that origin
        for (int i = 0; i < 3; ++i) {
            int ni = (i + 1) % 3;
            glm::dvec2 edgeVec = tri.vertices[ni] - tri.vertices[i]; // V_next - V_curr
            halfedgeVectorsInFace[faceHEs[i]] = edgeVec;
        }
    }

    // Mark exterior / invalid halfedges 
    const auto& halfEdgesRef = conn.getHalfEdges();
    for (size_t i = 0; i < halfEdgesRef.size(); ++i) {
        if (halfEdgesRef[i].face == HalfEdgeMesh::INVALID_INDEX) {
            halfedgeVectorsInFace[i] = glm::dvec2(std::numeric_limits<double>::quiet_NaN());
        }
    }
}

void SignpostMesh::updateFaceVectors(const std::vector<uint32_t>& faceIndices) {
    const auto& HEs = conn.getHalfEdges();
    const auto& faces = conn.getFaces();

    if (halfedgeVectorsInFace.size() != HEs.size()) {
        halfedgeVectorsInFace.resize(HEs.size(), glm::dvec2(0.0));
    }

    std::unordered_set<uint32_t> uniqueFaces;
    for (uint32_t faceIdx : faceIndices) {
        if (faceIdx < faces.size() && faces[faceIdx].halfEdgeIdx != HalfEdgeMesh::INVALID_INDEX) {
            uniqueFaces.insert(faceIdx);
        }
    }

    for (uint32_t f : uniqueFaces) {
        auto faceHEs = conn.getFaceHalfEdges(f);
        if (faceHEs.size() != 3) {
            continue;
        }

        Triangle2D tri = layoutTriangle(f);
        for (int i = 0; i < 3; ++i) {
            int ni = (i + 1) % 3;
            halfedgeVectorsInFace[faceHEs[i]] = tri.vertices[ni] - tri.vertices[i];
        }
    }
}

glm::dvec2 SignpostMesh::computeCircumcenter2D(const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c) const {
    double d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

    if (std::abs(d) < 1e-12) {
        return glm::dvec2(std::numeric_limits<double>::quiet_NaN());
    }

    double aSq = a.x * a.x + a.y * a.y;
    double bSq = b.x * b.x + b.y * b.y;
    double cSq = c.x * c.x + c.y * c.y;

    double x = (aSq * (b.y - c.y) + bSq * (c.y - a.y) + cSq * (a.y - b.y)) / d;
    double y = (aSq * (c.x - b.x) + bSq * (a.x - c.x) + cSq * (b.x - a.x)) / d;

    return glm::dvec2(x, y);
}

glm::dvec3 SignpostMesh::computeBarycentric2D(const glm::dvec2& p, const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c) const {
    glm::dvec2 v0 = c - a;
    glm::dvec2 v1 = b - a;
    glm::dvec2 v2 = p - a;

    double dot00 = glm::dot(v0, v0);
    double dot01 = glm::dot(v0, v1);
    double dot02 = glm::dot(v0, v2);
    double dot11 = glm::dot(v1, v1);
    double dot12 = glm::dot(v1, v2);

    double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    double w = 1.0 - u - v;
    return glm::dvec3(w, v, u);
}

double SignpostMesh::computeSplitDiagonalLength(uint32_t faceIdx, uint32_t originalVA, uint32_t originalVB, double splitFraction) const {
    // Lay out the triangle in 2D using intrinsic edge lengths
    auto triangle2D = layoutTriangle(faceIdx);

    // Find which indices belong to the original edge's endpoints
    int idxA = -1, idxB = -1, idxC = -1;
    for (int i = 0; i < 3; ++i) {
        if (triangle2D.indices[i] == INVALID_INDEX) {
            return 0.0;
        }
        if (triangle2D.indices[i] == originalVA)
            idxA = i;
        else if (triangle2D.indices[i] == originalVB)
            idxB = i;
        else
            idxC = i;
    }

    if (idxA < 0 || idxB < 0 || idxC < 0) {
        return 0.0;
    }

    // Calculate split point P
    glm::dvec2 A = triangle2D.vertices[idxA];
    glm::dvec2 B = triangle2D.vertices[idxB];
    glm::dvec2 P = (1.0 - splitFraction) * A + splitFraction * B;

    // Distance from P to the third corner
    glm::dvec2 C = triangle2D.vertices[idxC];
    double result = glm::length(P - C);

    return result;
}

double SignpostMesh::computeAngleFromLengths(double a, double b, double c, uint32_t faceIdx) const {
    const float MIN_LENGTH = 1e-6f;

    // Check if any edge is too small or triangle inequality is violated
    if (a < MIN_LENGTH || b < MIN_LENGTH || c < MIN_LENGTH ||
        a + b < c || a + c < b || b + c < a) {
        return -1.0f;
    }

    double cosAngle = (b * b + c * c - a * a) / (2.0 * b * c);
    cosAngle = std::max(-1.0, std::min(1.0, cosAngle));

    return std::acos(cosAngle);
}

std::array<double, 3> SignpostMesh::computeCornerAngles(uint32_t faceIdx) const {
    std::array<double, 3> angles = { 0.0, 0.0, 0.0 };

    const auto& faces = conn.getFaces();
    const auto& HEs = conn.getHalfEdges();

    if (faceIdx >= faces.size() || faces[faceIdx].halfEdgeIdx == INVALID_INDEX) {
        return angles;
    }

    const uint32_t he0 = faces[faceIdx].halfEdgeIdx;
    if (he0 == INVALID_INDEX || he0 >= HEs.size()) {
        return angles;
    }

    const uint32_t he1 = HEs[he0].next;
    if (he1 == INVALID_INDEX || he1 >= HEs.size()) {
        return angles;
    }

    const uint32_t he2 = HEs[he1].next;
    if (he2 == INVALID_INDEX || he2 >= HEs.size()) {
        return angles;
    }

    const double a = conn.getIntrinsicLengthFromHalfEdge(he0);
    const double b = conn.getIntrinsicLengthFromHalfEdge(he1);
    const double c = conn.getIntrinsicLengthFromHalfEdge(he2);

    angles[0] = std::max(0.0, computeAngleFromLengths(b, c, a, faceIdx));
    angles[1] = std::max(0.0, computeAngleFromLengths(c, a, b, faceIdx));
    angles[2] = std::max(0.0, computeAngleFromLengths(a, b, c, faceIdx));
    return angles;
}

void SignpostMesh::updateCornerAnglesForFace(uint32_t faceIdx) {
    const auto& faces = conn.getFaces();
    auto& HEs = conn.getHalfEdges();

    if (faceIdx >= faces.size() || faces[faceIdx].halfEdgeIdx == INVALID_INDEX)
        return;

    // Get the three halfedges of this face
    uint32_t he0 = faces[faceIdx].halfEdgeIdx;
    uint32_t he1 = HEs[he0].next;
    uint32_t he2 = HEs[he1].next;

    if (he0 == INVALID_INDEX || he1 == INVALID_INDEX || he2 == INVALID_INDEX)
        return;

    const auto angles = computeCornerAngles(faceIdx);

    // Each halfedge gets the angle at its origin vertex
    HEs[he0].cornerAngle = angles[0];  // Angle at origin of he0
    HEs[he1].cornerAngle = angles[1];  // Angle at origin of he1
    HEs[he2].cornerAngle = angles[2];  // Angle at origin of he2
}

void SignpostMesh::updateAllCornerAngles(const std::unordered_set<uint32_t>& skipFaces) {
    const auto& faces = conn.getFaces();
    for (uint32_t f = 0; f < faces.size(); ++f) {
        if (skipFaces.count(f))
            continue;
        updateCornerAnglesForFace(f);
    }
}

double SignpostMesh::standardizeAngleForVertex(uint32_t vertexIdx, double angleRad) const {
    // If boundary vertex, return angle unchanged
    if (conn.isBoundaryVertex(vertexIdx)) {
        return angleRad;
    }
    double sum = getVertexAngleSum(vertexIdx); 
    // wrap into [0, sum)
    double wrapped = std::fmod(angleRad, sum);
    if (wrapped < 0.0) wrapped += sum; 
    return wrapped;
}

void SignpostMesh::updateAngleFromCWNeighbor(uint32_t heIdx) {
    auto& HEs = conn.getHalfEdges();
    
    // Boundary handling 
    if (!conn.isInteriorHalfEdge(heIdx)) {
        // Last wedge angle
        HEs[heIdx].signpostAngle = getVertexAngleSum(HEs[heIdx].origin); 
        halfedgeVectorsInVertex[heIdx] = halfedgeVector(heIdx);
        return;
    }

    uint32_t twin = HEs[heIdx].opposite;
    if (!conn.isInteriorHalfEdge(twin)) {
        HEs[heIdx].signpostAngle = 0.0; // first wedge
        halfedgeVectorsInVertex[heIdx] = halfedgeVector(heIdx);
        return;
    }

    // CW neighbor = he.opp().next()
    uint32_t cw = HEs[twin].next; 
    // Neighbor stored raw angle
    double neighborRaw = HEs[cw].signpostAngle;
    // Corner angle of neighbor
    double cAngle = HEs[cw].cornerAngle;

    // Add neighbor raw angle and corner angle then convert into target vertex domain
    double updated = standardizeAngleForVertex(HEs[heIdx].origin, neighborRaw + cAngle);
    HEs[heIdx].signpostAngle = updated;
    halfedgeVectorsInVertex[heIdx] = halfedgeVector(heIdx);
}

float SignpostMesh::computeFaceArea(uint32_t faceIdx) const {
    const auto& faces = conn.getFaces();
    const auto& halfEdges = conn.getHalfEdges();

    if (faceIdx >= faces.size() || faces[faceIdx].halfEdgeIdx == HalfEdgeMesh::INVALID_INDEX) {
        return 0.0f;
    }

    uint32_t he0 = faces[faceIdx].halfEdgeIdx;
    if (he0 >= halfEdges.size()) {
        return 0.0f;
    }
    uint32_t he1 = halfEdges[he0].next;
    if (he1 >= halfEdges.size()) {
        return 0.0f;
    }
    uint32_t he2 = halfEdges[he1].next;
    if (he2 >= halfEdges.size() || halfEdges[he2].next != he0) {
        return 0.0f;
    }

    // Get the edge lengths from edge objects
    double a = conn.getIntrinsicLengthFromHalfEdge(he0);
    double b = conn.getIntrinsicLengthFromHalfEdge(he1);
    double c = conn.getIntrinsicLengthFromHalfEdge(he2);

    // Ensure minimum positive length for numerical stability
    a = std::max(a, 1e-12);
    b = std::max(b, 1e-12);
    c = std::max(c, 1e-12);

    // Calculate semiperimeter
    double s = (a + b + c) / 2.0f;

    // Calculate area using Heron's formula
    double area = std::sqrt(std::max(0.0, s * (s - a) * (s - b) * (s - c)));

    return static_cast<float>(area);
}

std::vector<float> SignpostMesh::getAllFaceAreas() const {
    std::vector<float> areas;
    areas.reserve(conn.getFaces().size());

    for (uint32_t i = 0; i < conn.getFaces().size(); ++i) {
        areas.push_back(computeFaceArea(i));
    }

    return areas;
}

bool SignpostMesh::isEdgeOnBoundary(uint32_t heIdx) const {
    const auto& HEs = conn.getHalfEdges();
    return (heIdx < HEs.size() && HEs[heIdx].opposite == INVALID_INDEX);
}

std::vector<uint32_t> SignpostMesh::getBoundaryVertices() const {
    std::vector<uint32_t> boundaryVerts;

    for (uint32_t i = 0; i < conn.getVertices().size(); ++i) {
        if (conn.isBoundaryVertex(i)) {
            boundaryVerts.push_back(i);
        }
    }

    return boundaryVerts;
}

uint32_t SignpostMesh::getVertexDegree(uint32_t vertexIdx) const {
    if (vertexIdx >= conn.getVertices().size()) {
        return 0;
    }

    std::vector<uint32_t> vertexHEs = conn.getVertexHalfEdges(vertexIdx);
    return static_cast<uint32_t>(vertexHEs.size());
}

double SignpostMesh::getCornerAngle(uint32_t halfEdgeIdx) const {
    const auto& HEs = conn.getHalfEdges();
    return (halfEdgeIdx < HEs.size())
        ? HEs[halfEdgeIdx].cornerAngle
        : 0.0;
}

void SignpostMesh::computeVertexAngleScales() {
    const auto& V = conn.getVertices();   

    vertexAngleScales.clear();
    vertexAngleScales.resize(V.size(), 1.0);

    for (uint32_t vid = 0; vid < V.size(); ++vid) {
        double intrinsicSum = 0.0;
        if (vid < vertexAngleSums.size()) intrinsicSum = vertexAngleSums[vid];

        double target = conn.isBoundaryVertex(vid) ? glm::pi<double>() : 2.0 * glm::pi<double>();

        if (intrinsicSum > 1e-12) {
            vertexAngleScales[vid] = target / intrinsicSum;
        }
        else {
            vertexAngleScales[vid] = 1.0; 
        }
    }
}

void SignpostMesh::updateVertexScales(const std::vector<uint32_t>& vertexIndices) {
    const auto& V = conn.getVertices();

    if (vertexAngleScales.size() != V.size()) {
        vertexAngleScales.resize(V.size(), 1.0);
    }

    std::unordered_set<uint32_t> uniqueVertices;
    for (uint32_t vertexIdx : vertexIndices) {
        if (vertexIdx < V.size()) {
            uniqueVertices.insert(vertexIdx);
        }
    }

    for (uint32_t vid : uniqueVertices) {
        double intrinsicSum = 0.0;
        if (vid < vertexAngleSums.size()) {
            intrinsicSum = vertexAngleSums[vid];
        }

        double target = conn.isBoundaryVertex(vid) ? glm::pi<double>() : 2.0 * glm::pi<double>();
        vertexAngleScales[vid] = (intrinsicSum > 1e-12) ? (target / intrinsicSum) : 1.0;
    }
}
