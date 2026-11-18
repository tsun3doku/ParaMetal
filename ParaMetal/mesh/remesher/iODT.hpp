#pragma once
#include <glm/glm.hpp>
#include <queue>
#include <deque>
#include <functional>
#include "GeodesicTracer.hpp"
#include "SignPostMesh.hpp"
#include "util/Structs.hpp"
#include "CommonSubdivision.hpp"
#include "SupportingHalfedge.hpp"

class iODT {
public:
    iODT(const std::vector<float>& pointPositions, const std::vector<uint32_t>& triangleIndices);
    ~iODT();
    
    static const uint32_t INVALID_INDEX = static_cast<uint32_t>(-1);

    bool optimalDelaunayTriangulation(int iterations, double minAngleDegrees, double maxEdgeLength, double stepSize);

    enum class RefinementStatus {
        Complete,
        Partial
    };

    GeodesicTracer::GeodesicTraceResult traceIntrinsicHalfedgeAlongInput(uint32_t intrinsicHalfedgeIdx);

    void createCommonSubdivision(Model& overlayModel, std::vector<CommonSubdivision::IntrinsicTriangle>& outIntrinsicTriangles);
    void saveCommonSubdivisionOBJ(const std::string& filename, const Model& overlayModel) const;

    std::vector<glm::vec3> getCommonSubdivision(uint32_t intrinsicHalfedgeIdx) const;
    const CommonSubdivision* getCommonSubdivisionObject() const { return commonSubdivision.get(); }
    
    SupportingHalfedge* getSupportingHalfedge() { return supportingHalfedge.get(); }
    const SupportingHalfedge* getSupportingHalfedge() const { return supportingHalfedge.get(); }

private:
    void refreshIntrinsicDirectionalData();
    void optimalReposition(int iterations, double tol, double maxEdgeLength, double stepSize);
    int splitLongEdges(double maxEdgeLength, int maxSplits);
    double repositionInsertedVertices(double stepSize, std::vector<uint32_t>& outMovedEdges);
    RefinementStatus delaunayRefinement(float minAngleDegrees);

    struct RefinementCallbacks {
        std::function<void(uint32_t)> onFlip;
        std::function<void(uint32_t)> onFace;
    };

    struct FaceCandidate {
        uint32_t faceIdx = INVALID_INDEX;
        float area = 0.0f;
        float priority = 0.0f;

        bool operator<(const FaceCandidate& other) const {
            return priority < other.priority;
        }
    };

    std::vector<uint32_t> collectLocalDelaunayEdges(HalfEdgeMesh& conn, uint32_t vertexIdx);
    void updateLocalEdges(HalfEdgeMesh& conn, const std::vector<uint32_t>& edgePatch);
    bool isValidFace(uint32_t faceIdx) const;
    bool isValidEdge(uint32_t edgeIdx) const;
    uint32_t getParentInputFace(uint32_t faceIdx) const;
    bool needsRefinement(uint32_t faceIdx, float minAngleThreshold);
    float refinementPriority(uint32_t faceIdx, float area);

    size_t countSharpCorners(SignpostMesh& mesh, const uint32_t* faceHEs);
    void queueRefineFace(uint32_t faceIdx, float minAngleThreshold, std::priority_queue<FaceCandidate>& faceQueue);
    void queueDelaunayEdge(uint32_t edgeIdx, std::deque<uint32_t>& edgeQueue, std::vector<uint8_t>& inQueue);

    bool insertPoint(uint32_t faceIdx, const glm::dvec3& baryCoords, uint32_t& outVertex, bool* outWasInserted = nullptr, const RefinementCallbacks* callbacks = nullptr);
    bool insertCircumcenter(uint32_t faceIdx, uint32_t& outNewVertex, const RefinementCallbacks* callbacks = nullptr);
    bool splitEdge(uint32_t edgeIdx, uint32_t& outNewVertex, uint32_t& outDiagFront, uint32_t& outDiagBack, uint32_t HESplit, double t = 0.5, const RefinementCallbacks* callbacks = nullptr);
    bool splitEdge(uint32_t heEdge, double tParam, uint32_t& outNewV, bool* outWasInserted = nullptr, const RefinementCallbacks* callbacks = nullptr);

    uint32_t removeInsertedVertex(uint32_t vertexIdx, const RefinementCallbacks* callbacks = nullptr);
    void collectFlipNeighborhood(uint32_t edgeIdx, std::vector<uint32_t>& outFaces, std::vector<uint32_t>& outEdges) const;
    void collectVerticesWithinBall(uint32_t startVertex, double radius, std::vector<uint32_t>& outVertices) const;
    void deleteInsertedVerticesNear(uint32_t insertedVertex, double ballRadius, const RefinementCallbacks* callbacks = nullptr);

    bool computeWeightedCircumcenter(const HalfEdgeMesh::VertexRing2D& ring, glm::dvec2& outAvgVec, double& outAvgLen);
    bool resolveVertex(uint32_t vertexIdx);
    bool chooseTraceStart(const uint32_t* candidates, size_t count, uint32_t& outStart) const;
    bool traceFromLocation(uint32_t startVertex, const GeodesicTracer::SurfacePoint& start,
                           const glm::dvec2& dir2D, double length,
                           GeodesicTracer::GeodesicTraceResult& outTrace);
    void storeVertexLocation(uint32_t vertexIdx, const GeodesicTracer::GeodesicTraceResult& trace);
    void anchorFanToArrival(uint32_t vertexIdx, uint32_t anchorHE, const glm::dvec2& outgoingVec);

    bool isEdgeOriginal(uint32_t edgeIdx) const;

    size_t countInvalidIntrinsicFaces() const;

    void initializeVertexLocations();
    SignpostMesh intrinsicMesh;     // Intrinsic mesh 
    SignpostMesh inputMesh;         // Input mesh 
    GeodesicTracer tracer;          // Tracer for the intrinsic mesh
    GeodesicTracer tracerInput;     // Tracer for the input mesh

    std::unordered_set<uint32_t> insertedVertices;
    
    // Maps every intrinsic vertex to a location on the original mesh surface
    std::unordered_map<uint32_t, GeodesicTracer::SurfacePoint> intrinsicVertexLocations;

    // Tracks which input face was used during vertex resolution
    std::unordered_map<uint32_t, uint32_t> vertexResolutionFaces;

    std::unique_ptr<CommonSubdivision> commonSubdivision;   
    std::unique_ptr<SupportingHalfedge> supportingHalfedge; 
};


