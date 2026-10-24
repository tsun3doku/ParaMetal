#pragma once

#include "VEMGeometry.hpp"

#include <vector>
#include <cstdint>

namespace vem {

enum class BoundaryType : uint8_t {
    Dirichlet,
    Neumann,
    Robin
};

struct BoundaryCondition {
    BoundaryType type = BoundaryType::Neumann;
    float value = 0.0f;
    float robinCoefficient = 0.0f;
    float referenceValue = 0.0f;
};

struct TraceCoupling {
    uint32_t traceA = 0;
    uint32_t traceB = 0;
    float area = 0.0f;
    float coefficientPerArea = 0.0f;
};

struct DirichletTrace {
    uint32_t traceIndex = 0;
    float value = 0.0f;
};

struct CellValidation {
    bool passed = false;
    float volume = 0.0f;
    float surfaceArea = 0.0f;
    float detNNT = 0.0f;
    float conditionNNT = 0.0f;
    float minEigenNNT = 0.0f;
    float maxEigenNNT = 0.0f;
    float gaussConsistencyError = 0.0f;
    float closureError = 0.0f;
    bool choleskySucceeded = false;
    int failedPivot = -1;
    double failedPivotValue = 0.0;
};

struct VEMTraceSystem {
    uint32_t traceCount = 0;
    uint32_t cellCount = 0;

    std::vector<int> rowOffsets;
    std::vector<int> columnIndices;
    std::vector<float> values;

    std::vector<float> cellRecoveryScale;
    std::vector<uint32_t> cellTraceOffsets;
    std::vector<uint32_t> cellTraceIndices;
    std::vector<float> cellTraceWeights;

    std::vector<uint32_t> traceScatterOffsets;
    std::vector<uint32_t> traceScatterCells;
    std::vector<float> traceScatterWeights;
    std::vector<float> traceBoundaryRhs;

    std::vector<uint8_t> traceIsDirichlet;
    std::vector<float> traceDirichletValue;
};

class VEMDiffusion {
public:
    static bool validateElementGeometry(
        const std::vector<FaceGeometry>& faceIncidences,
        const CellGeometry& cell,
        double conductivity,
        CellValidation& outValidation);

    static bool assembleTraceSystem(
        const std::vector<FaceGeometry>& faceIncidences,
        const std::vector<CellGeometry>& cells,
        const std::vector<float>& cellDiffusionCoefficient,
        const std::vector<float>& cellStorageOverDt,
        uint32_t totalTraceCount,
        const BoundaryCondition* boundaries,
        uint32_t boundaryCount,
        const std::vector<TraceCoupling>& traceCouplings,
        const std::vector<DirichletTrace>& dirichletTraces,
        VEMTraceSystem& outSystem);

private:
    struct CellFaceRef {
        uint32_t cellIndex = 0;
        uint16_t localFaceIndex = 0;
    };
    struct CouplingTerm {
        uint32_t coupledTrace = 0;
        float conductance = 0.0f;
    };
    struct RobinTerm {
        float coefficient = 0.0f;
    };
    struct DirichletConstraint {
        bool isDirichlet = false;
        float value = 0.0f;
    };
    struct RowTerm {
        uint32_t column = 0;
        float value = 0.0f;
    };

    static bool computeElementFluxOperator(
        const std::vector<FaceGeometry>& faceIncidences,
        const CellGeometry& cell,
        double conductivity,
        std::vector<double>& W_E,
        CellValidation& outValidation);
    static bool invert3x3(const double A[3][3], double invA[3][3]);
    static bool choleskyInvert(
        const std::vector<double>& matrix,
        size_t n,
        std::vector<double>& inverse,
        int* outFailedPivot,
        double* outFailedPivotValue);
    static bool rowTermLess(const RowTerm& a, const RowTerm& b);
    static int buildRowEntries(
        uint32_t trace,
        uint32_t totalTraceCount,
        const std::vector<uint32_t>& traceIncidenceOffsets,
        const std::vector<CellFaceRef>& incidences,
        const std::vector<FaceGeometry>& faceIncidences,
        const std::vector<CellGeometry>& cells,
        const std::vector<float>& cellMatrices,
        const std::vector<size_t>& cellMatrixOffsets,
        const std::vector<std::vector<CouplingTerm>>& couplings,
        const std::vector<RobinTerm>& robinTerms,
        RowTerm* stackEntries,
        int stackCapacity,
        std::vector<RowTerm>& heapEntries,
        RowTerm*& outEntries);
};

} // namespace vem
