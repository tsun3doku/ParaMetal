#include "VEMDiffusion.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <atomic>

namespace vem {

bool VEMDiffusion::invert3x3(const double A[3][3], double invA[3][3]) {
    double scale = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            if (!std::isfinite(A[row][column])) {
                return false;
            }
            scale = std::max(scale, std::abs(A[row][column]));
        }
    }
    if (!(scale > 0.0)) {
        return false;
    }

    double normalized[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            normalized[row][column] = A[row][column] / scale;
        }
    }

    const double det = normalized[0][0] * (normalized[1][1] * normalized[2][2] - normalized[1][2] * normalized[2][1]) -
                       normalized[0][1] * (normalized[1][0] * normalized[2][2] - normalized[1][2] * normalized[2][0]) +
                       normalized[0][2] * (normalized[1][0] * normalized[2][1] - normalized[1][1] * normalized[2][0]);
    if (std::abs(det) <= 64.0 * std::numeric_limits<double>::epsilon() || !std::isfinite(det)) {
        return false;
    }

    const double inverseScale = 1.0 / (scale * det);
    invA[0][0] = (normalized[1][1] * normalized[2][2] - normalized[1][2] * normalized[2][1]) * inverseScale;
    invA[0][1] = (normalized[0][2] * normalized[2][1] - normalized[0][1] * normalized[2][2]) * inverseScale;
    invA[0][2] = (normalized[0][1] * normalized[1][2] - normalized[0][2] * normalized[1][1]) * inverseScale;
    invA[1][0] = (normalized[1][2] * normalized[2][0] - normalized[1][0] * normalized[2][2]) * inverseScale;
    invA[1][1] = (normalized[0][0] * normalized[2][2] - normalized[0][2] * normalized[2][0]) * inverseScale;
    invA[1][2] = (normalized[0][2] * normalized[1][0] - normalized[0][0] * normalized[1][2]) * inverseScale;
    invA[2][0] = (normalized[1][0] * normalized[2][1] - normalized[1][1] * normalized[2][0]) * inverseScale;
    invA[2][1] = (normalized[0][1] * normalized[2][0] - normalized[0][0] * normalized[2][1]) * inverseScale;
    invA[2][2] = (normalized[0][0] * normalized[1][1] - normalized[0][1] * normalized[1][0]) * inverseScale;

    return true;
}

bool VEMDiffusion::choleskyInvert(
    const std::vector<double>& matrix,
    size_t n,
    std::vector<double>& inverse,
    int* outFailedPivot,
    double* outFailedPivotValue) {
    std::vector<double> L(n * n, 0.0);
    double matrixScale = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double diagonal = matrix[i * n + i];
        if (!std::isfinite(diagonal)) {
            return false;
        }
        matrixScale = std::max(matrixScale, std::abs(diagonal));
    }
    if (!(matrixScale > 0.0)) {
        return false;
    }
    const double pivotTolerance = 64.0 * std::numeric_limits<double>::epsilon() * matrixScale;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = matrix[i * n + j];
            for (size_t k = 0; k < j; ++k) {
                sum -= L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                if (sum <= pivotTolerance || !std::isfinite(sum)) {
                    if (outFailedPivot) *outFailedPivot = static_cast<int>(i);
                    if (outFailedPivotValue) *outFailedPivotValue = sum;
                    return false;
                }
                L[i * n + j] = std::sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }

    std::vector<double> invL(n * n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        invL[j * n + j] = 1.0 / L[j * n + j];
        for (size_t i = j + 1; i < n; ++i) {
            double sum = 0.0;
            for (size_t k = j; k < i; ++k) {
                sum += L[i * n + k] * invL[k * n + j];
            }
            invL[i * n + j] = -sum / L[i * n + i];
        }
    }

    inverse.assign(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t startK = std::max(i, j);
            for (size_t k = startK; k < n; ++k) {
                sum += invL[k * n + i] * invL[k * n + j];
            }
            inverse[i * n + j] = sum;
        }
    }

    return true;
}

bool VEMDiffusion::computeElementFluxOperator(
    const std::vector<FaceGeometry>& faceIncidences,
    const CellGeometry& cell,
    double conductivity,
    std::vector<double>& W_E,
    CellValidation& outValidation) {
    outValidation = CellValidation{};
    outValidation.volume = static_cast<float>(cell.volume);
    const size_t nF = cell.faceCount;
    if (nF < 4 || !std::isfinite(cell.volume) || cell.volume <= 0.0 ||
        !std::isfinite(conductivity) || conductivity <= 0.0) {
        return false;
    }

    const double V_E = cell.volume;
    const double h = std::cbrt(V_E);
    const double invH = 1.0 / h;
    const double invH2 = invH * invH;
    const glm::dvec3 x_E(cell.centroid);

    std::vector<double> R_hat(3 * nF);
    std::vector<double> N_hat(3 * nF);
    double rHatSqSum = 0.0;
    double totalArea = 0.0;
    double G_hat[3][3] = { {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} };
    double B_hat[3][3] = { {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} };
    glm::dvec3 normalSum(0.0);

    for (size_t f = 0; f < nF; ++f) {
        const FaceGeometry& face = faceIncidences[cell.faceOffset + f];
        const double A_f = face.area;
        const glm::dvec3 n_vec = face.areaVector;
        const glm::dvec3 m_vec = face.moment;
        if (!(A_f > 0.0) || !std::isfinite(A_f))
            return false;
        const glm::dvec3 x_f = m_vec / A_f;
        if (!std::isfinite(x_f.x) || !std::isfinite(x_f.y) || !std::isfinite(x_f.z))
            return false;
        const glm::dvec3 r_f = x_f - x_E;

        const glm::dvec3 r_h = r_f * invH;
        const glm::dvec3 n_h = n_vec * invH2;

        R_hat[0 * nF + f] = r_h.x;
        R_hat[1 * nF + f] = r_h.y;
        R_hat[2 * nF + f] = r_h.z;

        N_hat[0 * nF + f] = n_h.x;
        N_hat[1 * nF + f] = n_h.y;
        N_hat[2 * nF + f] = n_h.z;

        rHatSqSum += glm::dot(r_h, r_h);
        totalArea += A_f;
        normalSum += n_vec;

        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                G_hat[a][b] += r_h[a] * n_h[b];
                B_hat[a][b] += n_h[a] * n_h[b];
            }
        }
    }

    outValidation.surfaceArea = static_cast<float>(totalArea);

    double errFrob2 = 0.0;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            const double target = (a == b) ? 1.0 : 0.0;
            const double diff = G_hat[a][b] - target;
            errFrob2 += diff * diff;
        }
    }
    outValidation.gaussConsistencyError = static_cast<float>(std::sqrt(errFrob2));

    const double detB = B_hat[0][0] * (B_hat[1][1] * B_hat[2][2] - B_hat[1][2] * B_hat[2][1]) -
                        B_hat[0][1] * (B_hat[1][0] * B_hat[2][2] - B_hat[1][2] * B_hat[2][0]) +
                        B_hat[0][2] * (B_hat[1][0] * B_hat[2][1] - B_hat[1][1] * B_hat[2][0]);
    outValidation.detNNT = static_cast<float>(detB);

    const double trB = B_hat[0][0] + B_hat[1][1] + B_hat[2][2];
    const double q = trB / 3.0;
    const double p = ( (B_hat[0][0]-q)*(B_hat[0][0]-q) + (B_hat[1][1]-q)*(B_hat[1][1]-q) + (B_hat[2][2]-q)*(B_hat[2][2]-q) +
                       2.0 * (B_hat[0][1]*B_hat[0][1] + B_hat[0][2]*B_hat[0][2] + B_hat[1][2]*B_hat[1][2]) ) / 6.0;
    if (p < 1e-14) {
        outValidation.conditionNNT = 1.0f;
        outValidation.minEigenNNT = static_cast<float>(q);
        outValidation.maxEigenNNT = static_cast<float>(q);
    } else {
        const double sqrtP = std::sqrt(p);
        double Bnorm[3][3];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                Bnorm[r][c] = (B_hat[r][c] - (r == c ? q : 0.0)) / sqrtP;
            }
        }
        const double detBnorm = Bnorm[0][0] * (Bnorm[1][1] * Bnorm[2][2] - Bnorm[1][2] * Bnorm[2][1]) -
                                Bnorm[0][1] * (Bnorm[1][0] * Bnorm[2][2] - Bnorm[1][2] * Bnorm[2][0]) +
                                Bnorm[0][2] * (Bnorm[1][0] * Bnorm[2][1] - Bnorm[1][1] * Bnorm[2][0]);
        const double clamped = std::max(-1.0, std::min(1.0, detBnorm * 0.5));
        const double phi = std::acos(clamped) / 3.0;
        const double pi23 = 2.0 * 3.14159265358979323846 / 3.0;
        const double lam0 = q + 2.0 * sqrtP * std::cos(phi);
        const double lam1 = q + 2.0 * sqrtP * std::cos(phi + pi23);
        const double lam2 = q + 2.0 * sqrtP * std::cos(phi + 2.0 * pi23);

        const double lamMin = std::min(lam0, std::min(lam1, lam2));
        const double lamMax = std::max(lam0, std::max(lam1, lam2));

        outValidation.minEigenNNT = static_cast<float>(lamMin);
        outValidation.maxEigenNNT = static_cast<float>(lamMax);
        outValidation.conditionNNT = (lamMin > 1e-12) ? static_cast<float>(lamMax / lamMin) : 1e9f;
    }

    outValidation.closureError = (totalArea > 0.0 && std::isfinite(totalArea))
        ? static_cast<float>(glm::length(normalSum) / totalArea)
        : 0.0f;

    double invB[3][3];
    if (!invert3x3(B_hat, invB)) {
        return false;
    }

    std::vector<double> temp(3 * nF, 0.0);
    for (int a = 0; a < 3; ++a) {
        for (size_t f = 0; f < nF; ++f) {
            double sum = 0.0;
            for (int b = 0; b < 3; ++b) {
                sum += invB[a][b] * N_hat[b * nF + f];
            }
            temp[a * nF + f] = sum;
        }
    }

    const double invKh = 1.0 / (conductivity * h);
    const double alpha_E = rHatSqSum / (3.0 * conductivity * h);
    if (!(alpha_E > 1e-25) || !std::isfinite(alpha_E)) {
        return false;
    }

    const double rCoeff = invKh / alpha_E;
    std::vector<double> M_tilde(nF * nF, 0.0);
    for (size_t i = 0; i < nF; ++i) {
        for (size_t j = 0; j < nF; ++j) {
            const double dotR = R_hat[0 * nF + i] * R_hat[0 * nF + j] +
                                R_hat[1 * nF + i] * R_hat[1 * nF + j] +
                                R_hat[2 * nF + i] * R_hat[2 * nF + j];
            double projection = 0.0;
            for (int a = 0; a < 3; ++a) {
                projection += N_hat[a * nF + i] * temp[a * nF + j];
            }
            const double delta = (i == j) ? 1.0 : 0.0;
            M_tilde[i * nF + j] = rCoeff * dotR + delta - projection;
        }
    }

    W_E.resize(nF * nF);
    int failedPivot = -1;
    double failedPivotValue = 0.0;
    if (!choleskyInvert(M_tilde, nF, W_E, &failedPivot, &failedPivotValue)) {
        outValidation.failedPivot = failedPivot;
        outValidation.failedPivotValue = failedPivotValue;
        return false;
    }

    const double invAlpha = 1.0 / alpha_E;
    for (size_t i = 0; i < nF * nF; ++i) {
        W_E[i] *= invAlpha;
    }

    outValidation.choleskySucceeded = true;
    outValidation.passed = (outValidation.gaussConsistencyError < 0.02f) &&
                           (outValidation.closureError < 0.005f) &&
                           (outValidation.conditionNNT < 1e5f);
    return true;
}

bool VEMDiffusion::validateElementGeometry(
    const std::vector<FaceGeometry>& faceIncidences,
    const CellGeometry& cell,
    double conductivity,
    CellValidation& outValidation) {
    std::vector<double> fluxOperator;
    const bool operatorValid = computeElementFluxOperator(
        faceIncidences, cell, conductivity, fluxOperator, outValidation);
    return operatorValid && outValidation.passed;
}

bool VEMDiffusion::rowTermLess(const RowTerm& a, const RowTerm& b) {
    return a.column < b.column;
}

int VEMDiffusion::buildRowEntries(
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
    RowTerm*& outEntries) {
    size_t estimatedCount = couplings[trace].size() * 2 + 1;
    for (uint32_t inc = traceIncidenceOffsets[trace]; inc < traceIncidenceOffsets[trace + 1]; ++inc) {
        estimatedCount += cells[incidences[inc].cellIndex].faceCount;
    }

    RowTerm* entries = stackEntries;
    int capacity = stackCapacity;
    if (estimatedCount > static_cast<size_t>(stackCapacity)) {
        heapEntries.resize(estimatedCount);
        entries = heapEntries.data();
        capacity = static_cast<int>(estimatedCount);
    }

    int numEntries = 0;
    const uint32_t startInc = traceIncidenceOffsets[trace];
    const uint32_t endInc = traceIncidenceOffsets[trace + 1];
    for (uint32_t inc = startInc; inc < endInc; ++inc) {
        const CellFaceRef& incidence = incidences[inc];
        const CellGeometry& cell = cells[incidence.cellIndex];
        const size_t nF = cell.faceCount;
        const float* cellMatrix = &cellMatrices[cellMatrixOffsets[incidence.cellIndex]];
        const size_t row = incidence.localFaceIndex;
        for (size_t j = 0; j < nF; ++j) {
            const uint32_t column = faceIncidences[cell.faceOffset + j].traceIndex;
            if (column < totalTraceCount && numEntries < capacity) {
                entries[numEntries++] = { column, cellMatrix[row * nF + j] };
            }
        }
    }

    for (const CouplingTerm& coupling : couplings[trace]) {
        if (numEntries + 2 <= capacity) {
            entries[numEntries++] = { trace, coupling.conductance };
            entries[numEntries++] = { coupling.coupledTrace, -coupling.conductance };
        }
    }

    if (robinTerms[trace].coefficient > 0.0f && numEntries < capacity) {
        entries[numEntries++] = { trace, robinTerms[trace].coefficient };
    }

    std::sort(entries, entries + numEntries, rowTermLess);
    outEntries = entries;
    return numEntries;
}

bool VEMDiffusion::assembleTraceSystem(
    const std::vector<FaceGeometry>& faceIncidences,
    const std::vector<CellGeometry>& cells,
    const std::vector<float>& cellDiffusionCoefficient,
    const std::vector<float>& cellStorageOverDt,
    uint32_t totalTraceCount,
    const BoundaryCondition* boundaries,
    uint32_t boundaryCount,
    const std::vector<TraceCoupling>& traceCouplings,
    const std::vector<DirichletTrace>& dirichletTraces,
    VEMTraceSystem& outSystem) {
    if (totalTraceCount == 0 || cells.empty()) {
        return false;
    }

    const uint32_t numCells = static_cast<uint32_t>(cells.size());
    outSystem.traceCount = totalTraceCount;
    outSystem.cellCount = numCells;

    outSystem.cellRecoveryScale.resize(numCells);
    outSystem.cellTraceOffsets.resize(numCells + 1, 0);

    size_t totalIncidentFaceCount = 0;
    std::vector<size_t> cellKOffsets(numCells + 1, 0);
    size_t totalKEntries = 0;

    for (uint32_t c = 0; c < numCells; ++c) {
        const size_t nF = cells[c].faceCount;
        outSystem.cellTraceOffsets[c] = static_cast<uint32_t>(totalIncidentFaceCount);
        totalIncidentFaceCount += nF;

        cellKOffsets[c] = totalKEntries;
        totalKEntries += nF * nF;
    }
    outSystem.cellTraceOffsets[numCells] = static_cast<uint32_t>(totalIncidentFaceCount);
    cellKOffsets[numCells] = totalKEntries;

    outSystem.cellTraceIndices.resize(totalIncidentFaceCount);
    outSystem.cellTraceWeights.resize(totalIncidentFaceCount);

    std::vector<float> allK(totalKEntries, 0.0f);

    std::vector<uint32_t> traceCellCount(totalTraceCount, 0);
    for (uint32_t c = 0; c < numCells; ++c) {
        const CellGeometry& cell = cells[c];
        for (size_t f = 0; f < cell.faceCount; ++f) {
            const uint32_t t = faceIncidences[cell.faceOffset + f].traceIndex;
            if (t < totalTraceCount) {
                traceCellCount[t]++;
            }
        }
    }

    std::vector<uint32_t> traceIncOffsets(totalTraceCount + 1, 0);
    size_t totalInc = 0;
    for (uint32_t t = 0; t < totalTraceCount; ++t) {
        traceIncOffsets[t] = static_cast<uint32_t>(totalInc);
        totalInc += traceCellCount[t];
    }
    traceIncOffsets[totalTraceCount] = static_cast<uint32_t>(totalInc);
    std::vector<CellFaceRef> allInc(totalInc);
    std::vector<uint32_t> currentIncOffset = traceIncOffsets;

    std::vector<std::vector<CouplingTerm>> traceCouplingMap(totalTraceCount);
    for (size_t i = 0; i < traceCouplings.size(); ++i) {
        const TraceCoupling& coupling = traceCouplings[i];
        if (coupling.traceA < totalTraceCount && coupling.traceB < totalTraceCount) {
            const float conductance = coupling.area * coupling.coefficientPerArea;
            traceCouplingMap[coupling.traceA].push_back({ coupling.traceB, conductance });
            traceCouplingMap[coupling.traceB].push_back({ coupling.traceA, conductance });
        }
    }

    outSystem.traceBoundaryRhs.assign(totalTraceCount, 0.0f);

    std::vector<DirichletConstraint> traceDirichlet(totalTraceCount, { false, 0.0f });
    std::vector<RobinTerm> traceRobin(totalTraceCount, { 0.0f });

    for (uint32_t c = 0; c < numCells; ++c) {
        const CellGeometry& cell = cells[c];
        for (size_t f = 0; f < cell.faceCount; ++f) {
            const FaceGeometry& face = faceIncidences[cell.faceOffset + f];
            const uint32_t t = face.traceIndex;
            if (t >= totalTraceCount) continue;

            const uint32_t bc = face.boundaryCondition;
            if (bc < boundaryCount) {
                const BoundaryCondition& condition = boundaries[bc];
                const double A_f = static_cast<double>(face.area);
                if (condition.type == BoundaryType::Dirichlet) {
                    traceDirichlet[t] = { true, condition.value };
                } else if (condition.type == BoundaryType::Robin) {
                    const float coefficient = static_cast<float>(A_f * static_cast<double>(condition.robinCoefficient));
                    traceRobin[t].coefficient += coefficient;
                    outSystem.traceBoundaryRhs[t] += static_cast<float>(coefficient * static_cast<double>(condition.referenceValue));
                } else if (condition.type == BoundaryType::Neumann) {
                    outSystem.traceBoundaryRhs[t] += static_cast<float>(A_f * static_cast<double>(condition.value));
                }
            }
        }
    }

    for (size_t i = 0; i < dirichletTraces.size(); ++i) {
        const DirichletTrace& trace = dirichletTraces[i];
        if (trace.traceIndex < totalTraceCount) {
            traceDirichlet[trace.traceIndex] = { true, trace.value };
        }
    }

    std::atomic<bool> errorOccurred{false};
    std::atomic<int> failedCellCount{0};

    #pragma omp parallel for schedule(static)
    for (int c_int = 0; c_int < static_cast<int>(numCells); ++c_int) {
        const uint32_t c = static_cast<uint32_t>(c_int);
        const CellGeometry& cell = cells[c];
        const size_t nF = cell.faceCount;
        const double k_E = (c < cellDiffusionCoefficient.size()) ? static_cast<double>(cellDiffusionCoefficient[c]) : 1.0;
        const double capOverDt = (c < cellStorageOverDt.size()) ? static_cast<double>(cellStorageOverDt[c]) : 0.0;

        CellValidation validation{};
        std::vector<double> W_E;
        const bool ok = computeElementFluxOperator(faceIncidences, cell, k_E, W_E, validation);

        if (!ok || !validation.passed) {
            errorOccurred.store(true, std::memory_order_relaxed);
            const int failIdx = failedCellCount.fetch_add(1, std::memory_order_relaxed);
            if (failIdx < 10) {
                #pragma omp critical
                {
                    std::cerr << "[VEMDiffusion] Cell " << c << " FAILED validation: faces=" << nF
                              << ", vol=" << cell.volume << ", k=" << k_E
                              << ", closureErr=" << validation.closureError
                              << ", GaussErr=" << validation.gaussConsistencyError
                              << ", det(NN^T)=" << validation.detNNT
                              << ", eigMin(NN^T)=" << validation.minEigenNNT
                              << ", eigMax(NN^T)=" << validation.maxEigenNNT
                              << ", cond(NN^T)=" << validation.conditionNNT
                              << ", Cholesky=" << (ok ? "OK" : "FAILED")
                              << " (failedPivot=" << validation.failedPivot << ", pivotVal=" << validation.failedPivotValue << ")\n";
                    std::cerr << "  ownerModelId=" << cell.ownerModelId
                              << " seedId=" << cell.seedId
                              << " centroid=(" << cell.centroid.x << ", "
                              << cell.centroid.y << ", " << cell.centroid.z << ")\n";
                    for (size_t f = 0; f < nF; ++f) {
                        const FaceGeometry& face = faceIncidences[cell.faceOffset + f];
                        const glm::dvec3 faceCentroid = face.area > 0.0
                            ? face.moment / face.area
                            : glm::dvec3(0.0);
                        const glm::dvec3 faceNormal = face.area > 0.0
                            ? face.areaVector / face.area
                            : glm::dvec3(0.0);
                        const glm::dvec3 faceOffset = faceCentroid - cell.centroid;
                        std::cerr << "  face[" << f << "] trace=" << face.traceIndex
                                  << " neighbor=" << face.neighborCellId
                                  << " neighborOwner=" << face.neighborOwnerModelId
                                  << " neighborSeed=" << face.neighborSeedId
                                  << " area=" << face.area
                                  << " areaVector=(" << face.areaVector.x << ", "
                                  << face.areaVector.y << ", " << face.areaVector.z << ")"
                                  << " |areaVector|=" << glm::length(face.areaVector)
                                  << " normalOffset=" << glm::dot(faceOffset, faceNormal)
                                  << " centroid=(" << faceCentroid.x << ", "
                                  << faceCentroid.y << ", " << faceCentroid.z << ")\n";
                    }

                    std::vector<uint32_t> sameOwnerNeighbors;
                    for (size_t f = 0; f < nF; ++f) {
                        const uint32_t neighbor = faceIncidences[cell.faceOffset + f].neighborCellId;
                        if (neighbor >= numCells || neighbor == c ||
                            cells[neighbor].ownerModelId != cell.ownerModelId ||
                            std::find(sameOwnerNeighbors.begin(), sameOwnerNeighbors.end(), neighbor) != sameOwnerNeighbors.end()) {
                            continue;
                        }
                        sameOwnerNeighbors.push_back(neighbor);
                    }
                    for (const uint32_t neighbor : sameOwnerNeighbors) {
                        const CellGeometry& other = cells[neighbor];
                        std::vector<FaceGeometry> mergedFaces;
                        mergedFaces.reserve(size_t(cell.faceCount) + other.faceCount);
                        double sharedArea = 0.0;
                        for (size_t f = 0; f < cell.faceCount; ++f) {
                            const FaceGeometry& face = faceIncidences[cell.faceOffset + f];
                            if (face.neighborCellId == neighbor) {
                                sharedArea += face.area;
                            } else {
                                mergedFaces.push_back(face);
                            }
                        }
                        for (size_t f = 0; f < other.faceCount; ++f) {
                            const FaceGeometry& face = faceIncidences[other.faceOffset + f];
                            if (face.neighborCellId != c) mergedFaces.push_back(face);
                        }

                        CellGeometry merged{};
                        merged.faceOffset = 0;
                        merged.faceCount = static_cast<uint32_t>(mergedFaces.size());
                        merged.ownerModelId = cell.ownerModelId;
                        merged.volume = cell.volume + other.volume;
                        merged.centroid = (cell.centroid * cell.volume + other.centroid * other.volume) / merged.volume;

                        CellValidation mergedValidation{};
                        std::vector<double> mergedOperator;
                        const bool mergedOk = computeElementFluxOperator(
                            mergedFaces, merged, k_E, mergedOperator, mergedValidation);
                        std::cerr << "  sameOwnerMerge neighbor=" << neighbor
                                  << " neighborSeed=" << other.seedId
                                  << " sharedArea=" << sharedArea
                                  << " faces=" << merged.faceCount
                                  << " vol=" << merged.volume
                                  << " closureErr=" << mergedValidation.closureError
                                  << " GaussErr=" << mergedValidation.gaussConsistencyError
                                  << " eigMin(NN^T)=" << mergedValidation.minEigenNNT
                                  << " eigMax(NN^T)=" << mergedValidation.maxEigenNNT
                                  << " cond(NN^T)=" << mergedValidation.conditionNNT
                                  << " Cholesky=" << (mergedOk ? "OK" : "FAILED")
                                  << " qualityPass=" << (mergedOk && mergedValidation.passed ? "yes" : "no")
                                  << '\n';
                    }
                }
            }
            continue;
        }

        std::vector<double> w_E(nF, 0.0);
        double sumW = 0.0;
        for (size_t i = 0; i < nF; ++i) {
            double rowSum = 0.0;
            for (size_t j = 0; j < nF; ++j) {
                rowSum += W_E[i * nF + j];
            }
            w_E[i] = rowSum;
            sumW += rowSum;
        }

        const double d_E = capOverDt + sumW;
        if (d_E <= 1e-15 || !std::isfinite(d_E)) {
            errorOccurred.store(true, std::memory_order_relaxed);
            continue;
        }

        const double invD = 1.0 / d_E;
        outSystem.cellRecoveryScale[c] = static_cast<float>(invD);

        const uint32_t cellOffset = outSystem.cellTraceOffsets[c];
        for (size_t f = 0; f < nF; ++f) {
            const FaceGeometry& face = faceIncidences[cell.faceOffset + f];
            outSystem.cellTraceIndices[cellOffset + f] = face.traceIndex;
            outSystem.cellTraceWeights[cellOffset + f] = static_cast<float>(w_E[f]);
        }

        float* cellK = &allK[cellKOffsets[c]];
        for (size_t i = 0; i < nF; ++i) {
            for (size_t j = 0; j < nF; ++j) {
                cellK[i * nF + j] = static_cast<float>(W_E[i * nF + j] - invD * w_E[i] * w_E[j]);
            }
        }
    }

    if (errorOccurred.load(std::memory_order_relaxed)) {
        std::cerr << "[VEMDiffusion] Element inversion or validation failed for "
                  << failedCellCount.load() << " cells.\n";
        return false;
    }

    outSystem.traceScatterOffsets = traceIncOffsets;
    outSystem.traceScatterCells.resize(totalInc);
    outSystem.traceScatterWeights.resize(totalInc);
    for (uint32_t c = 0; c < numCells; ++c) {
        const CellGeometry& cell = cells[c];
        const uint32_t cellOffset = outSystem.cellTraceOffsets[c];
        for (size_t f = 0; f < cell.faceCount; ++f) {
            const uint32_t t = faceIncidences[cell.faceOffset + f].traceIndex;
            if (t < totalTraceCount) {
                const uint32_t dst = currentIncOffset[t]++;
                allInc[dst] = { c, static_cast<uint16_t>(f) };
                outSystem.traceScatterCells[dst] = c;
                outSystem.traceScatterWeights[dst] = outSystem.cellTraceWeights[cellOffset + f];
            }
        }
    }

    outSystem.rowOffsets.resize(totalTraceCount + 1, 0);
    std::vector<int> rowNnz(totalTraceCount, 0);

    #pragma omp parallel for schedule(static)
    for (int t_int = 0; t_int < static_cast<int>(totalTraceCount); ++t_int) {
        const uint32_t t = static_cast<uint32_t>(t_int);
        RowTerm stackEntries[64];
        std::vector<RowTerm> heapEntries;
        RowTerm* entries = nullptr;
        const int numEntries = buildRowEntries(
            t, totalTraceCount, traceIncOffsets, allInc, faceIncidences, cells, allK, cellKOffsets,
            traceCouplingMap, traceRobin, stackEntries, 64, heapEntries, entries);

        int uniqueCount = 0;
        for (int i = 0; i < numEntries; ++i) {
            if (i == 0 || entries[i].column != entries[i - 1].column) {
                ++uniqueCount;
            }
        }
        rowNnz[t] = uniqueCount;
    }

    size_t totalNnz = 0;
    for (uint32_t t = 0; t < totalTraceCount; ++t) {
        outSystem.rowOffsets[t] = static_cast<int>(totalNnz);
        totalNnz += rowNnz[t];
    }
    outSystem.rowOffsets[totalTraceCount] = static_cast<int>(totalNnz);

    outSystem.columnIndices.resize(totalNnz);
    outSystem.values.resize(totalNnz);

    #pragma omp parallel for schedule(static)
    for (int t_int = 0; t_int < static_cast<int>(totalTraceCount); ++t_int) {
        const uint32_t t = static_cast<uint32_t>(t_int);
        RowTerm stackEntries[64];
        std::vector<RowTerm> heapEntries;
        RowTerm* entries = nullptr;
        const int numEntries = buildRowEntries(
            t, totalTraceCount, traceIncOffsets, allInc, faceIncidences, cells, allK, cellKOffsets,
            traceCouplingMap, traceRobin, stackEntries, 64, heapEntries, entries);

        int dst = outSystem.rowOffsets[t];
        for (int i = 0; i < numEntries; ++i) {
            if (i > 0 && entries[i].column == entries[i - 1].column) {
                outSystem.values[dst - 1] += entries[i].value;
            } else {
                outSystem.columnIndices[dst] = static_cast<int>(entries[i].column);
                outSystem.values[dst] = entries[i].value;
                ++dst;
            }
        }
    }

    outSystem.traceIsDirichlet.assign(totalTraceCount, 0);
    outSystem.traceDirichletValue.assign(totalTraceCount, 0.0f);

    for (uint32_t t = 0; t < totalTraceCount; ++t) {
        if (!traceDirichlet[t].isDirichlet) continue;
        const float prescribed = traceDirichlet[t].value;
        outSystem.traceIsDirichlet[t] = 1;
        outSystem.traceDirichletValue[t] = prescribed;
        const int startIdx = outSystem.rowOffsets[t];
        const int endIdx = outSystem.rowOffsets[t + 1];

        for (int k = startIdx; k < endIdx; ++k) {
            const int column = outSystem.columnIndices[k];
            if (column == static_cast<int>(t)) {
                outSystem.values[k] = 1.0f;
            } else {
                const float value = outSystem.values[k];
                outSystem.values[k] = 0.0f;

                if (!traceDirichlet[column].isDirichlet) {
                    outSystem.traceBoundaryRhs[column] -= value * prescribed;
                }

                const int colStart = outSystem.rowOffsets[column];
                const int colEnd = outSystem.rowOffsets[column + 1];
                for (int ck = colStart; ck < colEnd; ++ck) {
                    if (outSystem.columnIndices[ck] == static_cast<int>(t)) {
                        outSystem.values[ck] = 0.0f;
                        break;
                    }
                }
            }
        }
        outSystem.traceBoundaryRhs[t] = prescribed;
    }

    uint32_t zeroIncidence = 0;
    uint32_t oneIncidence = 0;
    for (uint32_t t = 0; t < totalTraceCount; ++t) {
        const uint32_t incidence = traceIncOffsets[t + 1] - traceIncOffsets[t];
        if (incidence == 0) ++zeroIncidence;
        else if (incidence == 1) ++oneIncidence;
    }

    uint32_t zeroDiagonal = 0;
    uint32_t negativeDiagonal = 0;
    double diagonalMin = std::numeric_limits<double>::max();
    double diagonalMax = 0.0;
    uint32_t minDiagonalTrace = UINT32_MAX;
    for (uint32_t t = 0; t < totalTraceCount; ++t) {
        const int start = outSystem.rowOffsets[t];
        const int end = outSystem.rowOffsets[t + 1];
        double diagonal = 0.0;
        bool found = false;
        for (int k = start; k < end; ++k) {
            if (outSystem.columnIndices[k] == static_cast<int>(t)) {
                diagonal = outSystem.values[k];
                found = true;
                break;
            }
        }
        if (!found || !(std::abs(diagonal) > 0.0) || !std::isfinite(diagonal)) {
            ++zeroDiagonal;
            continue;
        }
        if (diagonal < 0.0) ++negativeDiagonal;
        const double magnitude = std::fabs(diagonal);
        if (magnitude < diagonalMin) {
            diagonalMin = magnitude;
            minDiagonalTrace = t;
        }
        diagonalMax = std::max(diagonalMax, magnitude);
    }

    uint32_t asymmetricEntries = 0;
    double maxSymmetryError = 0.0;
    const bool suspiciousConditioning =
        zeroDiagonal > 0 || negativeDiagonal > 0 ||
        diagonalMin <= 1e-9 * diagonalMax;
    if (suspiciousConditioning) {
        for (uint32_t t = 0; t < totalTraceCount; ++t) {
            const int start = outSystem.rowOffsets[t];
            const int end = outSystem.rowOffsets[t + 1];
            for (int k = start; k < end; ++k) {
                const int column = outSystem.columnIndices[k];
                if (column == static_cast<int>(t)) continue;
                const int otherStart = outSystem.rowOffsets[column];
                const int otherEnd = outSystem.rowOffsets[column + 1];
                int lo = otherStart;
                int hi = otherEnd;
                while (lo < hi) {
                    const int mid = (lo + hi) / 2;
                    if (outSystem.columnIndices[mid] < static_cast<int>(t)) lo = mid + 1;
                    else hi = mid;
                }
                const double value = outSystem.values[k];
                if (lo >= otherEnd || outSystem.columnIndices[lo] != static_cast<int>(t)) {
                    ++asymmetricEntries;
                    continue;
                }
                const double other = outSystem.values[lo];
                const double scale = std::max(std::max(std::fabs(value), std::fabs(other)), 1e-30);
                maxSymmetryError = std::max(maxSymmetryError, std::fabs(value - other) / scale);
            }
        }
    }

    double minVolume = std::numeric_limits<double>::max();
    double maxVolume = 0.0;
    double minFaceArea = std::numeric_limits<double>::max();
    double maxFaceArea = 0.0;
    for (uint32_t c = 0; c < numCells; ++c) {
        const CellGeometry& cell = cells[c];
        minVolume = std::min(minVolume, cell.volume);
        maxVolume = std::max(maxVolume, cell.volume);
        for (size_t f = 0; f < cell.faceCount; ++f) {
            const double area = faceIncidences[cell.faceOffset + f].area;
            if (!(area > 0.0)) continue;
            minFaceArea = std::min(minFaceArea, area);
            maxFaceArea = std::max(maxFaceArea, area);
        }
    }

    const bool reportDiagnostics =
        zeroIncidence > 0 || zeroDiagonal > 0 || negativeDiagonal > 0 ||
        asymmetricEntries > 0 || (diagonalMax > 0.0 && diagonalMin <= 1e-9 * diagonalMax);
    if (reportDiagnostics) {
        std::cerr << "[VEMDiffusion] traces=" << totalTraceCount
                  << " cells=" << numCells
                  << " zeroIncidence=" << zeroIncidence
                  << " oneIncidence=" << oneIncidence
                  << " zeroDiagonal=" << zeroDiagonal
                  << " negativeDiagonal=" << negativeDiagonal
                  << " asymmetricEntries=" << asymmetricEntries
                  << " maxSymmetryError=" << maxSymmetryError
                  << " diagonalMin=" << diagonalMin
                  << " diagonalMax=" << diagonalMax
                  << " minVolume=" << minVolume
                  << " maxVolume=" << maxVolume
                  << " minFaceArea=" << minFaceArea
                  << " maxFaceArea=" << maxFaceArea
                  << std::endl;

        if (minDiagonalTrace < totalTraceCount) {
            const uint32_t incStart = traceIncOffsets[minDiagonalTrace];
            const uint32_t incEnd = traceIncOffsets[minDiagonalTrace + 1];
            for (uint32_t inc = incStart; inc < incEnd; ++inc) {
                const CellFaceRef& incidence = allInc[inc];
                const CellGeometry& cell = cells[incidence.cellIndex];
                const FaceGeometry& face = faceIncidences[cell.faceOffset + incidence.localFaceIndex];
                std::cerr << "  minDiagonalTrace=" << minDiagonalTrace
                          << " cell=" << incidence.cellIndex
                          << " localFace=" << incidence.localFaceIndex
                          << " faceArea=" << face.area
                          << " cellVolume=" << cell.volume
                          << " cellFaceCount=" << cell.faceCount
                          << std::endl;
            }
        }
    }

    return true;
}

} // namespace vem
