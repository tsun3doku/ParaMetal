#include "VEMAgglomerator.hpp"

#include "VEMDiffusion.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <utility>

namespace vem {
namespace {

constexpr uint32_t InvalidCell = UINT32_MAX;
constexpr float FixedTemperatureTolerance = 1e-3f;

struct ActiveCell {
    CellGeometry geometry;
    std::vector<FaceGeometry> faces;
    std::vector<uint32_t> members;
    bool active = true;
};

struct Candidate {
    uint32_t neighbor = InvalidCell;
    std::unordered_set<uint32_t> sharedTraces;
    CellGeometry geometry;
    std::vector<FaceGeometry> faces;
    CellValidation validation;
    double sharedArea = 0.0;
};

float finiteOrInfinity(float value) {
    return std::isfinite(value) ? value : std::numeric_limits<float>::infinity();
}

bool fixedCompatible(float a, float b) {
    const bool aFixed = std::isfinite(a);
    const bool bFixed = std::isfinite(b);
    if (aFixed != bFixed) return false;
    return !aFixed || std::abs(a - b) <= FixedTemperatureTolerance;
}

bool candidateBetter(const Candidate& a, const Candidate& b) {
    if (a.validation.passed != b.validation.passed) return a.validation.passed;
    const float aCondition = finiteOrInfinity(a.validation.conditionNNT);
    const float bCondition = finiteOrInfinity(b.validation.conditionNNT);
    if (aCondition != bCondition) return aCondition < bCondition;
    const float aMinEigen = finiteOrInfinity(a.validation.minEigenNNT);
    const float bMinEigen = finiteOrInfinity(b.validation.minEigenNNT);
    if (aMinEigen != bMinEigen) return aMinEigen > bMinEigen;
    const float aGauss = finiteOrInfinity(a.validation.gaussConsistencyError);
    const float bGauss = finiteOrInfinity(b.validation.gaussConsistencyError);
    if (aGauss != bGauss) return aGauss < bGauss;
    const float aClosure = finiteOrInfinity(a.validation.closureError);
    const float bClosure = finiteOrInfinity(b.validation.closureError);
    if (aClosure != bClosure) return aClosure < bClosure;
    if (a.sharedArea != b.sharedArea) return a.sharedArea > b.sharedArea;
    return a.neighbor < b.neighbor;
}

bool improves(const Candidate& candidate, const CellValidation& current) {
    if (candidate.validation.passed) return true;
    if (!(candidate.validation.conditionNNT > 0.0f) ||
        !std::isfinite(candidate.validation.conditionNNT)) return false;
    if (!(current.conditionNNT > 0.0f) && candidate.validation.conditionNNT > 0.0f) return true;
    const float candidateCondition = finiteOrInfinity(candidate.validation.conditionNNT);
    const float currentCondition = finiteOrInfinity(current.conditionNNT);
    if (candidateCondition != currentCondition) return candidateCondition < currentCondition;
    const float candidateMinEigen = finiteOrInfinity(candidate.validation.minEigenNNT);
    const float currentMinEigen = finiteOrInfinity(current.minEigenNNT);
    if (candidateMinEigen != currentMinEigen) return candidateMinEigen > currentMinEigen;
    const float candidateGauss = finiteOrInfinity(candidate.validation.gaussConsistencyError);
    const float currentGauss = finiteOrInfinity(current.gaussConsistencyError);
    if (candidateGauss != currentGauss) return candidateGauss < currentGauss;
    return finiteOrInfinity(candidate.validation.closureError) <
           finiteOrInfinity(current.closureError);
}

bool validate(
    const std::vector<FaceGeometry>& faces,
    CellGeometry geometry,
    double conductivity,
    CellValidation& out) {
    geometry.faceOffset = 0;
    geometry.faceCount = static_cast<uint32_t>(faces.size());
    return VEMDiffusion::validateElementGeometry(faces, geometry, conductivity, out);
}

bool makeCandidate(
    uint32_t aIndex,
    uint32_t bIndex,
    const ActiveCell& a,
    const ActiveCell& b,
    const std::vector<float>& fixedTemperatures,
    double conductivity,
    Candidate& candidate) {
    if (a.geometry.ownerModelId == UINT32_MAX ||
        a.geometry.ownerModelId != b.geometry.ownerModelId ||
        aIndex == bIndex ||
        !fixedCompatible(fixedTemperatures[a.members.front()], fixedTemperatures[b.members.front()])) {
        return false;
    }

    candidate = Candidate{};
    candidate.neighbor = bIndex;
    for (const FaceGeometry& face : a.faces) {
        if (face.neighborCellId == bIndex) {
            candidate.sharedTraces.insert(face.traceIndex);
            candidate.sharedArea += face.area;
        }
    }
    if (candidate.sharedTraces.empty()) return false;

    std::unordered_set<uint32_t> matched;
    for (const FaceGeometry& face : b.faces) {
        if (face.neighborCellId == aIndex && candidate.sharedTraces.count(face.traceIndex)) {
            matched.insert(face.traceIndex);
        }
    }
    if (matched.size() != candidate.sharedTraces.size()) return false;

    candidate.geometry = a.geometry;
    const double volume = a.geometry.volume + b.geometry.volume;
    if (!(volume > 0.0) || !std::isfinite(volume)) return false;
    candidate.geometry.volume = volume;
    candidate.geometry.centroid =
        (a.geometry.centroid * a.geometry.volume + b.geometry.centroid * b.geometry.volume) / volume;
    candidate.geometry.faceOffset = 0;
    candidate.geometry.faceCount = 0;
    candidate.geometry.seedId = std::min(a.geometry.seedId, b.geometry.seedId);

    candidate.faces.reserve(a.faces.size() + b.faces.size());
    auto appendExterior = [&](const std::vector<FaceGeometry>& source) {
        for (FaceGeometry face : source) {
            if (candidate.sharedTraces.count(face.traceIndex) &&
                (face.neighborCellId == aIndex || face.neighborCellId == bIndex)) {
                continue;
            }
            if (face.neighborCellId == aIndex || face.neighborCellId == bIndex) {
                face.neighborCellId = aIndex;
            }
            candidate.faces.push_back(face);
        }
    };
    appendExterior(a.faces);
    appendExterior(b.faces);
    candidate.geometry.faceCount = static_cast<uint32_t>(candidate.faces.size());
    validate(candidate.faces, candidate.geometry, conductivity, candidate.validation);
    return true;
}

} // namespace

bool VEMAgglomerator::remapInputs(
    const VEMCellRemap& remap,
    const VEMCellInputs& fragmentInputs,
    VEMCellInputs& cellInputs) {
    const size_t fragmentCount = fragmentInputs.conductivity.size();
    if (fragmentInputs.capacityOverDt.size() != fragmentCount ||
        fragmentInputs.sourceAdditions.size() != fragmentCount ||
        fragmentInputs.fixedTemperatures.size() != fragmentCount ||
        remap.fragmentToCell.size() != fragmentCount ||
        remap.cellFragmentOffsets.empty() || remap.cellFragmentOffsets.front() != 0 ||
        remap.cellFragmentOffsets.back() != remap.cellFragments.size()) {
        return false;
    }

    const size_t cellCount = remap.cellFragmentOffsets.size() - 1;
    cellInputs = VEMCellInputs{};
    cellInputs.conductivity.resize(cellCount);
    cellInputs.capacityOverDt.assign(cellCount, 0.0f);
    cellInputs.sourceAdditions.assign(cellCount, 0.0f);
    cellInputs.fixedTemperatures.assign(
        cellCount, std::numeric_limits<float>::quiet_NaN());
    std::vector<uint8_t> mappedFragments(fragmentCount, 0);

    for (size_t cell = 0; cell < cellCount; ++cell) {
        const uint32_t begin = remap.cellFragmentOffsets[cell];
        const uint32_t end = remap.cellFragmentOffsets[cell + 1];
        if (begin >= end || end > remap.cellFragments.size() ||
            (cell > 0 && begin < remap.cellFragmentOffsets[cell - 1])) return false;
        const uint32_t first = remap.cellFragments[begin];
        if (first >= fragmentCount) return false;
        cellInputs.conductivity[cell] = fragmentInputs.conductivity[first];
        cellInputs.fixedTemperatures[cell] = fragmentInputs.fixedTemperatures[first];
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t fragment = remap.cellFragments[i];
            if (fragment >= fragmentCount ||
                remap.fragmentToCell[fragment] != cell ||
                mappedFragments[fragment] != 0 ||
                !fixedCompatible(fragmentInputs.fixedTemperatures[first], fragmentInputs.fixedTemperatures[fragment]) ||
                std::abs(fragmentInputs.conductivity[first] - fragmentInputs.conductivity[fragment]) >
                    1e-5f * std::max(1.0f, std::abs(fragmentInputs.conductivity[first]))) {
                return false;
            }
            mappedFragments[fragment] = 1;
            cellInputs.capacityOverDt[cell] += fragmentInputs.capacityOverDt[fragment];
            cellInputs.sourceAdditions[cell] += fragmentInputs.sourceAdditions[fragment];
        }
    }
    return std::find(mappedFragments.begin(), mappedFragments.end(), uint8_t(0)) == mappedFragments.end();
}

bool VEMAgglomerator::build(
    VEMMesh mesh,
    const VEMCellInputs& fragmentInputs,
    VEMAgglomerationResult& result) {
    result = VEMAgglomerationResult{};
    const size_t fragmentCount = mesh.cells.size();
    if (fragmentCount == 0 || mesh.faceIncidences.empty() ||
        fragmentInputs.conductivity.size() != fragmentCount ||
        fragmentInputs.capacityOverDt.size() != fragmentCount ||
        fragmentInputs.sourceAdditions.size() != fragmentCount ||
        fragmentInputs.fixedTemperatures.size() != fragmentCount) {
        return false;
    }

    std::vector<ActiveCell> cells(fragmentCount);
    for (uint32_t cell = 0; cell < fragmentCount; ++cell) {
        cells[cell].geometry = mesh.cells[cell];
        cells[cell].members.push_back(cell);
        const CellGeometry& geometry = mesh.cells[cell];
        if (size_t(geometry.faceOffset) + geometry.faceCount > mesh.faceIncidences.size()) return false;
        cells[cell].faces.insert(
            cells[cell].faces.end(),
            mesh.faceIncidences.begin() + geometry.faceOffset,
            mesh.faceIncidences.begin() + geometry.faceOffset + geometry.faceCount);
    }

    const size_t mergeLimit = std::min<size_t>(fragmentCount > 0 ? fragmentCount - 1 : 0, 4096);
    size_t mergeCount = 0;
    while (mergeCount < mergeLimit) {
        uint32_t failingCell = InvalidCell;
        CellValidation failingValidation{};
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            if (!cells[cell].active) continue;
            CellValidation validation;
            const bool valid = validate(
                cells[cell].faces, cells[cell].geometry,
                fragmentInputs.conductivity[cells[cell].members.front()], validation);
            if (!valid) {
                failingCell = cell;
                failingValidation = validation;
                break;
            }
        }
        if (failingCell == InvalidCell) break;

        std::unordered_set<uint32_t> neighbors;
        for (const FaceGeometry& face : cells[failingCell].faces) {
            if (face.neighborCellId < cells.size() && face.neighborCellId != failingCell &&
                cells[face.neighborCellId].active) {
                neighbors.insert(face.neighborCellId);
            }
        }
        std::vector<uint32_t> orderedNeighbors(neighbors.begin(), neighbors.end());
        std::sort(orderedNeighbors.begin(), orderedNeighbors.end());

        Candidate best;
        bool haveBest = false;
        for (uint32_t neighbor : orderedNeighbors) {
            Candidate candidate;
            const bool eligible = makeCandidate(
                failingCell, neighbor, cells[failingCell], cells[neighbor],
                fragmentInputs.fixedTemperatures,
                fragmentInputs.conductivity[cells[failingCell].members.front()], candidate);
            if (!eligible || !improves(candidate, failingValidation)) continue;
            if (!haveBest || candidateBetter(candidate, best)) {
                best = std::move(candidate);
                haveBest = true;
            }
        }

        if (!haveBest) {
            std::cerr << "[VEMAgglomerator] no compatible improving neighbor for cell seed="
                      << cells[failingCell].geometry.seedId
                      << " owner=" << cells[failingCell].geometry.ownerModelId
                      << " condition=" << failingValidation.conditionNNT << std::endl;
            return false;
        }

        ActiveCell& survivor = cells[failingCell];
        ActiveCell& absorbed = cells[best.neighbor];
        const uint32_t absorbedCell = best.neighbor;
        survivor.geometry = best.geometry;
        survivor.faces = std::move(best.faces);
        survivor.members.insert(survivor.members.end(), absorbed.members.begin(), absorbed.members.end());
        std::sort(survivor.members.begin(), survivor.members.end());
        absorbed.active = false;

        for (ActiveCell& cell : cells) {
            if (!cell.active) continue;
            for (FaceGeometry& face : cell.faces) {
                if (face.neighborCellId == absorbedCell) face.neighborCellId = failingCell;
            }
        }

        ++mergeCount;
        std::cerr << "[VEMAgglomerator] merged seed=" << cells[failingCell].geometry.seedId
                  << " neighborSeed=" << cells[absorbedCell].geometry.seedId
                  << " faces=" << cells[failingCell].faces.size()
                  << " condition=" << best.validation.conditionNNT
                  << " passed=" << (best.validation.passed ? "yes" : "no") << std::endl;
    }

    for (const ActiveCell& cell : cells) {
        if (!cell.active) continue;
        CellValidation validation;
        if (!validate(cell.faces, cell.geometry,
                      fragmentInputs.conductivity[cell.members.front()], validation)) {
            std::cerr << "[VEMAgglomerator] mesh still has an invalid cell seed="
                      << cell.geometry.seedId << " condition=" << validation.conditionNNT << std::endl;
            return false;
        }
    }

    std::vector<uint32_t> activeToCompact(cells.size(), InvalidCell);
    std::vector<uint8_t> usedTraces(mesh.totalTraceCount, 0);
    uint32_t compactCellCount = 0;
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        if (!cells[cell].active) continue;
        activeToCompact[cell] = compactCellCount++;
        for (const FaceGeometry& face : cells[cell].faces) {
            if (face.traceIndex >= usedTraces.size()) return false;
            usedTraces[face.traceIndex] = 1;
        }
    }

    std::vector<uint32_t> traceRemap(mesh.totalTraceCount, InvalidCell);
    uint32_t compactTraceCount = 0;
    for (uint32_t trace = 0; trace < usedTraces.size(); ++trace) {
        if (usedTraces[trace]) traceRemap[trace] = compactTraceCount++;
    }

    result.mesh = VEMMesh{};
    result.mesh.totalTraceCount = compactTraceCount;
    result.mesh.cells.reserve(compactCellCount);
    result.remap.fragmentToCell.assign(fragmentCount, InvalidCell);
    result.remap.cellFragmentOffsets.reserve(size_t(compactCellCount) + 1);
    result.remap.cellFragmentOffsets.push_back(0);
    result.remap.cellFragments.reserve(fragmentCount);
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        if (!cells[cell].active) continue;
        CellGeometry geometry = cells[cell].geometry;
        geometry.faceOffset = static_cast<uint32_t>(result.mesh.faceIncidences.size());
        geometry.faceCount = static_cast<uint32_t>(cells[cell].faces.size());
        for (FaceGeometry face : cells[cell].faces) {
            if (face.traceIndex >= traceRemap.size() || traceRemap[face.traceIndex] == InvalidCell) return false;
            face.traceIndex = traceRemap[face.traceIndex];
            if (face.neighborCellId < cells.size()) {
                face.neighborCellId = activeToCompact[face.neighborCellId];
                if (face.neighborCellId == InvalidCell) return false;
            } else {
                face.neighborCellId = InvalidCell;
            }
            result.mesh.faceIncidences.push_back(face);
        }
        const uint32_t compact = activeToCompact[cell];
        for (uint32_t fragment : cells[cell].members) {
            if (fragment >= result.remap.fragmentToCell.size() ||
                result.remap.fragmentToCell[fragment] != InvalidCell) return false;
            result.remap.fragmentToCell[fragment] = compact;
            result.remap.cellFragments.push_back(fragment);
        }
        result.mesh.cells.push_back(geometry);
        result.remap.cellFragmentOffsets.push_back(
            static_cast<uint32_t>(result.remap.cellFragments.size()));
    }
    if (std::find(result.remap.fragmentToCell.begin(), result.remap.fragmentToCell.end(), InvalidCell) !=
        result.remap.fragmentToCell.end()) return false;

    result.mesh.traceCouplings.reserve(mesh.traceCouplings.size());
    for (const TraceCoupling& coupling : mesh.traceCouplings) {
        if (coupling.traceA >= traceRemap.size() || coupling.traceB >= traceRemap.size() ||
            traceRemap[coupling.traceA] == InvalidCell || traceRemap[coupling.traceB] == InvalidCell) {
            std::cerr << "[VEMAgglomerator] contact coupling references a removed trace" << std::endl;
            return false;
        }
        TraceCoupling remapped = coupling;
        remapped.traceA = traceRemap[coupling.traceA];
        remapped.traceB = traceRemap[coupling.traceB];
        result.mesh.traceCouplings.push_back(remapped);
    }

    if (!remapInputs(result.remap, fragmentInputs, result.inputs)) return false;
    std::cerr << "[VEMAgglomerator] complete fragments=" << fragmentCount
              << " cells=" << result.mesh.cells.size()
              << " merges=" << mergeCount
              << " traces=" << mesh.totalTraceCount << "->" << result.mesh.totalTraceCount
              << std::endl;
    return true;
}

} // namespace vem
