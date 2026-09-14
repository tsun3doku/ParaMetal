#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDTopology.hpp"
#include "RVDRestriction.cuh"

#include <vector>
#include <cstdint>

namespace voronoi {

bool buildFragments(
    const RVDInput& input,
    const BaseCellsContext& baseCtx,
    const std::vector<ModelRestrictionData>& models,
    std::vector<std::vector<uint32_t>>& fragmentIdByModelCell,
    std::vector<uint32_t>& unownedFragmentIdByCell,
    RVDResult& result);

bool buildFragmentFaces(
    const BaseCellsContext& baseCtx,
    const TopologyContext& topoCtx,
    const std::vector<ModelRestrictionData>& models,
    const std::vector<std::vector<uint32_t>>& fragmentIdByModelCell,
    const std::vector<uint32_t>& unownedFragmentIdByCell,
    RVDResult& result);

} // namespace voronoi
