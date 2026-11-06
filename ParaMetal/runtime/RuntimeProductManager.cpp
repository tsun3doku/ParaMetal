#include "RuntimeProductManager.hpp"

#include <cassert>

RuntimeProductManager::~RuntimeProductManager() {
    assert(modelProducts.empty() && remeshProducts.empty() && voronoiProducts.empty() &&
           pointProducts.empty() && heatProducts.empty() && cfdProducts.empty());
}
