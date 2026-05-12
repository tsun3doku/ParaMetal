#pragma once

#include <cstdint>

// Legacy compatibility header. The heat solver now uses the solver-agnostic
// contact/ContactSystemComputeStage directly.
#include "contact/ContactSystemComputeStage.hpp"

using HeatSystemContactStage = ContactSystemComputeStage;
