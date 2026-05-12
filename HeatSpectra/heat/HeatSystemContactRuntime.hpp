#pragma once


// Legacy compatibility header. The old HeatSystemContactRuntime manager has been
// replaced by per-coupling HeatContactRuntime instances owned by HeatSystem.
#include "heat/HeatContactRuntime.hpp"

using HeatSystemContactRuntime = HeatContactRuntime;
