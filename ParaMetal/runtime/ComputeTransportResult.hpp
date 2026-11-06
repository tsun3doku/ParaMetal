#pragma once

#include <cstdint>

enum class ComputeApplyResult : uint8_t {
    Failed,
    Unchanged,
    Prepared
};
