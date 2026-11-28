#pragma once

#include <cstdint>

enum class SubselectionCommand : uint8_t {
    Apply = 1,
    Cancel = 2,
    Clear = 3
};
