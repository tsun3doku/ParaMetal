#pragma once

#include "domain/ContactData.hpp"
#include "domain/GeometryData.hpp"
#include "domain/HeatData.hpp"
#include "domain/RemeshData.hpp"
#include "domain/VoronoiData.hpp"

#include <cstdint>

namespace payloadtypes {
    extern uint8_t None;
    extern uint8_t Geometry;
    extern uint8_t Remesh;
    extern uint8_t HeatModel;
    extern uint8_t Heat;
    extern uint8_t Voronoi;
    extern uint8_t Contact;
    extern uint8_t Points;
}
