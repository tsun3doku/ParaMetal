#pragma once

#include <cstdint>

//                                                          [ HashProduct
//                                                            -Centralized product sealing
//                                                            -Computes domain specific hash values for each
//                                                            runtime product type
//                                                            -All product hash construction flows through here ]

struct ModelProduct;
struct RemeshProduct;
struct VoronoiProduct;
struct PointProduct;
struct HeatProduct;
struct CFDProduct;

class HashProduct {
public:
    static void seal(ModelProduct& product);
    static void seal(RemeshProduct& product);
    static void seal(VoronoiProduct& product);
    static void seal(PointProduct& product);
    static void seal(HeatProduct& product);
    static void seal(CFDProduct& product, uint64_t geometryHash, uint64_t simulationHash);
};
