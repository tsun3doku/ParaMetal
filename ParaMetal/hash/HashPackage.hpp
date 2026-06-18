#pragma once

//                                                          [ HashPackage
//                                                            -Centralized package sealing
//                                                            -Computes domain specific hash values for each
//                                                            runtime package type
//                                                            -All package hash construction flows through here ]

struct ModelPackage;
struct RemeshPackage;
struct VoronoiPackage;
struct PointPackage;
struct HeatPackage;
struct ContactPackage;

class HashPackage {
public:
    static void seal(ModelPackage& pkg);
    static void seal(RemeshPackage& pkg);
    static void seal(VoronoiPackage& pkg);
    static void seal(PointPackage& pkg);
    static void seal(HeatPackage& pkg);
    static void seal(ContactPackage& pkg);
};
