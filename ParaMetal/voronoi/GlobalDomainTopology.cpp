#include "GlobalDomainTopology.hpp"

#include <cmath>
#include <iostream>

namespace topology {

bool validateBoundaryFaceIndices(const GlobalDomain& domain) {
    size_t slot = 0;
    for (size_t faceIndex = 0; faceIndex < domain.faces.size(); ++faceIndex) {
        const Face& face = domain.faces[faceIndex];
        if (face.type != FaceType::DomainBoundary) {
            continue;
        }
        if (slot >= domain.boundaryFaceIndices.size()) {
            std::cerr << "[GlobalDomain] face " << faceIndex
                      << " is a domain boundary face with no boundary slot\n";
            return false;
        }
        const uint32_t declared = domain.boundaryFaceIndices[slot];
        if (declared != faceIndex) {
            std::cerr << "[GlobalDomain] boundary slot " << slot << " declares face " << declared
                      << " but the next domain boundary face is " << faceIndex << '\n';
            return false;
        }
        if (face.fragmentA >= domain.fragments.size()) {
            std::cerr << "[GlobalDomain] boundary slot " << slot << " face " << faceIndex
                      << " references out-of-range fragment " << face.fragmentA << '\n';
            return false;
        }
        if (face.domainFace >= DomainFaceCount) {
            std::cerr << "[GlobalDomain] boundary slot " << slot << " face " << faceIndex
                      << " references domain face " << face.domainFace << '\n';
            return false;
        }
        if (!std::isfinite(face.areaA) || !(face.areaA > 0.0)) {
            std::cerr << "[GlobalDomain] boundary slot " << slot << " face " << faceIndex
                      << " has area " << face.areaA << '\n';
            return false;
        }
        if (!std::isfinite(face.momentA.x) || !std::isfinite(face.momentA.y) ||
            !std::isfinite(face.momentA.z) ||
            !std::isfinite(face.areaVectorA.x) || !std::isfinite(face.areaVectorA.y) ||
            !std::isfinite(face.areaVectorA.z)) {
            std::cerr << "[GlobalDomain] boundary slot " << slot << " face " << faceIndex
                      << " has a non-finite moment or area vector\n";
            return false;
        }
        ++slot;
    }
    if (slot != domain.boundaryFaceIndices.size()) {
        std::cerr << "[GlobalDomain] boundary slot " << slot << " declares face "
                  << domain.boundaryFaceIndices[slot]
                  << " which is not a domain boundary face\n";
        return false;
    }
    return true;
}

} // namespace topology
