#include "CFDPackageCompiler.hpp"

#include "PackageCompileContext.hpp"
#include "domain/BoundaryData.hpp"
#include "domain/CFDData.hpp"
#include "domain/FluidData.hpp"
#include "domain/VoronoiData.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashProductHandle.hpp"
#include "nodegraph/NodeCFDParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"

#include <utility>

namespace runtimepackage {

static void sealCFDPackage(CFDPackage& package, const HashValues& authoredHashes) {
    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combine(simulationHash, authoredHashes.simulation);
    combineProductHandleHash(simulationHash, package.domainVoronoiProduct, HashDomain::Geometry);
    HashBuilder::combineFloat(simulationHash, package.fluid.density);
    HashBuilder::combineFloat(simulationHash, package.fluid.kinematicViscosity);
    HashBuilder::combineFloat(simulationHash, package.initialVelocity.x);
    HashBuilder::combineFloat(simulationHash, package.initialVelocity.y);
    HashBuilder::combineFloat(simulationHash, package.initialVelocity.z);
    HashBuilder::combineFloat(simulationHash, package.fixedTimeStep);
    HashBuilder::combineFloat(simulationHash, package.targetCourant);
    HashBuilder::combine(simulationHash, package.active ? 1u : 0u);

    uint64_t displayHash = HashBuilder::start();
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showStreamlines ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showDomainBox ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showVelocityPalette ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showSlice ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.sliceAxis));
    HashBuilder::combineFloat(displayHash, package.display.slicePosition);
    HashBuilder::combineFloat(displayHash, package.canonicalToWorldScale);

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, simulationHash);
    HashBuilder::combine(fullHash, displayHash);
    HashBuilder::combine(fullHash, authoredHashes.full);

    package.hashes.full = fullHash;
    package.hashes.simulation = simulationHash;
    package.hashes.geometry = package.domainVoronoiProduct.hashes.geometry;
    package.hashes.display = displayHash;
}

bool compileCFDPackage(const PackageCompileContext& context) {
    const CFDNodeParams params = readCFDNodeParams(context.node);
    const CFDData* cfd = context.payloads.get<CFDData>(context.output.payloadHandle);
    if (!cfd) {
        context.errors.push_back("CFD package is missing its authored payload.");
        return false;
    }

    CFDPackage package{};
    const CFDPackage* previous = context.packages.findStored<CFDPackage>(context.outputSocketKey);
    package.productHandle = previous ? previous->productHandle : ProductHandle{};
    package.initialVelocity = cfd->initialVelocity;
    package.fixedTimeStep = cfd->fixedTimeStep;
    package.targetCourant = cfd->targetCourant;
    package.pressureIterations = cfd->pressureIterations;
    package.pressureTolerance = cfd->pressureTolerance;
    package.simulationDuration = cfd->simulationDuration;
    package.active = cfd->active;
    package.canonicalToWorldScale = units::canonicalToWorldScale(context.worldUnit);
    package.display.showStreamlines = params.preview.showStreamlines;
    package.display.showDomainBox = params.preview.showDomainBox;
    package.display.showVelocityPalette = params.preview.showVelocityPalette;
    package.display.showSlice = params.preview.showSlice;
    package.display.sliceAxis = params.preview.sliceAxis;
    package.display.slicePosition = static_cast<float>(params.preview.slicePosition);
    package.display.showSliceWireframe = params.preview.showSliceWireframe;

    const BoundaryData* boundary = cfd->boundaryHandle.key != 0
        ? context.payloads.get<BoundaryData>(cfd->boundaryHandle)
        : nullptr;
    if (boundary) {
        for (uint32_t i = 0; i < 6; ++i) {
            package.domainBoundaryTypes[i] = static_cast<uint32_t>(boundary->faces[i].flowType);
            package.domainBoundaryVelocities[i] = boundary->faces[i].velocity;
        }
    }

    const bool canResolveDomain = cfd->domainVoronoiHandle.key != 0;
    if (cfd->active) {
        if (!boundary) {
            context.errors.push_back("CFD requires a Boundary Conditions input.");
            return false;
        }
        if (!canResolveDomain) {
            context.errors.push_back("CFD requires an active Voronoi domain input.");
            return false;
        }
        const VoronoiData* voronoi = context.payloads.get<VoronoiData>(cfd->domainVoronoiHandle);
        if (!voronoi || !voronoi->active) {
            context.errors.push_back("CFD's domain Voronoi input is invalid.");
            return false;
        }
        if (!cfd->hasFluid || cfd->fluidHandle.key == 0) {
            context.errors.push_back("CFD requires a Fluid input.");
            return false;
        }
    }

    const VoronoiPackage* domainPackage = canResolveDomain
        ? context.packages.findAny<VoronoiPackage>(cfd->domainVoronoiHandle.key)
        : nullptr;
    if (canResolveDomain && (!domainPackage || !domainPackage->productHandle.isValid())) {
        if (cfd->active) {
            context.errors.push_back("CFD is missing its Voronoi domain product.");
            return false;
        }
        domainPackage = nullptr;
    }

    if (domainPackage) {
        package.domainVoronoiProduct = domainPackage->productHandle;
        package.displayDependencyKeys.push_back(domainPackage->productHandle.outputSocketKey);
        for (const ProductHandle& handle : domainPackage->globalRemeshProducts) {
            package.displayDependencyKeys.push_back(handle.outputSocketKey);
        }
    }

    if (cfd->hasFluid && cfd->fluidHandle.key != 0) {
        const FluidData* fluid = context.payloads.get<FluidData>(cfd->fluidHandle);
        if (!fluid) {
            if (cfd->active) {
                context.errors.push_back("CFD cannot resolve its Fluid input.");
                return false;
            }
        } else {
            package.fluid.density = fluid->density;
            package.fluid.kinematicViscosity = fluid->kinematicViscosity;
        }
    }

    sealCFDPackage(package, context.output.hashes);
    context.packages.apply<CFDPackage>(context.outputSocketKey, std::move(package));
    return true;
}

}
