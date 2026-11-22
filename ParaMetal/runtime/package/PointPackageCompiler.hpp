#pragma once

class RuntimeProductManager;

namespace runtimepackage {

struct PackageCompileContext;

bool compilePointPackage(
    const PackageCompileContext& context,
    const RuntimeProductManager& products);

}
