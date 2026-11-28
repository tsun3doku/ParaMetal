#pragma once

class NodeGraphRuntime;
class NodePayloadRegistry;

namespace runtimepackage {

struct PackageCompileContext;

bool compileRemeshPackage(const PackageCompileContext& context);
bool preflightRemeshPackage(
    const PackageCompileContext& context,
    NodeGraphRuntime& graphRuntime,
    NodePayloadRegistry& payloads,
    bool& backpressured);
bool publishRemeshPackage(
    const PackageCompileContext& context,
    NodeGraphRuntime& graphRuntime,
    NodePayloadRegistry& payloads);

}
