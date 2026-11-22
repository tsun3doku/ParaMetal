#include "NodeGraphInit.hpp"

#include "NodeGraphModules.hpp"
#include "NodeGraphRegistry.hpp"

bool initNodeGraph(NodeGraphRegistry& registry) {
    if (!registerNodePayloads(registry)) {
        return false;
    }
    if (!registerNodeDefinitions(registry)) {
        return false;
    }
    return registry.isValid();
}
