#pragma once

class NodeGraphRegistry;
class NodeGraphKernels;

bool registerNodePayloads(NodeGraphRegistry& registry);
bool registerNodeDefinitions(NodeGraphRegistry& registry);
bool registerNodeKernels(NodeGraphKernels& kernels);
