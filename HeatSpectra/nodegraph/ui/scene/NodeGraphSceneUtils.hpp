#pragma once

#include "nodegraph/NodeGraphTypes.hpp"

#include <vector>

namespace nodegraphsceneutils {

int findSocketIndexById(const std::vector<NodeGraphSocket>& sockets, NodeGraphSocketId socketId);
NodeGraphSocketId socketByIndex(const std::vector<NodeGraphSocket>& sockets, int index);
std::vector<NodeGraphSocketId> matchingInputSocketsByType(const NodeGraphNode& node, NodeGraphValueType valueType);
int valueTypeOrdinalAtInputIndex(const NodeGraphNode& node, int inputIndex);

} 
