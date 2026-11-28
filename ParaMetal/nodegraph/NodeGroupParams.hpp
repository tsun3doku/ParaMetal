#pragma once

#include "NodeGraphTypes.hpp"
#include "mesh/MeshElementType.hpp"

#include <cstdint>
#include <string>

struct NodeGraphNode;

enum class GroupOperation : int64_t {
    Replace = 0,
    Union = 1,
    Subtract = 2,
    Intersect = 3
};

struct NodeGroupParams {
    MeshElementType elementType = MeshElementType::Face;
    std::string groupName = "Group";
    GroupOperation operation = GroupOperation::Replace;
    bool useBaseGroup = false;
    std::string selectedElements;
    std::string selectionSignature;
};

NodeGroupParams readNodeGroupParams(const NodeGraphNode& node);

namespace nodegraphparams::group {
inline constexpr uint32_t ElementType = 2;
inline constexpr uint32_t GroupName = 3;
inline constexpr uint32_t Operation = 4;
inline constexpr uint32_t SelectedElements = 6;
inline constexpr uint32_t SelectionSignature = 7;
inline constexpr uint32_t UseBaseGroup = 8;
inline constexpr uint32_t ActionSelectElements = 1;
}
