#include "NodeModel.hpp"
#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeModelParams.hpp"
#include "NodePayloadRegistry.hpp"
#include "scene/MeshImporter.hpp"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <cmath>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct ImportedModel {
    GeometryData geometry;
    uint64_t geometryHash = 0;
    uint64_t groupHash = 0;
    std::filesystem::file_time_type writeTime{};
    std::uintmax_t fileSize = 0;
};

static std::unordered_map<std::string, ImportedModel> importedModels;

static std::vector<std::string> resolveCandidateModelPaths(const std::string& modelPath);
static bool parseMeshGeometry(const std::string& modelPath, GeometryData& geometry);
static const ImportedModel* findOrLoadImportedModel(const std::string& modelPath);

static bool readFileMetadata(
    const std::string& path,
    std::filesystem::file_time_type& outWriteTime,
    std::uintmax_t& outFileSize) {
    std::error_code error;
    outWriteTime = std::filesystem::last_write_time(path, error);
    if (error) {
        return false;
    }
    outFileSize = std::filesystem::file_size(path, error);
    return !error;
}

static uint64_t hashGeometryContent(const GeometryData& geometry) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combineString(hash, "model.geometry.v1");
    HashBuilder::combinePodVector(hash, geometry.vertexPositions);
    HashBuilder::combinePodVector(hash, geometry.triangleIndices);
    HashBuilder::combinePodVector(hash, geometry.renderPositions);
    HashBuilder::combinePodVector(hash, geometry.renderNormals);
    HashBuilder::combinePodVector(hash, geometry.renderTexcoords);
    HashBuilder::combinePodVector(hash, geometry.renderIndices);
    return hash;
}

static uint64_t hashGroupContent(const GeometryData& geometry) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combineString(hash, "model.groups.v1");
    HashBuilder::combine(hash, static_cast<uint64_t>(geometry.groups.size()));
    for (const MeshGroup& group : geometry.groups) {
        HashBuilder::combineString(hash, group.key);
        HashBuilder::combineString(hash, group.name);
        HashBuilder::combineString(hash, group.source);
        HashBuilder::combine(hash, static_cast<uint64_t>(group.elementType));
        HashBuilder::combine(hash, static_cast<uint64_t>(group.colorRGBA));
        HashBuilder::combine(hash, static_cast<uint64_t>(group.elements.size()));
        for (uint64_t element : group.elements) {
            HashBuilder::combine(hash, element);
        }
    }
    return hash;
}

static bool validateImportedGeometry(const GeometryData& geometry, std::string& outError) {
    if (geometry.vertexPositions.empty() || geometry.vertexPositions.size() % 3 != 0) {
        outError = "Imported geometry has incomplete vertex positions.";
        return false;
    }
    if (geometry.triangleIndices.empty() || geometry.triangleIndices.size() % 3 != 0) {
        outError = "Imported geometry has incomplete triangle indices.";
        return false;
    }

    const uint64_t vertexCount = geometry.vertexPositions.size() / 3;
    for (uint32_t index : geometry.triangleIndices) {
        if (index >= vertexCount) {
            outError = "Imported geometry references a missing vertex.";
            return false;
        }
    }

    const bool hasRenderData =
        !geometry.renderPositions.empty() ||
        !geometry.renderNormals.empty() ||
        !geometry.renderTexcoords.empty() ||
        !geometry.renderIndices.empty();
    if (hasRenderData) {
        const std::size_t renderVertexCount = geometry.renderPositions.size() / 3;
        if (geometry.renderPositions.empty() ||
            geometry.renderPositions.size() % 3 != 0 ||
            geometry.renderNormals.size() != geometry.renderPositions.size() ||
            geometry.renderTexcoords.size() != renderVertexCount * 2 ||
            geometry.renderIndices.size() != geometry.triangleIndices.size()) {
            outError = "Imported render mesh data is inconsistent.";
            return false;
        }
        for (uint32_t index : geometry.renderIndices) {
            if (index >= renderVertexCount) {
                outError = "Imported render mesh references a missing render vertex.";
                return false;
            }
        }
    }

    const uint64_t triangleCount = geometry.triangleIndices.size() / 3;
    std::unordered_set<std::string> groupKeys;
    for (const MeshGroup& group : geometry.groups) {
        if (group.key.empty() || !groupKeys.insert(group.key).second) {
            outError = "Imported geometry has an empty or duplicate group key.";
            return false;
        }
        if (group.elementType != MeshElementType::Face) {
            outError = "Imported geometry contains an unsupported group element type.";
            return false;
        }
        if (!std::is_sorted(group.elements.begin(), group.elements.end()) ||
            std::adjacent_find(group.elements.begin(), group.elements.end()) !=
                group.elements.end()) {
            outError = "Imported group membership is unsorted or duplicated.";
            return false;
        }
        for (uint64_t element : group.elements) {
            if (element >= triangleCount) {
                outError = "Imported group references a missing face.";
                return false;
            }
        }
    }
    return true;
}

static NodeTypeDefinition buildModelNode() {
    return {
        nodegraphtypes::Model,
        "Model",
        NodeGraphNodeCategory::Geometry,
        {
            makeOutputSocket("Mesh", NodeGraphValueType::Mesh, payloadtypes::Geometry, true),
        },
        {
            {nodegraphparams::model::Path, "Model Path", NodeGraphParamType::String, 0.0, 0, false, "", false},
            makeEnumParamDefinition(nodegraphparams::model::SourceUnit, "Import Units", "Meters", {"Meters", "Centimeters", "Millimeters"}),
        },
    };
}

NodeTypeDefinition buildModelDefinition() {
    return buildModelNode();
}

std::unique_ptr<NodeKernel> createModelKernel() {
    return std::make_unique<NodeModel>();
}

const char* NodeModel::typeId() const {
    return nodegraphtypes::Model;
}

void NodeModel::execute(NodeKernelEval& eval) const {
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    const ModelNodeParams params = readModelNodeParams(eval.node);
    const std::string& modelPath = params.path;
    GeometryData geometry{};
    bool hasGeometry = false;
    if (!modelPath.empty()) {
        const ImportedModel* imported = findOrLoadImportedModel(modelPath);
        if (!imported) {
            eval.error = "Model geometry could not be loaded or validated.";
            return;
        }
        geometry = imported->geometry;
        hasGeometry = true;
        const float sourceToCanonical = units::sourceToCanonicalScale(params.sourceUnit);
        for (float& coordinate : geometry.vertexPositions) {
            coordinate *= sourceToCanonical;
        }
        for (float& coordinate : geometry.renderPositions) {
            coordinate *= sourceToCanonical;
        }
        const std::string nodePrefix =
            "model:" + std::to_string(eval.node.id.value) + ":";
        for (std::size_t index = 0; index < geometry.groups.size(); ++index) {
            MeshGroup& group = geometry.groups[index];
            constexpr const char* importPrefix = "import:";
            if (group.key.rfind(importPrefix, 0) != 0) {
                eval.error = "Model import produced an invalid group key.";
                return;
            }
            group.key = nodePrefix + group.key.substr(
                std::char_traits<char>::length(importPrefix));
        }
    }

    for (std::size_t outputIndex = 0;
         outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry ||
            outputValue.dataType != payloadtypes::Geometry ||
            !hasGeometry) {
            continue;
        }

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(geometry), eval.outputHashes);
        if (outputValue.payloadHandle.key == 0) {
            eval.error = "Payload publication backpressured.";
            return;
        }
    }
}

HashValues NodeModel::computeOutputHashes(const NodeKernelHash& hash) const {
    const ModelNodeParams params = readModelNodeParams(hash.node);
    const ImportedModel* imported = params.path.empty()
        ? nullptr
        : findOrLoadImportedModel(params.path);
    const uint64_t sourceGeometryHash = imported ? imported->geometryHash : 0;
    const uint64_t sourceGroupHash = imported ? imported->groupHash : 0;

    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combineString(geometryHash, nodegraphtypes::Model);
    HashBuilder::combineString(geometryHash, params.path);
    HashBuilder::combine(geometryHash, static_cast<uint64_t>(params.sourceUnit));
    HashBuilder::combine(geometryHash, sourceGeometryHash);

    uint64_t propertiesHash = HashBuilder::start();
    HashBuilder::combineString(propertiesHash, "model.groups");
    HashBuilder::combine(propertiesHash, static_cast<uint64_t>(hash.node.id.value));
    HashBuilder::combineString(propertiesHash, params.path);
    HashBuilder::combine(propertiesHash, static_cast<uint64_t>(params.sourceUnit));
    HashBuilder::combine(propertiesHash, sourceGroupHash);

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, geometryHash);
    HashBuilder::combine(fullHash, propertiesHash);

    HashValues values{};
    values.full = fullHash;
    values.geometry = geometryHash;
    values.properties = propertiesHash;
    values.simulation = fullHash;
    return values;
}

static bool parseMeshGeometry(const std::string& modelPath, GeometryData& geometry) {
    geometry = {};

    MeshImporter::Mesh importedMesh;
    if (!MeshImporter::loadMesh(modelPath, importedMesh) || !importedMesh.isValid()) {
        return false;
    }
    if (importedMesh.triangleCornerIndices.size() !=
        importedMesh.triangleGroupIds.size() * 3) {
        return false;
    }

    geometry.vertexPositions = importedMesh.positions;
    geometry.triangleIndices.reserve(importedMesh.triangleCornerIndices.size());
    for (uint32_t cornerIndex : importedMesh.triangleCornerIndices) {
        geometry.triangleIndices.push_back(importedMesh.corners[cornerIndex].vertexIndex);
    }

    geometry.renderPositions.reserve(importedMesh.triangleCornerIndices.size() * 3);
    geometry.renderNormals.reserve(importedMesh.triangleCornerIndices.size() * 3);
    geometry.renderTexcoords.reserve(importedMesh.triangleCornerIndices.size() * 2);
    geometry.renderIndices.reserve(importedMesh.triangleCornerIndices.size());

    for (uint32_t cornerIndex : importedMesh.triangleCornerIndices) {
        const MeshImporter::Corner& corner = importedMesh.corners[cornerIndex];
        const uint32_t renderIndex = static_cast<uint32_t>(geometry.renderPositions.size() / 3);
        const std::size_t positionOffset = static_cast<std::size_t>(corner.vertexIndex) * 3;
        geometry.renderPositions.insert(
            geometry.renderPositions.end(),
            importedMesh.positions.begin() + positionOffset,
            importedMesh.positions.begin() + positionOffset + 3);

        if (corner.normalIndex >= 0) {
            const std::size_t normalOffset = static_cast<std::size_t>(corner.normalIndex) * 3;
            float nx = importedMesh.normals[normalOffset + 0];
            float ny = importedMesh.normals[normalOffset + 1];
            float nz = importedMesh.normals[normalOffset + 2];
            const float lengthSquared = nx * nx + ny * ny + nz * nz;
            if (lengthSquared > 1e-12f) {
                const float inverseLength = 1.0f / std::sqrt(lengthSquared);
                nx *= inverseLength;
                ny *= inverseLength;
                nz *= inverseLength;
            } else {
                nx = 0.0f;
                ny = 0.0f;
                nz = 1.0f;
            }
            geometry.renderNormals.insert(
                geometry.renderNormals.end(), {nx, ny, nz});
        } else {
            geometry.renderNormals.insert(
                geometry.renderNormals.end(), {0.0f, 0.0f, 0.0f});
        }

        if (corner.texcoordIndex >= 0) {
            const std::size_t texcoordOffset = static_cast<std::size_t>(corner.texcoordIndex) * 2;
            geometry.renderTexcoords.push_back(importedMesh.texcoords[texcoordOffset + 0]);
            geometry.renderTexcoords.push_back(1.0f - importedMesh.texcoords[texcoordOffset + 1]);
        } else {
            geometry.renderTexcoords.insert(
                geometry.renderTexcoords.end(), {0.0f, 0.0f});
        }

        geometry.renderIndices.push_back(renderIndex);
    }

    assert(geometry.renderIndices.size() == geometry.triangleIndices.size());
    assert(geometry.renderIndices.size() % 3 == 0);
    assert(geometry.renderPositions.size() / 3 == geometry.renderIndices.size());

    geometry.groups.reserve(importedMesh.groups.size());
    std::unordered_map<uint32_t, std::size_t> groupIndexById;
    for (const auto& group : importedMesh.groups) {
        groupIndexById.emplace(group.id, geometry.groups.size());
        MeshGroup g{};
        g.key = "import:" + std::to_string(group.id);
        g.name = group.name;
        g.source = group.source;
        g.elementType = MeshElementType::Face;
        geometry.groups.push_back(std::move(g));
    }

    const std::size_t triangleCount = importedMesh.triangleGroupIds.size();
    for (std::size_t triangle = 0; triangle < triangleCount; ++triangle) {
        const auto it = groupIndexById.find(importedMesh.triangleGroupIds[triangle]);
        if (it == groupIndexById.end()) {
            return false;
        }
        geometry.groups[it->second].elements.push_back(static_cast<uint64_t>(triangle));
    }

    return !geometry.triangleIndices.empty();
}

static const ImportedModel* findOrLoadImportedModel(const std::string& modelPath) {
    for (const std::string& candidatePath : resolveCandidateModelPaths(modelPath)) {
        std::filesystem::file_time_type writeTime{};
        std::uintmax_t fileSize = 0;
        if (!readFileMetadata(candidatePath, writeTime, fileSize)) {
            continue;
        }

        const auto cacheIt = importedModels.find(candidatePath);
        if (cacheIt != importedModels.end()) {
            if (cacheIt->second.writeTime == writeTime &&
                cacheIt->second.fileSize == fileSize) {
                return &cacheIt->second;
            }
            importedModels.erase(cacheIt);
        }

        GeometryData candidateGeometry;
        std::string validationError;
        if (!parseMeshGeometry(candidatePath, candidateGeometry) ||
            !validateImportedGeometry(candidateGeometry, validationError)) {
            continue;
        }
        candidateGeometry.baseModelPath = candidatePath;

        ImportedModel entry{};
        entry.geometry = std::move(candidateGeometry);
        entry.geometryHash = hashGeometryContent(entry.geometry);
        entry.groupHash = hashGroupContent(entry.geometry);
        entry.writeTime = writeTime;
        entry.fileSize = fileSize;
        const auto inserted = importedModels.emplace(candidatePath, std::move(entry));
        return &inserted.first->second;
    }
    return nullptr;
}

static std::vector<std::string> resolveCandidateModelPaths(const std::string& modelPath) {
    std::vector<std::string> candidates;
    if (modelPath.empty()) {
        return candidates;
    }

    std::unordered_set<std::string> seenPaths;
    auto addCandidate = [&](const std::filesystem::path& path) {
        const std::string candidate = path.lexically_normal().string();
        if (!candidate.empty() && seenPaths.insert(candidate).second) {
            candidates.push_back(candidate);
        }
    };

    const std::filesystem::path rawPath(modelPath);
    addCandidate(rawPath);

    if (!rawPath.is_absolute()) {
        const std::filesystem::path currentPath = std::filesystem::current_path();
        addCandidate(currentPath / rawPath);
        addCandidate(currentPath / "ParaMetal" / rawPath);
    }

    return candidates;
}
