#include "NodeGraphUiModel.hpp"

#include "nodegraph/NodeGraphUtils.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeHeatMaterialPresets.hpp"
#include "nodegraph/NodeParamUtils.hpp"
#include "nodegraph/NodeGraphParamUtils.hpp"
#include "nodegraph/NodeWriteParams.hpp"
#include "serial/SerialPort.hpp"

#include <QGuiApplication>
#include <QCursor>
#include <QPixmap>
#include <QCoreApplication>
#include <algorithm>
#include <unordered_set>
#include <cctype>

static QString nodeCategoryName(NodeGraphNodeCategory category) {
    switch (category) {
    case NodeGraphNodeCategory::Geometry: return QStringLiteral("Geometry");
    case NodeGraphNodeCategory::Meshing:  return QStringLiteral("Meshing");
    case NodeGraphNodeCategory::System:   return QStringLiteral("System");
    case NodeGraphNodeCategory::Misc:     return QStringLiteral("Misc");
    }
    return QStringLiteral("Misc");
}

static int nodeCategoryOrder(NodeGraphNodeCategory category) {
    switch (category) {
    case NodeGraphNodeCategory::Geometry: return 0;
    case NodeGraphNodeCategory::Meshing:  return 1;
    case NodeGraphNodeCategory::System:   return 2;
    case NodeGraphNodeCategory::Misc:     return 3;
    }
    return 3;
}

NodeGraphUiModel::NodeGraphUiModel(QObject* parent)
    : QAbstractListModel(parent) {
    refreshSerialPorts();
}

NodeGraphUiModel::~NodeGraphUiModel() {
    if (m_probePickingActive) {
        m_probePickingActive = false;
        QGuiApplication::restoreOverrideCursor();
    }
}

int NodeGraphUiModel::rowCount(const QModelIndex& parent) const {
    return parent.isValid() ? 0 : static_cast<int>(nodes.size());
}

QVariant NodeGraphUiModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() < 0 || index.row() >= static_cast<int>(nodes.size())) {
        return {};
    }

    const NodeGraphNode& node = nodes[static_cast<std::size_t>(index.row())];
    switch (role) {
    case NodeIdRole: return static_cast<int>(node.id.value);
    case TitleRole: return QString::fromStdString(node.title);
    case TypeRole: return QString::fromStdString(node.typeId);
    case NodeXRole: return node.x;
    case NodeYRole: return node.y;
    case InputsRole: return socketList(node.id, node.inputs);
    case OutputsRole: return socketList(node.id, node.outputs);
    case DisplayEnabledRole: return node.state.isPrimaryDisplay();
    case FrozenRole: return node.state.isFrozen();
    case SelectedRole:
        return std::find(selectedIds.begin(), selectedIds.end(), node.id.value) != selectedIds.end();
    default: return {};
    }
}

QHash<int, QByteArray> NodeGraphUiModel::roleNames() const {
    return {
        {NodeIdRole, "nodeId"},
        {TitleRole, "title"},
        {TypeRole, "typeId"},
        {NodeXRole, "nodeX"},
        {NodeYRole, "nodeY"},
        {InputsRole, "inputs"},
        {OutputsRole, "outputs"},
        {DisplayEnabledRole, "displayEnabled"},
        {FrozenRole, "frozen"},
        {SelectedRole, "selected"}
    };
}

QVariantList NodeGraphUiModel::edges() const {
    return edgeItems;
}

QVariantList NodeGraphUiModel::nodeCategories() const {
    return categoryItems;
}

int NodeGraphUiModel::selectedNodeId() const {
    return selectedId;
}

bool NodeGraphUiModel::isNodeSelected(int nodeId) const {
    if (nodeId <= 0) return false;
    return std::find(selectedIds.begin(), selectedIds.end(), static_cast<uint32_t>(nodeId)) != selectedIds.end();
}

QString NodeGraphUiModel::selectedNodeTitle() const {
    return selectedTitle;
}

QString NodeGraphUiModel::selectedNodeType() const {
    return selectedType;
}

QString NodeGraphUiModel::selectedNodeDescription() const {
    if (selectedType == QStringLiteral("model")) return QStringLiteral("Choose a 3D model");
    if (selectedType == QStringLiteral("transform")) return QStringLiteral("Adjust placement, orientation and scale of a 3D model");
    if (selectedType == QStringLiteral("group")) return QStringLiteral("Target source groups and write grouped mesh selections");
    if (selectedType == QStringLiteral("remesh")) return QStringLiteral("Intrinsically remesh an underlying 3D model while preserving its shape");
    if (selectedType == QStringLiteral("voronoi")) return QStringLiteral("Generate a volumetric Voronoi domain");
    if (selectedType == QStringLiteral("heat_model")) return QStringLiteral("Assign thermal properties and boundary conditions to a 3D model");
    if (selectedType == QStringLiteral("heat_solve")) return QStringLiteral("Simulate the transient transfer of heat between 3D geometry");
    if (selectedType == QStringLiteral("serial_temperature")) return QStringLiteral("Read a live Celsius value from a serial temperature sensor");
    if (selectedType == QStringLiteral("heat_probe")) return QStringLiteral("Sample surface temperature and stream live telemetry");
    if (selectedType == QStringLiteral("points")) return QStringLiteral("Generate a point cloud domain");
    if (selectedType == QStringLiteral("mesh_points")) return QStringLiteral("Extract mesh surface vertices as a point cloud");
    return selectedId > 0 ? QStringLiteral("Inspect this node's parameters and generated data") :
                            QStringLiteral("Select a node in the graph to inspect parameters and dataflow");
}

QVariantList NodeGraphUiModel::selectedNodeParameters() const {
    return selectedParameters;
}

QVariantList NodeGraphUiModel::serialPorts() const {
    return serialPortItems;
}

void NodeGraphUiModel::initializeGraph(
    const NodeGraphState& state,
    const std::vector<NodeTypeDefinition>& definitions) {
    typeDefinitions = definitions;
    graphState = state;
    rebuildNodeCategories();
    revision = UINT64_MAX;
    refresh();
}

void NodeGraphUiModel::replaceGraphState(const NodeGraphState& state) {
    graphState = state;
    revision = UINT64_MAX;
    refresh();
    relabelExportProducts();
    if (selectedType == QStringLiteral("write")) requestCurrentExportProducts();
}

void NodeGraphUiModel::applyDelta(const NodeGraphDelta& delta) {
    bool structural = false;
    for (const NodeGraphChange& change : delta.changes) {
        if (change.reason == NodeGraphChangeReason::Topology &&
            (change.type == NodeGraphChangeType::NodeUpsert ||
             change.type == NodeGraphChangeType::NodeRemoved ||
             change.type == NodeGraphChangeType::EdgeUpsert ||
             change.type == NodeGraphChangeType::EdgeRemoved ||
             change.type == NodeGraphChangeType::Reset)) structural = true;
    }
    const bool applied = applyNodeGraphDelta(graphState, delta);
    Q_ASSERT(applied);
    if (!applied) return;
    refresh();
    relabelExportProducts();
    if (structural && selectedType == QStringLiteral("write")) requestCurrentExportProducts();
    else if (selectedType == QStringLiteral("write")) emit exportProductsChanged();
}

void NodeGraphUiModel::handleNodesPasted(const std::vector<NodeGraphNodeId>& nodeIds) {
    selectedIds.clear();
    for (NodeGraphNodeId id : nodeIds) selectedIds.push_back(id.value);
    notifySelectionChanged(true);
}

void NodeGraphUiModel::setRuntimeSelectedNodeId(int nodeId) {
    const uint32_t runtimeNodeId = nodeId > 0 ? static_cast<uint32_t>(nodeId) : 0;
    if ((runtimeNodeId == 0 && selectedIds.empty()) ||
        (runtimeNodeId != 0 && selectedIds.size() == 1 && selectedIds.front() == runtimeNodeId)) {
        return;
    }
    selectedIds.clear();
    if (runtimeNodeId != 0) {
        selectedIds.push_back(runtimeNodeId);
    }
    notifySelectionChanged(false);
}

void NodeGraphUiModel::refresh() {
    const NodeGraphState& updatedState = graphState;
    const uint64_t updatedRevision = graphState.revision;
    if (updatedRevision == revision) {
        return;
    }

    std::vector<NodeGraphNode> updatedNodes;
    updatedNodes.reserve(updatedState.nodes.size());
    for (const auto& entry : updatedState.nodes) {
        updatedNodes.push_back(entry.second);
    }
    std::sort(updatedNodes.begin(), updatedNodes.end(), [](const NodeGraphNode& lhs, const NodeGraphNode& rhs) {
        return lhs.id.value < rhs.id.value;
    });

    QVariantList updatedEdges;
    updatedEdges.reserve(static_cast<qsizetype>(updatedState.edges.size()));
    for (const auto& entry : updatedState.edges) {
        const NodeGraphEdge& edge = entry.second;
        QVariantMap item;
        item.insert(QStringLiteral("edgeId"), static_cast<int>(edge.id.value));
        item.insert(QStringLiteral("fromNode"), static_cast<int>(edge.fromNode.value));
        item.insert(QStringLiteral("fromSocket"), static_cast<int>(edge.fromSocket.value));
        item.insert(QStringLiteral("toNode"), static_cast<int>(edge.toNode.value));
        item.insert(QStringLiteral("toSocket"), static_cast<int>(edge.toSocket.value));
        item.insert(QStringLiteral("valueType"), static_cast<int>(
            socketType(updatedState, edge.fromNode, edge.fromSocket)));
        updatedEdges.push_back(item);
    }

    beginResetModel();
    nodes = updatedNodes;
    edgeItems = updatedEdges;
    revision = updatedRevision;
    endResetModel();
    selectedIds.erase(std::remove_if(selectedIds.begin(), selectedIds.end(), [this](uint32_t nodeId) {
        return graphState.nodes.find(nodeId) == graphState.nodes.end();
    }), selectedIds.end());
    if (std::find(selectedIds.begin(), selectedIds.end(), static_cast<uint32_t>(selectedId)) == selectedIds.end()) {
        selectedId = selectedIds.empty() ? 0 : static_cast<int>(selectedIds.back());
    }
    rebuildSelectedNode();
    emit edgesChanged();
}

void NodeGraphUiModel::moveNode(int nodeId, qreal x, qreal y) {
    if (nodeId > 0) emit moveNodeRequested(NodeGraphNodeId{static_cast<uint32_t>(nodeId)}, static_cast<float>(x), static_cast<float>(y));
}

void NodeGraphUiModel::removeNode(int nodeId) {
    if (nodeId > 0) {
        selectedIds.erase(std::remove(selectedIds.begin(), selectedIds.end(), static_cast<uint32_t>(nodeId)), selectedIds.end());
        notifySelectionChanged(true);
        emit removeNodeRequested(NodeGraphNodeId{static_cast<uint32_t>(nodeId)});
    }
}

void NodeGraphUiModel::startProbePicking(int nodeId, int probeIndex) {
    if (!m_probePickingActive) {
        m_probePickingActive = true;
        emit probePickingActiveChanged();

        const QString iconPath = QCoreApplication::applicationDirPath() + QStringLiteral("/textures/icons/Marker/marker_a/32w/Artboard 1.png");
        QPixmap markerPixmap(iconPath);
        if (!markerPixmap.isNull()) {
            QGuiApplication::setOverrideCursor(QCursor(markerPixmap, 15, 29));
        } else {
            QGuiApplication::setOverrideCursor(QCursor(Qt::CrossCursor));
        }
    }
    emit probePickingRequested(nodeId, probeIndex);
}

void NodeGraphUiModel::cancelProbePicking() {
    if (m_probePickingActive) {
        m_probePickingActive = false;
        emit probePickingActiveChanged();
        QGuiApplication::restoreOverrideCursor();
    }
    emit probePickingRequested(0, 0);
}

void NodeGraphUiModel::removeSelectedNodes() {
    const std::vector<uint32_t> ids = selectedIds;
    for (uint32_t nodeId : ids) {
        emit removeNodeRequested(NodeGraphNodeId{nodeId});
    }
    selectedIds.clear();
    notifySelectionChanged(true);
}

void NodeGraphUiModel::resetToDefaultGraph() {
    emit resetRequested();
}

int NodeGraphUiModel::addNode(const QString& typeId, qreal x, qreal y) {
    emit addNodeRequested(typeId, static_cast<float>(x), static_cast<float>(y));
    return 0;
}

void NodeGraphUiModel::toggleNodeDisplay(int nodeId) {
    if (nodeId > 0) emit toggleNodeDisplayRequested(NodeGraphNodeId{static_cast<uint32_t>(nodeId)});
}

void NodeGraphUiModel::toggleNodeFrozen(int nodeId) {
    if (nodeId > 0) emit toggleNodeFrozenRequested(NodeGraphNodeId{static_cast<uint32_t>(nodeId)});
}

void NodeGraphUiModel::setSelectedNodeId(int nodeId) {
    setNodeSelected(nodeId, false);
}

void NodeGraphUiModel::setNodeSelected(int nodeId, bool additive) {
    const uint32_t id = nodeId > 0 ? static_cast<uint32_t>(nodeId) : 0;
    if (!additive) {
        selectedIds.clear();
        if (id != 0) {
            selectedIds.push_back(id);
        }
    } else if (id != 0) {
        const auto it = std::find(selectedIds.begin(), selectedIds.end(), id);
        if (it == selectedIds.end()) {
            selectedIds.push_back(id);
        } else {
            selectedIds.erase(it);
        }
    }
    if (id != 0 && std::find(selectedIds.begin(), selectedIds.end(), id) != selectedIds.end()) {
        selectedId = static_cast<int>(id);
    } else if (selectedId == static_cast<int>(id)) {
        selectedId = selectedIds.empty() ? 0 : static_cast<int>(selectedIds.back());
    }
    notifySelectionChanged(true);
}

void NodeGraphUiModel::copySelectedNodes() {
    copiedNodes.clear();
    copiedEdges.clear();
    if (selectedIds.empty()) {
        return;
    }

    const std::unordered_set<uint32_t> selectedSet(selectedIds.begin(), selectedIds.end());
    for (uint32_t nodeId : selectedIds) {
        const auto nodeIt = graphState.nodes.find(nodeId);
        if (nodeIt == graphState.nodes.end()) {
            continue;
        }
        const NodeGraphNode& node = nodeIt->second;
        NodeGraphEditor::CopiedNode copy{};
        copy.sourceNodeId = node.id;
        copy.typeId = node.typeId;
        copy.title = node.title;
        copy.x = node.x;
        copy.y = node.y;
        copy.parameters = node.parameters;
        for (const NodeGraphSocket& socket : node.outputs) {
            copy.outputSocketIds.push_back(socket.id);
        }
        copiedNodes.push_back(copy);
    }

    for (const auto& entry : graphState.edges) {
        const NodeGraphEdge& edge = entry.second;
        if (selectedSet.find(edge.fromNode.value) == selectedSet.end() ||
            selectedSet.find(edge.toNode.value) == selectedSet.end()) {
            continue;
        }
        copiedEdges.push_back({edge.fromNode, edge.fromSocket, edge.toNode, edge.toSocket});
    }
}

void NodeGraphUiModel::pasteCopiedNodes() {
    if (copiedNodes.empty()) return;
    GraphPastePayload payload{};
    payload.nodes = copiedNodes;
    payload.edges = copiedEdges;
    emit pasteRequested(payload);
}

bool NodeGraphUiModel::connectSockets(int fromNode, int fromSocket, int toNode, int toSocket) {
    if (fromNode <= 0 || fromSocket <= 0 || toNode <= 0 || toSocket <= 0) {
        return false;
    }
    emit connectSocketsRequested(
        NodeGraphNodeId{static_cast<uint32_t>(fromNode)},
        NodeGraphSocketId{static_cast<uint32_t>(fromSocket)},
        NodeGraphNodeId{static_cast<uint32_t>(toNode)},
        NodeGraphSocketId{static_cast<uint32_t>(toSocket)});
    return true;
}

void NodeGraphUiModel::removeConnection(int edgeId) {
    if (edgeId > 0) emit removeConnectionRequested(NodeGraphEdgeId{static_cast<uint32_t>(edgeId)});
}

void NodeGraphUiModel::setParameterValue(int parameterId, const QVariant& value) {
    if (selectedId <= 0 || parameterId <= 0) {
        return;
    }

    const NodeGraphNodeId nodeId{static_cast<uint32_t>(selectedId)};
    const NodeGraphNode* selectedNode = graphState.node(nodeId);
    if (!selectedNode) return;

    for (NodeGraphParamValue parameter : selectedNode->parameters) {
        if (parameter.id != static_cast<uint32_t>(parameterId)) {
            continue;
        }
        switch (parameter.type) {
        case NodeGraphParamType::Float: parameter.floatValue = value.toDouble(); break;
        case NodeGraphParamType::Int: parameter.intValue = value.toLongLong(); break;
        case NodeGraphParamType::Bool: parameter.boolValue = value.toBool(); break;
        case NodeGraphParamType::String: parameter.stringValue = value.toString().toStdString(); break;
        case NodeGraphParamType::Enum: parameter.enumValue = value.toString().toStdString(); break;
        case NodeGraphParamType::Array:
            return;
        default: return;
        }
        emit setParameterRequested(nodeId, parameter);
        return;
    }
}

void NodeGraphUiModel::setHeatMaterialPreset(const QString& presetName) {
    if (selectedType != QStringLiteral("heat_model")) return;
    HeatMaterialPresetId presetId = HeatMaterialPresetId::Custom;
    if (presetName == QStringLiteral("Aluminum")) presetId = HeatMaterialPresetId::Aluminum;
    else if (presetName == QStringLiteral("Copper")) presetId = HeatMaterialPresetId::Copper;
    else if (presetName == QStringLiteral("Iron")) presetId = HeatMaterialPresetId::Iron;
    else if (presetName == QStringLiteral("Ceramic")) presetId = HeatMaterialPresetId::Ceramic;

    setParameterValue(nodegraphparams::heatmodel::MaterialPreset, presetName);
    if (presetId == HeatMaterialPresetId::Custom) return;
    const HeatMaterialPreset& preset = heatMaterialPresetById(presetId);
    setParameterValue(nodegraphparams::heatmodel::Density, preset.density);
    setParameterValue(nodegraphparams::heatmodel::SpecificHeat, preset.specificHeat);
    setParameterValue(nodegraphparams::heatmodel::Conductivity, preset.conductivity);
}

void NodeGraphUiModel::refreshSerialPorts() {
    QVariantList updated;
    for (const SerialPortInfo& port : SerialPort::enumeratePorts()) {
        QVariantMap item;
        item.insert(QStringLiteral("text"), QString::fromStdString(port.displayName));
        item.insert(QStringLiteral("value"), QString::fromStdString(port.portName));
        updated.push_back(item);
    }
    if (updated != serialPortItems) {
        serialPortItems = updated;
        emit serialPortsChanged();
    }
}

QString NodeGraphUiModel::nodeTitle(int nodeId) const {
    const NodeGraphNode* node = nodeId > 0
        ? graphState.node(NodeGraphNodeId{static_cast<uint32_t>(nodeId)}) : nullptr;
    return node ? QString::fromStdString(node->title) : QString{};
}

QString NodeGraphUiModel::selectedWriteFile() const {
    const NodeGraphNode* node = selectedId > 0
        ? graphState.node(NodeGraphNodeId{static_cast<uint32_t>(selectedId)}) : nullptr;
    return node && node->typeId == nodegraphtypes::Write
        ? QString::fromStdString(readWriteNodeParams(*node).filePath) : QString{};
}

QStringList NodeGraphUiModel::exportProductLabels() const {
    QStringList labels;
    labels.reserve(exportProductItems.size());
    for (const QVariant& value : exportProductItems)
        labels.push_back(value.toMap().value("displayLabel").toString());
    return labels;
}

int NodeGraphUiModel::selectedExportProductIndex() const {
    const WriteProductSelection selection = currentWriteSelection();
    if (selection.empty()) return -1;
    for (int i = 0; i < exportProductItems.size(); ++i) {
        const QVariantMap row = exportProductItems[i].toMap();
        if (row.value("producerNodeId").toInt() == static_cast<int>(selection.producerNodeId) &&
            row.value("outputSocketId").toInt() == static_cast<int>(selection.outputSocketId) &&
            row.value("productId").toString() == QString::fromStdString(selection.productId))
            return i;
    }
    return -1;
}

WriteProductSelection NodeGraphUiModel::currentWriteSelection() const {
    const NodeGraphNode* node = selectedId > 0
        ? graphState.node(NodeGraphNodeId{static_cast<uint32_t>(selectedId)}) : nullptr;
    return node && node->typeId == nodegraphtypes::Write
        ? readWriteNodeParams(*node).product : WriteProductSelection{};
}

QVariantMap NodeGraphUiModel::selectedExportRow() const {
    const int index = selectedExportProductIndex();
    return index >= 0 ? exportProductItems[index].toMap() : QVariantMap{};
}

QStringList NodeGraphUiModel::exportProductFormats() const {
    return selectedExportRow().value("supportedFormats").toStringList();
}

QString NodeGraphUiModel::exportProductFormat() const {
    return QString::fromStdString(currentWriteSelection().format);
}

bool NodeGraphUiModel::exportProductAvailable() const {
    const QVariantMap row = selectedExportRow();
    return !row.isEmpty() && row.value("available").toBool() &&
        exportProductFormats().contains(exportProductFormat());
}

QString NodeGraphUiModel::exportProductStatus() const {
    const QVariantMap row = selectedExportRow();
    return row.isEmpty() || row.value("available").toBool()
        ? QString{} : row.value("unavailableReason").toString();
}

void NodeGraphUiModel::setWriteSelection(const WriteProductSelection& selection) {
    if (selectedId <= 0 || selectedType != QStringLiteral("write")) return;
    if (!graphState.node(NodeGraphNodeId{static_cast<uint32_t>(selectedId)})) return;
    NodeGraphParamValue product{};
    product.id = nodegraphparams::write::Product;
    product.type = NodeGraphParamType::Struct;
    auto addInt = [&](const char* name, uint32_t id, uint32_t value) {
        NodeGraphParamValue field{}; field.id = id; field.type = NodeGraphParamType::Int; field.intValue = value;
        product.fieldValues.push_back({name, std::make_shared<NodeGraphParamValue>(std::move(field))});
    };
    auto addString = [&](const char* name, uint32_t id, const std::string& value) {
        NodeGraphParamValue field{}; field.id = id; field.type = NodeGraphParamType::String; field.stringValue = value;
        product.fieldValues.push_back({name, std::make_shared<NodeGraphParamValue>(std::move(field))});
    };
    addInt("producerNodeId", 1, selection.producerNodeId);
    addInt("outputSocketId", 2, selection.outputSocketId);
    addString("productId", 3, selection.productId);
    NodeGraphParamValue format{};
    format.id = 4;
    format.type = NodeGraphParamType::Enum;
    format.enumValue = selection.format;
    product.fieldValues.push_back({"format", std::make_shared<NodeGraphParamValue>(std::move(format))});
    emit setParameterRequested(NodeGraphNodeId{static_cast<uint32_t>(selectedId)}, product);
}

void NodeGraphUiModel::selectExportProduct(int rowIndex) {
    if (rowIndex < 0 || rowIndex >= exportProductItems.size()) return;
    const QVariantMap row = exportProductItems[rowIndex].toMap();
    WriteProductSelection selection = currentWriteSelection();
    const bool alreadySelected = !selection.empty() &&
        selection.producerNodeId == static_cast<uint32_t>(row.value("producerNodeId").toInt()) &&
        selection.outputSocketId == static_cast<uint32_t>(row.value("outputSocketId").toInt()) &&
        selection.productId == row.value("productId").toString().toStdString();
    if (!alreadySelected) {
        selection.producerNodeId = static_cast<uint32_t>(row.value("producerNodeId").toInt());
        selection.outputSocketId = static_cast<uint32_t>(row.value("outputSocketId").toInt());
        selection.productId = row.value("productId").toString().toStdString();
        selection.format = row.value("defaultFormat").toString().toStdString();
        setWriteSelection(selection);
    }
}

void NodeGraphUiModel::setExportProductFormat(const QString& format) {
    WriteProductSelection selection = currentWriteSelection();
    if (selection.empty() || !exportProductFormats().contains(format)) return;
    selection.format = format.toStdString();
    setWriteSelection(selection);
}

void NodeGraphUiModel::requestCurrentExportProducts() {
    if (selectedId <= 0 || selectedType != QStringLiteral("write")) {
        const bool changed = !exportProductItems.empty() || !exportProductsErrorText.isEmpty();
        exportProductItems.clear();
        exportProductsErrorText.clear();
        if (changed) emit exportProductsChanged();
        return;
    }
    exportProductsErrorText.clear();
    const uint64_t requestId = nextExportProductsRequestId++;
    expectedExportProductsRequestId = requestId;
    expectedExportProductsRevision = graphState.revision;
    expectedExportProductsNodeId = selectedId;
    emit exportProductsRequested(selectedId, requestId, graphState.revision);
}

void NodeGraphUiModel::applyExportProducts(int nodeId, qulonglong requestId, qulonglong graphRevision,
                                           const QVariantList& products, const QString& error) {
    if (nodeId != expectedExportProductsNodeId || requestId != expectedExportProductsRequestId ||
        graphRevision != expectedExportProductsRevision || nodeId != selectedId) return;
    exportProductsErrorText = error;
    if (!error.isEmpty()) { emit exportProductsChanged(); return; }
    exportProductItems = products;
    assignExportProductLabels();

    WriteProductSelection selection = currentWriteSelection();
    if (!selection.empty()) {
        const QVariantMap row = selectedExportRow();
        if (row.isEmpty()) {
            setWriteSelection(WriteProductSelection{});
        } else if (!exportProductFormats().contains(exportProductFormat())) {
            selection.format = row.value("defaultFormat").toString().toStdString();
            setWriteSelection(selection);
        }
    }
    emit exportProductsChanged();
}

bool NodeGraphUiModel::assignExportProductLabels() {
    QStringList used;
    bool changed = false;
    for (QVariant& value : exportProductItems) {
        QVariantMap row = value.toMap();
        const QString title = nodeTitle(row.value("labelNodeId").toInt());
        const QString typeName = row.value("typeName").toString();
        QString display = title.isEmpty() ? typeName
            : typeName.isEmpty() ? title : title + QStringLiteral(".") + typeName;
        if (used.contains(display))
            display += QStringLiteral(" (#%1)").arg(row.value("labelNodeId").toInt());
        if (used.contains(display))
            display = QStringLiteral("%1 (#%2:%3)").arg(title.isEmpty() ? typeName : title)
                .arg(row.value("producerNodeId").toInt()).arg(row.value("outputSocketId").toInt());
        used.push_back(display);
        if (row.value("displayLabel").toString() != display) {
            row.insert("displayLabel", display);
            value = row;
            changed = true;
        }
    }
    return changed;
}

void NodeGraphUiModel::relabelExportProducts() {
    if (assignExportProductLabels()) emit exportProductsChanged();
}

void NodeGraphUiModel::exportNode(int nodeId) {
    if (nodeId <= 0) return;
    const NodeGraphNodeId id{static_cast<uint32_t>(nodeId)};
    const NodeGraphNode* node = graphState.node(id);
    if (!node || node->typeId != nodegraphtypes::Write) return;

    emit exportRequested(nodeId);
}

QVariantList NodeGraphUiModel::socketList(NodeGraphNodeId nodeId, const std::vector<NodeGraphSocket>& sockets) const {
    QVariantList result;
    result.reserve(static_cast<qsizetype>(sockets.size()));
    for (const NodeGraphSocket& socket : sockets) {
        QVariantMap item;
        item.insert(QStringLiteral("socketId"), static_cast<int>(socket.id.value));
        item.insert(QStringLiteral("name"), QString::fromStdString(socket.name));
        item.insert(QStringLiteral("valueType"), static_cast<int>(socketType(graphState, nodeId, socket.id)));
        QVariantList acceptedValueTypes;
        acceptedValueTypes.reserve(static_cast<qsizetype>(socket.acceptedValueTypes.size()));
        for (NodeGraphValueType acceptedType : socket.acceptedValueTypes) {
            acceptedValueTypes.push_back(static_cast<int>(acceptedType));
        }
        item.insert(QStringLiteral("acceptedValueTypes"), acceptedValueTypes);
        result.push_back(item);
    }
    return result;
}

void NodeGraphUiModel::rebuildNodeCategories() {
    categoryItems.clear();
    std::vector<NodeTypeDefinition> definitions = typeDefinitions;
    std::sort(definitions.begin(), definitions.end(), [](const NodeTypeDefinition& lhs, const NodeTypeDefinition& rhs) {
        const int lhsCategory = nodeCategoryOrder(lhs.category);
        const int rhsCategory = nodeCategoryOrder(rhs.category);
        return lhsCategory == rhsCategory ? lhs.displayName < rhs.displayName : lhsCategory < rhsCategory;
    });

    for (int order = 0; order <= 3; ++order) {
        QVariantList types;
        NodeGraphNodeCategory category = NodeGraphNodeCategory::Misc;
        for (const NodeTypeDefinition& definition : definitions) {
            if (nodeCategoryOrder(definition.category) != order) {
                continue;
            }
            category = definition.category;
            QVariantMap type;
            type.insert(QStringLiteral("typeId"), QString::fromStdString(definition.id));
            type.insert(QStringLiteral("name"), QString::fromStdString(definition.displayName));
            types.push_back(type);
        }
        if (!types.empty()) {
            QVariantMap categoryItem;
            categoryItem.insert(QStringLiteral("name"), nodeCategoryName(category));
            categoryItem.insert(QStringLiteral("types"), types);
            categoryItems.push_back(categoryItem);
        }
    }
    emit nodeCategoriesChanged();
}

void NodeGraphUiModel::notifySelectionChanged(bool requestRuntimeSelection) {
    if (std::find(selectedIds.begin(), selectedIds.end(), static_cast<uint32_t>(selectedId)) == selectedIds.end()) {
        selectedId = selectedIds.empty() ? 0 : static_cast<int>(selectedIds.back());
    }
    if (m_probePickingActive && selectedId <= 0) {
        cancelProbePicking();
    }
    rebuildSelectedNode();
    if (!nodes.empty()) {
        emit dataChanged(index(0, 0), index(static_cast<int>(nodes.size()) - 1, 0), {SelectedRole});
    }
    if (requestRuntimeSelection) {
        emit selectionRequested(selectedId);
    }
    requestCurrentExportProducts();
}

void NodeGraphUiModel::rebuildSelectedNode() {
    selectedTitle.clear();
    selectedType.clear();
    selectedParameters.clear();

    if (selectedId > 0) {
        const NodeGraphNode* selectedNode = graphState.node(NodeGraphNodeId{static_cast<uint32_t>(selectedId)});
        if (selectedNode) {
            const NodeGraphNode& node = *selectedNode;
            selectedTitle = QString::fromStdString(node.title);
            selectedType = QString::fromStdString(node.typeId);
            const NodeTypeDefinition* definition = nullptr;
            for (const NodeTypeDefinition& candidate : typeDefinitions) {
                if (candidate.id == node.typeId) { definition = &candidate; break; }
            }
            for (const NodeGraphParamValue& parameter : node.parameters) {
                QVariantMap item;
                item.insert(QStringLiteral("id"), static_cast<int>(parameter.id));
                item.insert(QStringLiteral("type"), static_cast<int>(parameter.type));
                QString name = QStringLiteral("Parameter %1").arg(parameter.id);
                QVariant value;
                QStringList options;
                if (definition) {
                    for (const NodeGraphParamDefinition& parameterDefinition : definition->parameters) {
                        if (parameterDefinition.id == parameter.id) {
                            name = QString::fromStdString(parameterDefinition.name);
                            for (const std::string& option : parameterDefinition.enumOptions) {
                                options.push_back(QString::fromStdString(option));
                            }
                            break;
                        }
                    }
                }
                switch (parameter.type) {
                case NodeGraphParamType::Float: value = parameter.floatValue; break;
                case NodeGraphParamType::Int: value = static_cast<qlonglong>(parameter.intValue); break;
                case NodeGraphParamType::Bool: value = parameter.boolValue; break;
                case NodeGraphParamType::String: value = QString::fromStdString(parameter.stringValue); break;
                case NodeGraphParamType::Enum: value = QString::fromStdString(parameter.enumValue); break;
                default: continue;
                }
                item.insert(QStringLiteral("name"), name);
                item.insert(QStringLiteral("value"), value);
                item.insert(QStringLiteral("options"), options);
                selectedParameters.push_back(item);
            }
        }
    }

    emit selectedNodeChanged();
}
