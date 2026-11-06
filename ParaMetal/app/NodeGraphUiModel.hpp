#pragma once

#include "UiRuntimeTypes.hpp"
#include "nodegraph/NodeGraphState.hpp"
#include "nodegraph/NodeGraphEditor.hpp"
#include "nodegraph/NodeWriteParams.hpp"

#include <QAbstractListModel>
#include <QStringList>
#include <QVariantList>
#include <QVariantMap>

#include <cstdint>
#include <vector>

class NodeGraphUiModel final : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(QVariantList edges READ edges NOTIFY edgesChanged)
    Q_PROPERTY(QVariantList nodeCategories READ nodeCategories NOTIFY nodeCategoriesChanged)
    Q_PROPERTY(int selectedNodeId READ selectedNodeId WRITE setSelectedNodeId NOTIFY selectedNodeChanged)
    Q_PROPERTY(QString selectedNodeTitle READ selectedNodeTitle NOTIFY selectedNodeChanged)
    Q_PROPERTY(QString selectedNodeType READ selectedNodeType NOTIFY selectedNodeChanged)
    Q_PROPERTY(QString selectedNodeDescription READ selectedNodeDescription NOTIFY selectedNodeChanged)
    Q_PROPERTY(QVariantList selectedNodeParameters READ selectedNodeParameters NOTIFY selectedNodeChanged)
    Q_PROPERTY(QVariantList serialPorts READ serialPorts NOTIFY serialPortsChanged)
    Q_PROPERTY(bool probePickingActive READ isProbePickingActive NOTIFY probePickingActiveChanged)
    Q_PROPERTY(QStringList exportProductLabels READ exportProductLabels NOTIFY exportProductsChanged)
    Q_PROPERTY(int selectedExportProductIndex READ selectedExportProductIndex NOTIFY exportProductsChanged)
    Q_PROPERTY(QStringList exportProductFormats READ exportProductFormats NOTIFY exportProductsChanged)
    Q_PROPERTY(QString exportProductFormat READ exportProductFormat NOTIFY exportProductsChanged)
    Q_PROPERTY(bool exportProductAvailable READ exportProductAvailable NOTIFY exportProductsChanged)
    Q_PROPERTY(QString exportProductStatus READ exportProductStatus NOTIFY exportProductsChanged)
    Q_PROPERTY(QString exportProductsError READ exportProductsError NOTIFY exportProductsChanged)

public:
    bool isProbePickingActive() const { return m_probePickingActive; }
    enum Role {
        NodeIdRole = Qt::UserRole + 1,
        TitleRole,
        TypeRole,
        NodeXRole,
        NodeYRole,
        InputsRole,
        OutputsRole,
        DisplayEnabledRole,
        FrozenRole,
        SelectedRole
    };
    Q_ENUM(Role)

    explicit NodeGraphUiModel(QObject* parent = nullptr);
    ~NodeGraphUiModel() override;

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;

    QVariantList edges() const;
    QVariantList nodeCategories() const;
    int selectedNodeId() const;
    bool isNodeSelected(int nodeId) const;
    QString selectedNodeTitle() const;
    QString selectedNodeType() const;
    QString selectedNodeDescription() const;
    QVariantList selectedNodeParameters() const;
    QVariantList serialPorts() const;
    QStringList exportProductLabels() const;
    int selectedExportProductIndex() const;
    QStringList exportProductFormats() const;
    QString exportProductFormat() const;
    bool exportProductAvailable() const;
    QString exportProductStatus() const;
    QString exportProductsError() const { return exportProductsErrorText; }

    void setRuntimeSelectedNodeId(int nodeId);

public slots:
    void initializeGraph(const NodeGraphState& state, const std::vector<NodeTypeDefinition>& definitions);
    void replaceGraphState(const NodeGraphState& state);
    void applyDelta(const NodeGraphDelta& delta);
    void handleNodesPasted(const std::vector<NodeGraphNodeId>& nodeIds);
    void applyExportProducts(int nodeId, qulonglong requestId, qulonglong graphRevision,
                             const QVariantList& products, const QString& error);

public:
    Q_INVOKABLE void moveNode(int nodeId, qreal x, qreal y);
    Q_INVOKABLE void removeNode(int nodeId);
    Q_INVOKABLE void startProbePicking(int nodeId, int probeIndex = 0);
    Q_INVOKABLE void cancelProbePicking();
    Q_INVOKABLE void removeSelectedNodes();
    Q_INVOKABLE void resetToDefaultGraph();
    Q_INVOKABLE int addNode(const QString& typeId, qreal x, qreal y);
    Q_INVOKABLE void toggleNodeDisplay(int nodeId);
    Q_INVOKABLE void toggleNodeFrozen(int nodeId);
    Q_INVOKABLE void setSelectedNodeId(int nodeId);
    Q_INVOKABLE void setNodeSelected(int nodeId, bool additive);
    Q_INVOKABLE void copySelectedNodes();
    Q_INVOKABLE void pasteCopiedNodes();
    Q_INVOKABLE bool connectSockets(int fromNode, int fromSocket, int toNode, int toSocket);
    Q_INVOKABLE void removeConnection(int edgeId);
    Q_INVOKABLE void setParameterValue(int parameterId, const QVariant& value);
    Q_INVOKABLE void setHeatMaterialPreset(const QString& presetName);
    Q_INVOKABLE void refreshSerialPorts();
    Q_INVOKABLE void exportNode(int nodeId);
    Q_INVOKABLE QString nodeTitle(int nodeId) const;
    Q_INVOKABLE QString selectedWriteFile() const;
    Q_INVOKABLE void selectExportProduct(int rowIndex);
    Q_INVOKABLE void setExportProductFormat(const QString& format);

signals:
    void edgesChanged();
    void nodeCategoriesChanged();
    void selectedNodeChanged();
    void selectionRequested(int nodeId);
    void probePickingRequested(int nodeId, int probeIndex);
    void probePickingActiveChanged();
    void serialPortsChanged();
    void resetRequested();
    void addNodeRequested(const QString& typeId, float x, float y);
    void removeNodeRequested(NodeGraphNodeId nodeId);
    void moveNodeRequested(NodeGraphNodeId nodeId, float x, float y);
    void toggleNodeDisplayRequested(NodeGraphNodeId nodeId);
    void toggleNodeFrozenRequested(NodeGraphNodeId nodeId);
    void connectSocketsRequested(NodeGraphNodeId fromNode, NodeGraphSocketId fromSocket,
                                 NodeGraphNodeId toNode, NodeGraphSocketId toSocket);
    void removeConnectionRequested(NodeGraphEdgeId edgeId);
    void setParameterRequested(NodeGraphNodeId nodeId, const NodeGraphParamValue& parameter);
    void pasteRequested(const GraphPastePayload& payload);
    void exportRequested(int nodeId);
    void exportProductsRequested(int nodeId, qulonglong requestId, qulonglong graphRevision);
    void exportProductsChanged();

private:
    QVariantList socketList(NodeGraphNodeId nodeId, const std::vector<NodeGraphSocket>& sockets) const;
    void refresh();
    void rebuildNodeCategories();
    void rebuildSelectedNode();
    void notifySelectionChanged(bool requestRuntimeSelection);
    void requestCurrentExportProducts();
    WriteProductSelection currentWriteSelection() const;
    void setWriteSelection(const WriteProductSelection& selection);
    QVariantMap selectedExportRow() const;
    bool assignExportProductLabels();
    void relabelExportProducts();

    uint64_t revision = UINT64_MAX;
    std::vector<NodeGraphNode> nodes;
    NodeGraphState graphState;
    QVariantList edgeItems;
    QVariantList categoryItems;
    std::vector<uint32_t> selectedIds;
    std::vector<NodeGraphEditor::CopiedNode> copiedNodes;
    std::vector<NodeGraphEditor::CopiedEdge> copiedEdges;
    std::vector<NodeTypeDefinition> typeDefinitions;
    int selectedId = 0;
    QString selectedTitle;
    QString selectedType;
    QVariantList selectedParameters;
    QVariantList serialPortItems;
    bool m_probePickingActive = false;
    QVariantList exportProductItems;
    QString exportProductsErrorText;
    uint64_t nextExportProductsRequestId = 1;
    uint64_t expectedExportProductsRequestId = 0;
    uint64_t expectedExportProductsRevision = 0;
    int expectedExportProductsNodeId = 0;
};
