#pragma once

#include "nodegraph/NodeGraphEditor.hpp"
#include "nodegraph/NodeGraphTypes.hpp"

#include <QWidget>

class NodeGraphBridge;
class NodeGraphCanvas;
class NodeGraphScene;
class NodePanel;
class ModelSelection;
class RuntimeQuery;
class SceneController;

class NodeGraphEditorWidget : public QWidget {
    Q_OBJECT
public:
    explicit NodeGraphEditorWidget(QWidget* parent = nullptr);

    void setRuntimeQuery(const RuntimeQuery* runtimeQuery);
    void setBridge(NodeGraphBridge* bridge);
    void setSceneController(const SceneController* sceneController);
    void setModelSelection(ModelSelection* modelSelection);
    void refreshGraph();
    void syncSelection();

private slots:
    void onRequestCreateMenu(QPoint globalPos, QPointF scenePos, bool requireEmptySpace);
    void onRequestDeleteSelected();
    void onRequestCopySelected();
    void onRequestPaste();

private:
    void createUi();
    void handleGraphSelectionChanged(NodeGraphNodeId nodeId);
    void openInspectorForNode(NodeGraphNodeId nodeId);
    void syncViewportSelectionToGraph();
    void addNodeAt(const NodeTypeId& typeId, const QPointF& scenePos);

    const RuntimeQuery* runtimeQuery = nullptr;
    NodeGraphBridge* bridge = nullptr;
    NodeGraphEditor editor;
    const SceneController* sceneController = nullptr;
    ModelSelection* modelSelection = nullptr;
    NodeGraphScene* graphScene = nullptr;
    NodePanel* nodePanel = nullptr;
    NodeGraphCanvas* graphCanvas = nullptr;
    bool suppressGraphSelectionHandling = false;
    uint32_t lastObservedRuntimeModelId = 0;
};
