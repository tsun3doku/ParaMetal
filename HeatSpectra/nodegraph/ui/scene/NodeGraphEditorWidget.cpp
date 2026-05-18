#include "NodeGraphEditorWidget.hpp"

#include "nodegraph/NodeGraphPayloadTypes.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeGraphUtils.hpp"
#include "nodegraph/ui/scene/NodeGraphCanvas.hpp"
#include "nodegraph/ui/scene/NodeGraphScene.hpp"
#include "nodegraph/ui/widgets/NodePanel.hpp"
#include "nodegraph/NodeGraphBridge.hpp"
#include "runtime/RuntimeInterfaces.hpp"
#include "scene/SceneController.hpp"
#include "scene/ModelSelection.hpp"
#include "util/UiTheme.hpp"

#include <QAction>
#include <QContextMenuEvent>
#include <QCursor>
#include <QFrame>
#include <QGraphicsItem>
#include <QGraphicsView>
#include <QKeyEvent>
#include <QKeySequence>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPoint>
#include <QSplitter>
#include <QTransform>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QWidget>

#include <map>
#include <limits>

namespace {

QString nodeCategoryDisplayName(NodeGraphNodeCategory category) {
    switch (category) {
    case NodeGraphNodeCategory::Model:
        return "Model";
    case NodeGraphNodeCategory::PointSurface:
        return "Point/Surface";
    case NodeGraphNodeCategory::Meshing:
        return "Meshing";
    case NodeGraphNodeCategory::System:
        return "System";
    case NodeGraphNodeCategory::Custom:
        return "Misc";
    }
    return "Misc";
}

int nodeCategorySortKey(NodeGraphNodeCategory category) {
    switch (category) {
    case NodeGraphNodeCategory::Model:
        return 0;
    case NodeGraphNodeCategory::PointSurface:
        return 1;
    case NodeGraphNodeCategory::Meshing:
        return 2;
    case NodeGraphNodeCategory::System:
        return 3;
    case NodeGraphNodeCategory::Custom:
        return 4;
    }
    return 4;
}

} // namespace

NodeGraphEditorWidget::NodeGraphEditorWidget(QWidget* parent)
    : QWidget(parent) {
    createUi();
}

void NodeGraphEditorWidget::setRuntimeQuery(const RuntimeQuery* runtimeQueryPtr) {
    if (runtimeQuery == runtimeQueryPtr) {
        return;
    }
    runtimeQuery = runtimeQueryPtr;
    if (nodePanel) {
        nodePanel->bind(bridge, runtimeQuery);
    }
}

void NodeGraphEditorWidget::setBridge(NodeGraphBridge* bridgePtr) {
    bridge = bridgePtr;
    editor.setBridge(bridgePtr);
    if (graphScene) {
        graphScene->setBridge(bridge);
    }
    if (nodePanel) {
        nodePanel->bind(bridge, runtimeQuery);
    }
}

void NodeGraphEditorWidget::setSceneController(const SceneController* sceneControllerPtr) {
    sceneController = sceneControllerPtr;
}

void NodeGraphEditorWidget::setModelSelection(ModelSelection* modelSelectionPtr) {
    modelSelection = modelSelectionPtr;
    lastObservedRuntimeModelId = std::numeric_limits<uint32_t>::max();
}

void NodeGraphEditorWidget::refreshGraph() {
    if (graphScene) {
        graphScene->applyPendingChanges();
    }
}

void NodeGraphEditorWidget::syncSelection() {
    syncViewportSelectionToGraph();
}

void NodeGraphEditorWidget::createUi() {
    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    graphScene = new NodeGraphScene(this);
    NodeGraphCanvas* canvas = new NodeGraphCanvas(this);
    graphCanvas = canvas;
    canvas->setScene(graphScene);
    canvas->setFocusPolicy(Qt::StrongFocus);

    nodePanel = new NodePanel(this);
    nodePanel->bind(bridge, runtimeQuery);

    QSplitter* splitter = new QSplitter(Qt::Vertical, this);
    ui::configureSplitter(*splitter);
    splitter->addWidget(nodePanel);
    splitter->addWidget(canvas);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 1);
    splitter->setCollapsible(0, false);
    splitter->setCollapsible(1, false);
    splitter->setSizes({100000, 100000});
    rootLayout->addWidget(splitter, 1);

    connect(graphScene, &NodeGraphScene::nodeActivated,
            this, &NodeGraphEditorWidget::openInspectorForNode);
    connect(graphScene, &NodeGraphScene::nodeSelectionChanged,
            this, &NodeGraphEditorWidget::handleGraphSelectionChanged);
    connect(graphScene, &NodeGraphScene::graphPopulated,
            canvas, &NodeGraphCanvas::centerOnContent,
            Qt::SingleShotConnection);

    connect(canvas, &NodeGraphCanvas::requestCreateMenu, this, &NodeGraphEditorWidget::onRequestCreateMenu);
    connect(canvas, &NodeGraphCanvas::requestDeleteSelected, this, &NodeGraphEditorWidget::onRequestDeleteSelected);
    connect(canvas, &NodeGraphCanvas::requestCopySelected, this, &NodeGraphEditorWidget::onRequestCopySelected);
    connect(canvas, &NodeGraphCanvas::requestPaste, this, &NodeGraphEditorWidget::onRequestPaste);
}

void NodeGraphEditorWidget::onRequestCreateMenu(QPoint globalPos, QPointF scenePos, bool requireEmptySpace) {
    if (!bridge || !graphScene) {
        return;
    }
    if (requireEmptySpace) {
        const QGraphicsItem* hitItem = graphScene->itemAt(scenePos, QTransform());
        if (hitItem) {
            return;
        }
    }

    QMenu addMenu(this);
    std::map<int, QMenu*> categoryMenus;
    auto getCategoryMenu = [&](NodeGraphNodeCategory category) -> QMenu* {
        const int key = nodeCategorySortKey(category);
        const auto existing = categoryMenus.find(key);
        if (existing != categoryMenus.end()) {
            return existing->second;
        }
        QMenu* menu = addMenu.addMenu(nodeCategoryDisplayName(category));
        categoryMenus.emplace(key, menu);
        return menu;
    };

    const std::vector<NodeTypeDefinition>& definitions = bridge ? bridge->getRegistry().allNodeTypes() : std::vector<NodeTypeDefinition>{};
    for (const NodeTypeDefinition& definition : definitions) {
        QMenu* categoryMenu = getCategoryMenu(definition.category);
        QAction* action = categoryMenu->addAction(QString::fromStdString(definition.displayName));
        action->setData(QString::fromStdString(definition.id));
    }

    QAction* selectedAction = addMenu.exec(globalPos);
    if (!selectedAction) {
        return;
    }
    addNodeAt(selectedAction->data().toString().toStdString(), scenePos);
}

void NodeGraphEditorWidget::onRequestDeleteSelected() {
    if (!graphScene) {
        return;
    }
    graphScene->removeSelectedNodes();
}

void NodeGraphEditorWidget::onRequestCopySelected() {
    if (!graphScene) {
        return;
    }
    graphScene->copySelectedNodes();
}

void NodeGraphEditorWidget::onRequestPaste() {
    if (!graphScene) {
        return;
    }
    graphScene->pasteCopiedNodes();
}

void NodeGraphEditorWidget::handleGraphSelectionChanged(NodeGraphNodeId nodeId) {
    if (suppressGraphSelectionHandling) {
        return;
    }

    if (nodePanel && nodePanel->isVisible() && bridge && nodeId.isValid()) {
        nodePanel->bind(bridge, runtimeQuery);
        nodePanel->setNode(nodeId);
    }

    uint32_t runtimeModelId = 0;
    if (bridge && sceneController && nodeId.isValid()) {
        NodeGraphNode node{};
        if (bridge->getNode(nodeId, node) && getNodeTypeId(node.typeId) == nodegraphtypes::Model) {
            const NodeGraphSocket* outputSocket = node.outputOfType(payloadtypes::Geometry);
            if (outputSocket && outputSocket->id.isValid()) {
                sceneController->tryGetSocketRuntimeModelId(NodeSocketKey(node.id, outputSocket->id), runtimeModelId);
            }
        }
    }

    if (modelSelection) {
        if (runtimeModelId != 0) {
            modelSelection->setSelectedModelID(runtimeModelId);
        } else {
            modelSelection->clearSelection();
        }
    }

    lastObservedRuntimeModelId = runtimeModelId;
}

void NodeGraphEditorWidget::openInspectorForNode(NodeGraphNodeId nodeId) {
    if (!nodePanel || !graphCanvas) {
        return;
    }
    nodePanel->bind(bridge, runtimeQuery);
    if (!nodePanel->setNode(nodeId)) {
        return;
    }
    graphCanvas->setFocus(Qt::OtherFocusReason);
}

void NodeGraphEditorWidget::syncViewportSelectionToGraph() {
    if (!graphScene || !modelSelection || !sceneController) {
        return;
    }

    const uint32_t selectedRuntimeModelId = modelSelection->getSelectedModelID();
    if (selectedRuntimeModelId == lastObservedRuntimeModelId) {
        return;
    }

    lastObservedRuntimeModelId = selectedRuntimeModelId;

    NodeGraphNodeId nodeId{};
    if (selectedRuntimeModelId != 0) {
        uint64_t outputSocketKey = 0;
        NodeGraphSocketId outputSocketId{};
        if (sceneController->tryGetRuntimeModelSocketKey(selectedRuntimeModelId, outputSocketKey)) {
            const NodeSocketKey key(outputSocketKey);
            nodeId = key.node();
            outputSocketId = key.socket();
        }
    }

    suppressGraphSelectionHandling = true;
    if (nodeId.isValid()) {
        graphScene->setSelectedNode(nodeId);
    } else {
        graphScene->clearNodeSelection();
    }
    suppressGraphSelectionHandling = false;

    if (nodeId.isValid()) {
        openInspectorForNode(nodeId);
    }
}

void NodeGraphEditorWidget::addNodeAt(const NodeTypeId& typeId, const QPointF& scenePos) {
    if (!bridge || !graphScene) {
        return;
    }
    editor.addNode(typeId, "", static_cast<float>(scenePos.x()), static_cast<float>(scenePos.y()));
    graphScene->applyPendingChanges();
}
