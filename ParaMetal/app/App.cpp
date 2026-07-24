#include "App.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "GraphHost.hpp"
#include "NodeGraphCanvasItem.hpp"
#include "UiRuntimeTypes.hpp"
#include "ViewportItem.hpp"
#include "ui/UiFontRegistry.hpp"
#include "ui/UiTypographyQml.hpp"

#include <QtCore/QCoreApplication>
#include <QtCore/QFileInfo>
#include <QtGui/QSurfaceFormat>
#include <QtQuick/QQuickGraphicsConfiguration>
#include <QtQuick/QQuickWindow>
#include <QtQml/QQmlContext>
#include <QtQml/QQmlEngine>
#include <QtQml/QJSEngine>

static int& prepareQt(int& argc) {
    qputenv("QT_QUICK_CONTROLS_STYLE", "Basic");
    QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);
    return argc;
}

App::App(int& argc, char** argv)
    : application(prepareQt(argc), argv) {
    QGuiApplication::setApplicationName("ParaMetal");
    QGuiApplication::setOrganizationName("ParaMetal");
    ui::UiFontRegistry::installBundledFonts(application);
}

App::~App() {
    graphThread.stop();
}

int App::run() {
    const int initializationResult = initializePresentation();
    if (initializationResult != 0) return initializationResult;

    graphThread.start();
    window.show();
    return application.exec();
}

int App::initializePresentation() {
    QQuickGraphicsConfiguration graphicsConfiguration;
    graphicsConfiguration.setDeviceExtensions({
        QByteArrayLiteral("VK_KHR_external_memory"),
        QByteArrayLiteral("VK_KHR_external_memory_win32"),
        QByteArrayLiteral("VK_KHR_external_semaphore"),
        QByteArrayLiteral("VK_KHR_external_semaphore_win32")
    });

    qmlRegisterType<ViewportItem>("ParaMetal", 1, 0, "ViewportItem");
    qmlRegisterType<NodeGraphCanvasItem>("ParaMetal", 1, 0, "NodeGraphCanvasItem");
    qmlRegisterSingletonType<ui::UiTypographyQml>(
        "ParaMetal", 1, 0, "UiTypography",
        [](QQmlEngine*, QJSEngine*) -> QObject* {
            return new ui::UiTypographyQml;
        });

    qRegisterMetaType<ViewportUiState>();
    qRegisterMetaType<HeatPaletteUiState>();
    qRegisterMetaType<TimelineUiState>();
    qRegisterMetaType<SimulationUiState>();
    qRegisterMetaType<SerialUiState>();
    qRegisterMetaType<GraphPastePayload>();
    qRegisterMetaType<PythonResult>();
    qRegisterMetaType<NodeGraphState>();
    qRegisterMetaType<NodeGraphDelta>();
    qRegisterMetaType<NodeGraphNodeId>();
    qRegisterMetaType<NodeGraphSocketId>();
    qRegisterMetaType<NodeGraphEdgeId>();
    qRegisterMetaType<NodeGraphParamValue>();
    qRegisterMetaType<InputAction>();
    qRegisterMetaType<std::vector<NodeTypeDefinition>>();
    qRegisterMetaType<std::vector<NodeGraphNodeId>>();

    QSurfaceFormat surfaceFormat = window.format();
    surfaceFormat.setSwapInterval(0);
    window.setFormat(surfaceFormat);
    window.setGraphicsConfiguration(graphicsConfiguration);
    window.setResizeMode(QQuickView::SizeRootObjectToView);
    window.setTitle("ParaMetal");
    window.resize(1600, 900);
    window.setMinimumSize(QSize(700, 400));
    window.rootContext()->setContextProperty(QStringLiteral("ui"), &uiModel);

    const QString qmlPath = QCoreApplication::applicationDirPath() + QStringLiteral("/qml/Main.qml");
    if (!QFileInfo::exists(qmlPath)) return 2;

    window.setSource(QUrl::fromLocalFile(qmlPath));
    if (window.status() == QQuickView::Error) return 3;

    ViewportItem* viewport = window.rootObject()->findChild<ViewportItem*>(QStringLiteral("viewportItem"));
    if (!viewport) return 4;

    viewport->bindRuntime(runtimeSystems, runtimeNotifier);
    GraphHost* graphHost = graphThread.host();
    Q_ASSERT(graphHost);
    connectUi(*viewport, *graphHost);
    connectRuntime(*graphHost);
    connectGraph(*viewport, *graphHost);
    return 0;
}

void App::connectUi(ViewportItem& viewport, GraphHost& graphHost) {
    QObject::connect(uiModel.viewport(), &ViewportUiModel::wireframeModeRequested,
                     &viewport, &ViewportItem::requestWireframeMode);
    QObject::connect(uiModel.viewport(), &ViewportUiModel::gridEnabledRequested,
                     &viewport, &ViewportItem::requestGridEnabled);
    QObject::connect(uiModel.heatPalette(), &HeatPaletteUiModel::paletteRequested,
                     &viewport, &ViewportItem::requestHeatPalette);
    QObject::connect(uiModel.heatPalette(), &HeatPaletteUiModel::rangeRequested,
                     &viewport, &ViewportItem::requestHeatPaletteRange);
    QObject::connect(uiModel.timeline(), &TimelineUiModel::playingRequested,
                     &viewport, &ViewportItem::requestTimelinePlaying);
    QObject::connect(uiModel.timeline(), &TimelineUiModel::playingRequested,
                     &graphHost, [&graphHost](bool playing) {
                         if (playing) graphHost.activateTimeline();
                     }, Qt::QueuedConnection);
    QObject::connect(uiModel.timeline(), &TimelineUiModel::resetRequested,
                     &viewport, &ViewportItem::requestTimelineReset);
    QObject::connect(uiModel.timeline(), &TimelineUiModel::scrubRequested,
                     &viewport, &ViewportItem::requestTimelineScrub);
    QObject::connect(uiModel.timeline(), &TimelineUiModel::stepRequested,
                     &viewport, &ViewportItem::requestTimelineStep);
    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::selectionRequested,
                     &viewport, &ViewportItem::requestSelection);

    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::resetRequested,
                     &graphHost, &GraphHost::resetGraph, Qt::QueuedConnection);
    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::addNodeRequested,
                     &graphHost, &GraphHost::addNode, Qt::QueuedConnection);
    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::removeNodeRequested,
                     &graphHost, &GraphHost::removeNode, Qt::QueuedConnection);
    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::moveNodeRequested,
                     &graphHost, &GraphHost::moveNode, Qt::QueuedConnection);
    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::toggleNodeDisplayRequested,
                     &graphHost, &GraphHost::toggleNodeDisplay, Qt::QueuedConnection);
    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::toggleNodeFrozenRequested,
                     &graphHost, &GraphHost::toggleNodeFrozen, Qt::QueuedConnection);
    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::connectSocketsRequested,
                     &graphHost, &GraphHost::connectSockets, Qt::QueuedConnection);
    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::removeConnectionRequested,
                     &graphHost, &GraphHost::removeConnection, Qt::QueuedConnection);
    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::setParameterRequested,
                     &graphHost, &GraphHost::setParameter, Qt::QueuedConnection);
    QObject::connect(uiModel.nodeGraph(), &NodeGraphModel::pasteRequested,
                     &graphHost, &GraphHost::pasteFragment, Qt::QueuedConnection);
    QObject::connect(uiModel.console(), &ConsoleModel::executeRequested,
                     &graphHost, &GraphHost::executePython, Qt::QueuedConnection);
    QObject::connect(uiModel.console(), &ConsoleModel::resetGraphRequested,
                     &graphHost, &GraphHost::resetGraph, Qt::QueuedConnection);
}

void App::connectRuntime(GraphHost& graphHost) {
    QObject::connect(&runtimeNotifier, &RuntimeNotifier::runtimeReadyChanged,
                     uiModel.runtime(), &RuntimeStatusUiModel::setReady, Qt::QueuedConnection);
    QObject::connect(&runtimeNotifier, &RuntimeNotifier::viewportStateChanged,
                     uiModel.viewport(), &ViewportUiModel::applyState, Qt::QueuedConnection);
    QObject::connect(&runtimeNotifier, &RuntimeNotifier::heatPaletteVisibilityChanged,
                     uiModel.heatPalette(), &HeatPaletteUiModel::applyVisibility, Qt::QueuedConnection);
    QObject::connect(&runtimeNotifier, &RuntimeNotifier::timelineStateChanged,
                     uiModel.timeline(), &TimelineUiModel::applyState, Qt::QueuedConnection);
    QObject::connect(&runtimeNotifier, &RuntimeNotifier::simulationStateChanged,
                     uiModel.runtime(), &RuntimeStatusUiModel::applySimulation, Qt::QueuedConnection);
    QObject::connect(&runtimeNotifier, &RuntimeNotifier::serialStateChanged,
                     uiModel.runtime(), &RuntimeStatusUiModel::applySerial, Qt::QueuedConnection);
    QObject::connect(&runtimeNotifier, &RuntimeNotifier::graphSelectionChanged,
                     uiModel.nodeGraph(), [model = uiModel.nodeGraph()](NodeGraphNodeId id) {
                         model->setRuntimeSelectedNodeId(static_cast<int>(id.value));
                     }, Qt::QueuedConnection);
    QObject::connect(&runtimeNotifier, &RuntimeNotifier::inputActionRequested,
                     &graphHost, [&graphHost](const InputAction& action) {
                         if (const auto* edit = std::get_if<SetNodeParametersAction>(&action)) {
                             graphHost.setParameters(edit->nodeId, edit->parameters);
                         }
                     }, Qt::QueuedConnection);
}

void App::connectGraph(ViewportItem& viewport, GraphHost& graphHost) {
    QObject::connect(&graphHost, &GraphHost::initialized, &uiModel,
        [this, &viewport](const NodeGraphState& state,
                         const std::vector<NodeTypeDefinition>& definitions,
                         const QString& pythonVersion) {
            uiModel.nodeGraph()->initializeGraph(state, definitions);
            uiModel.console()->setPythonVersion(pythonVersion);
            viewport.initializeGraph(state);
        }, Qt::QueuedConnection);
    QObject::connect(&graphHost, &GraphHost::graphChanged,
                     uiModel.nodeGraph(), &NodeGraphModel::applyDelta, Qt::QueuedConnection);
    QObject::connect(&graphHost, &GraphHost::graphChanged,
                     &viewport, &ViewportItem::queueGraphDelta, Qt::QueuedConnection);
    QObject::connect(&graphHost, &GraphHost::graphReplaced,
                     &viewport, &ViewportItem::initializeGraph, Qt::QueuedConnection);
    QObject::connect(&graphHost, &GraphHost::graphReplaced,
                     uiModel.nodeGraph(), &NodeGraphModel::replaceGraphState, Qt::QueuedConnection);
    QObject::connect(&graphHost, &GraphHost::nodesPasted,
                     uiModel.nodeGraph(), &NodeGraphModel::handleNodesPasted, Qt::QueuedConnection);
    QObject::connect(&graphHost, &GraphHost::pythonFinished,
                     uiModel.console(), &ConsoleModel::applyResult, Qt::QueuedConnection);
    QObject::connect(&graphHost, &GraphHost::timelineRangeChanged,
                     &viewport, &ViewportItem::requestTimelineRange, Qt::QueuedConnection);
}
