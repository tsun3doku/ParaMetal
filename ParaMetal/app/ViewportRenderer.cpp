#include "ViewportRenderer.hpp"

#include "AppTypes.hpp"
#include "RuntimeNotifier.hpp"
#include "ViewportItem.hpp"
#include "ViewportMailbox.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "runtime/RuntimeInterfaces.hpp"
#include "runtime/RuntimeSystems.hpp"
#include "runtime/export/ExportTypes.hpp"
#include "runtime/TimelineController.hpp"
#include "scene/Camera.hpp"
#include "scene/CameraController.hpp"
#include "scene/MarkerInteractionController.hpp"
#include "scene/ModelSelection.hpp"

#include <QtCore/QElapsedTimer>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QVariantMap>
#include <QtQuick/QQuickWindow>
#include <QtGui/6.9.3/QtGui/rhi/qrhi.h>
#include <QtGui/6.9.3/QtGui/rhi/qrhi_platform.h>

#include <algorithm>

ViewportRenderer::ViewportRenderer(
    RuntimeSystems& runtime,
    RuntimeNotifier& notifier,
    ViewportMailbox& mailbox,
    ViewportItem& viewportItem)
    : runtime(runtime), notifier(notifier), mailbox(mailbox), viewportItem(viewportItem) {
}

ViewportRenderer::~ViewportRenderer() {
    runtime.shutdown();
}

void ViewportRenderer::initialize(QRhiCommandBuffer*) {
    if (!rhi() || rhi()->backend() != QRhi::Vulkan || !colorTexture()) {
        return;
    }

    const QRhiVulkanNativeHandles* nativeHandles =
        static_cast<const QRhiVulkanNativeHandles*>(rhi()->nativeHandles());
    if (!nativeHandles ||
        nativeHandles->physDev == VK_NULL_HANDLE ||
        nativeHandles->dev == VK_NULL_HANDLE ||
        nativeHandles->gfxQueue == VK_NULL_HANDLE) {
        return;
    }

    const QRhiTexture::NativeTexture nativeTexture = colorTexture()->nativeTexture();
    const VkImage image = reinterpret_cast<VkImage>(nativeTexture.object);
    const QSize size = colorTexture()->pixelSize();
    const VkExtent2D extent{
        static_cast<uint32_t>(std::max(1, size.width())),
        static_cast<uint32_t>(std::max(1, size.height()))
    };

    WindowRuntimeState& state = mailbox.runtimeState();
    state.width.store(extent.width, std::memory_order_release);
    state.height.store(extent.height, std::memory_order_release);

    if (!runtime.isInitialized()) {
        AppVulkanContext context{};
        context.physicalDevice = nativeHandles->physDev;
        context.device = nativeHandles->dev;
        context.graphicsQueue = nativeHandles->gfxQueue;
        context.queueFamilyIndex = nativeHandles->gfxQueueFamilyIdx;
        context.viewportImage = image;
        context.viewportFormat = VK_FORMAT_R8G8B8A8_UNORM;
        context.viewportExtent = extent;
        initialized = runtime.initialize(state, context);
        if (initialized) {
            runtimeFresh = true;
            frameTimer.start();
        }
        return;
    }

    initialized = runtime.updateViewportTarget(
        image,
        VK_FORMAT_R8G8B8A8_UNORM,
        extent);
}

void ViewportRenderer::synchronize(QQuickRhiItem* item) {
    WindowRuntimeState& state = mailbox.runtimeState();
    state.devicePixelRatio.store(
        static_cast<float>(item->window() ? item->window()->devicePixelRatio() : 1.0),
        std::memory_order_release);

    int worldUnit = static_cast<int>(units::defaultLengthUnit());
    if (mailbox.takeWorldUnit(worldUnit, runtimeFresh)) {
        runtime.setWorldUnit(worldUnit);
    }

    if (const NodeGraphState* graphState = mailbox.graphReplacement(runtimeFresh)) {
        applyGraphReplacement(*graphState);
        mailbox.graphReplacementApplied();
    } else {
        for (const NodeGraphDelta& delta : mailbox.graphDeltas()) {
            if (!runtime.applyGraphDelta(delta)) {
                const NodeGraphState* cachedState = mailbox.graphReplacement(true);
                Q_ASSERT(cachedState);
                if (cachedState) applyGraphReplacement(*cachedState);
                break;
            }
        }
        mailbox.graphDeltasApplied();
    }

    app::WireframeMode wireframeMode{};
    if (mailbox.takeWireframeMode(wireframeMode, runtimeFresh)) {
        runtime.setWireframeMode(wireframeMode);
    }
    bool gridEnabled = false;
    if (mailbox.takeGridEnabled(gridEnabled, runtimeFresh)) {
        runtime.setGridEnabled(gridEnabled);
    }
    app::BackgroundMode backgroundMode{};
    if (mailbox.takeBackgroundMode(backgroundMode, runtimeFresh)) {
        runtime.setBackgroundMode(backgroundMode);
    }
    bool navigationCubeVisible = true;
    if (mailbox.takeNavigationCubeVisible(navigationCubeVisible, runtimeFresh)) {
        runtime.setNavigationCubeVisible(navigationCubeVisible);
    }
    bool axisLabelsVisible = true;
    if (mailbox.takeAxisLabelsVisible(axisLabelsVisible, runtimeFresh)) {
        runtime.setAxisLabelsVisible(axisLabelsVisible);
    }
    if (mailbox.takeFocusWorldOrigin()) {
        runtime.focusCameraOnWorldOrigin();
    }
    CameraProjectionMode projectionMode{};
    if (mailbox.takeProjectionMode(projectionMode)) {
        runtime.setCameraProjectionMode(projectionMode);
    }
    float cameraFov = Camera::DefaultFov;
    if (mailbox.takeCameraFov(cameraFov)) {
        runtime.setCameraFov(cameraFov);
    }
    float cameraZoomSpeed = Camera::DefaultZoomSpeed;
    if (mailbox.takeCameraZoomSpeed(cameraZoomSpeed)) {
        runtime.setCameraZoomSpeed(cameraZoomSpeed);
    }
    float cameraPanSpeed = Camera::DefaultPanSpeed;
    if (mailbox.takeCameraPanSpeed(cameraPanSpeed)) {
        runtime.setCameraPanSpeed(cameraPanSpeed);
    }

    TimelineController* timeline = runtime.timelineController();
    uint32_t frameCount = 0;
    float fps = 0.0f;
    if (mailbox.takeTimelineRange(frameCount, fps, runtimeFresh)) {
        timeline->setFps(fps);
        timeline->setFrameCount(frameCount);
    }
    if (mailbox.takeTimelineReset()) {
        timeline->reset();
    }
    const int timelineStep = mailbox.takeTimelineStep();
    if (timelineStep != 0) {
        timeline->stepFrames(timelineStep);
    }
    bool timelinePlaying = false;
    if (mailbox.takeTimelinePlaying(timelinePlaying, runtimeFresh)) {
        timeline->setPlaying(timelinePlaying);
    }
    int selectedNode = 0;
    if (mailbox.takeSelection(selectedNode, runtimeFresh)) {
        applySelection(selectedNode);
    }
    int probeNode = 0;
    int probeIndex = 0;
    if (mailbox.takeProbePicking(probeNode, probeIndex)) {
        if (probeNode > 0) {
            runtime.startProbePicking(static_cast<uint32_t>(probeNode), probeIndex);
        } else {
            runtime.cancelProbePicking();
        }
    }
    float paletteMin = 0.0f, paletteMax = 100.0f;
    if (mailbox.takeHeatPaletteRange(paletteMin, paletteMax, runtimeFresh)) runtime.setHeatPaletteRange(paletteMin, paletteMax);
    int palette = 0;
    if (mailbox.takeHeatPalette(palette, runtimeFresh)) runtime.setHeatPalette(palette);
    ProjectFile::Viewport viewportProjectState;
    if (mailbox.takeAppliedViewportProjectState(viewportProjectState)) {
        runtime.setBackgroundMode(viewportProjectState.backgroundMode);
        runtime.setNavigationCubeVisible(viewportProjectState.navigationCubeVisible);
        runtime.setAxisLabelsVisible(viewportProjectState.axisLabelsVisible);
        if (CameraController* cameraController = runtime.getCameraController()) {
            cameraController->setCameraState(
                viewportProjectState.lookAt,
                viewportProjectState.orientation,
                viewportProjectState.radius,
                viewportProjectState.fov,
                viewportProjectState.projectionMode,
                viewportProjectState.orthographicHeight,
                viewportProjectState.zoomSpeed,
                viewportProjectState.panSpeed);
        }
    }
    if (mailbox.takeCurrentViewportProjectState()) {
        if (const CameraController* cameraController = runtime.getCameraController()) {
            const Camera& camera = cameraController->getCamera();
            ProjectFile::Viewport viewport{};
            viewport.lookAt = camera.getLookAt();
            viewport.orientation = camera.getOrientation();
            viewport.radius = camera.getRadius();
            viewport.fov = camera.getBaseFov();
            viewport.projectionMode = camera.getProjectionMode();
            viewport.orthographicHeight = camera.getOrthographicHeight();
            viewport.zoomSpeed = camera.getZoomSpeed();
            viewport.panSpeed = camera.getPanSpeed();
            const app::RenderSettings& renderSettings = runtime.renderSettings();
            viewport.backgroundMode = renderSettings.backgroundMode;
            viewport.navigationCubeVisible = renderSettings.navigationCubeVisible;
            viewport.axisLabelsVisible = renderSettings.axisLabelsVisible;
            viewportItem.projectViewportStateReady(viewport);
        }
    }
    for (int exportNodeId : mailbox.takePendingExports()) {
        runtime.requestExport(exportNodeId);
    }
    for (const ExportProductsRequest& request : mailbox.takePendingExportProducts()) {
        runtime.requestExportProducts(request.nodeId, request.requestId, request.graphRevision);
    }
    runtimeFresh = false;
}

void ViewportRenderer::render(QRhiCommandBuffer* commandBuffer) {
    if (!initialized || !commandBuffer || !colorTexture() || !rhi()) {
        return;
    }

    float deltaSeconds = 1.0f / 60.0f;
    if (frameTimer.isValid()) {
        const qint64 elapsedNanoseconds = frameTimer.nsecsElapsed();
        frameTimer.restart();
        deltaSeconds = std::clamp(
            static_cast<float>(elapsedNanoseconds) * 1.0e-9f,
            0.0f,
            0.1f);
    } else {
        frameTimer.start();
    }

    const uint32_t scrubFrame = mailbox.takeTimelineScrub();
    if (scrubFrame != ViewportMailbox::noPendingScrubFrame) {
        runtime.timelineController()->scrubToFrame(scrubFrame);
    }

    commandBuffer->beginExternal();
    const QRhiVulkanCommandBufferNativeHandles* nativeCommandBuffer =
        static_cast<const QRhiVulkanCommandBufferNativeHandles*>(commandBuffer->nativeHandles());
    if (!nativeCommandBuffer || nativeCommandBuffer->commandBuffer == VK_NULL_HANDLE) {
        commandBuffer->endExternal();
        return;
    }

    runtime.tickFrame(
        deltaSeconds,
        nativeCommandBuffer->commandBuffer,
        static_cast<uint32_t>(rhi()->currentFrameSlot()));
    commandBuffer->endExternal();

    runtime.processDeferredWork();

    NodeGraphNodeId nodeId{};
    std::vector<NodeGraphParamValue> parameters;
    while (runtime.takePendingNodeParameters(nodeId, parameters)) {
        for (const auto& param : parameters) {
            if (param.id == nodegraphparams::heatprobe::ProbesJson) {
                notifier.notifyProbePickingCompleted(static_cast<int>(nodeId.value));
                break;
            }
        }
        emit notifier.nodeParametersRequested(nodeId, parameters);
    }
    int exportNodeId = 0;
    bool exportSuccess = false;
    std::string exportMessage;
    while (runtime.takePendingExportResult(exportNodeId, exportSuccess, exportMessage)) {
        notifier.publishExportStatus(exportNodeId, exportSuccess, QString::fromStdString(exportMessage));
    }
    ExportProductsReply productsReply{};
    while (runtime.takeExportProductsReady(productsReply)) {
        QVariantList rows;
        for (const ExportProduct& product : productsReply.products) {
            QVariantMap row;
            row.insert(QStringLiteral("producerNodeId"), static_cast<int>(product.key.producerNodeId.value));
            row.insert(QStringLiteral("outputSocketId"), static_cast<int>(product.key.outputSocketId.value));
            row.insert(QStringLiteral("productId"), QString::fromStdString(product.key.productId));
            row.insert(QStringLiteral("labelNodeId"), static_cast<int>(product.labelNodeId.value));
            row.insert(QStringLiteral("typeName"), QString::fromStdString(product.typeName));
            QStringList formats;
            for (ExportFormat format : product.supportedFormats)
                formats.push_back(QString::fromStdString(exportutils::exportFormatToString(format)));
            row.insert(QStringLiteral("supportedFormats"), formats);
            row.insert(QStringLiteral("defaultFormat"), QString::fromStdString(exportutils::exportFormatToString(product.defaultFormat)));
            row.insert(QStringLiteral("available"), product.available);
            row.insert(QStringLiteral("unavailableReason"), QString::fromStdString(product.unavailableReason));
            rows.push_back(row);
        }
        notifier.publishExportProducts(productsReply.nodeId, productsReply.requestId,
            productsReply.graphRevision, rows, QString::fromStdString(productsReply.error));
    }
    colorTexture()->setNativeLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!snapshotTimer.isValid() || snapshotTimer.elapsed() >= 16) {
        publishState();
        snapshotTimer.restart();
    }

    update();
}

void ViewportRenderer::applyGraphReplacement(const NodeGraphState& graphState) {
    runtime.replaceGraphState(graphState);
    notifier.resetNodeStatuses();
}

void ViewportRenderer::applySelection(int selectedNode) {
    ModelSelection* selection = runtime.getModelSelection();
    NodeGraphController* controller = runtime.getNodeGraphController();
    if (!selection) return;
    selection->clearSelection();
    if (selectedNode <= 0 || !controller) return;
    std::vector<uint32_t> runtimeModelIds;
    if (!controller->runtimeModelIdsForNode(NodeGraphNodeId{static_cast<uint32_t>(selectedNode)}, runtimeModelIds)) return;
    for (uint32_t id : runtimeModelIds) selection->addSelectedModelID(id);
}

void ViewportRenderer::publishState() {
    const RuntimeQuery* query = runtime.runtimeQuery();
    if (!query) return;

    const app::RenderSettings& renderSettings = runtime.renderSettings();
    ViewportUiState viewport{};
    viewport.wireframeMode = renderSettings.wireframeMode;
    viewport.gridEnabled = renderSettings.gridEnabled;
    viewport.backgroundMode = renderSettings.backgroundMode;
    viewport.navigationCubeVisible = renderSettings.navigationCubeVisible;
    viewport.axisLabelsVisible = renderSettings.axisLabelsVisible;
    if (const CameraController* cameraController = runtime.getCameraController()) {
        const Camera& camera = cameraController->getCamera();
        viewport.projectionMode = camera.getProjectionMode();
        viewport.baseFov = camera.getBaseFov();
        viewport.zoomSpeed = camera.getZoomSpeed();
        viewport.panSpeed = camera.getPanSpeed();
    }
    if (viewport != publishedViewport) {
        publishedViewport = viewport;
        emit notifier.viewportStateChanged(viewport);
    }
    const bool paletteVisible = runtime.isHeatPaletteVisible();
    if (paletteVisible != publishedHeatPaletteVisible) {
        publishedHeatPaletteVisible = paletteVisible;
        emit notifier.heatPaletteVisibilityChanged(paletteVisible);
    }

    TimelineUiState timeline{};
    timeline.playing = query->isTimelinePlaying();
    timeline.currentFrame = query->getTimelineCurrentFrame();
    timeline.frameCount = query->getTimelineFrameCount();
    timeline.recordedFrames = query->getSimulationRecordedTimelineFrames();
    timeline.startFrame = query->getTimelineStartDisplayFrame();
    timeline.endFrame = query->getTimelineEndDisplayFrame();
    timeline.currentSeconds = query->getTimelineCurrentSeconds();
    timeline.durationSeconds = query->getTimelineDurationSeconds();
    if (timeline != publishedTimeline) {
        publishedTimeline = timeline;
        emit notifier.timelineStateChanged(timeline);
    }

    notifier.publishHeatSolveStatus(query->isSimulationActive(), query->isSimulationPaused());

    const NodeGraphController* graphController = runtime.getNodeGraphController();
    if (graphController) {
        notifier.publishHeatSolveError(QString::fromStdString(graphController->lastPackageErrors()));

        for (const auto& [nodeIdVal, node] : graphController->graphState().nodes) {
            if (node.typeId == nodegraphtypes::HeatProbe) {
                std::vector<float> temps;
                if (query->getHeatProbeTemperatures(node.id.value, temps)) {
                    notifier.publishHeatProbeTemperatures(static_cast<int>(node.id.value), temps);
                }
            } else if (node.typeId == nodegraphtypes::SerialTemperature && !node.outputs.empty()) {
                SerialTemperatureRuntime::Status status;
                const uint64_t sourceKey = NodeSocketKey(node.id, node.outputs.front().id).value;
                if (!query->getSerialTemperatureStatus(sourceKey, status)) continue;

                QString connectionText;
                switch (status.connection) {
                case SerialTemperatureRuntime::ConnectionState::Error:
                    connectionText = QStringLiteral("Error: ") + QString::fromStdString(status.detail);
                    break;
                case SerialTemperatureRuntime::ConnectionState::DisconnectedHolding:
                    connectionText = QStringLiteral("Disconnected - holding last reading: ") +
                        QString::fromStdString(status.detail);
                    break;
                default:
                    connectionText = QString::fromStdString(status.detail);
                    break;
                }

                QString temperatureText = QStringLiteral("--");
                QString pollingRateText = QStringLiteral("--");
                if (status.latestReading) {
                    temperatureText = QString::number(status.latestReading->temperatureC, 'f', 2) +
                        QStringLiteral(" \u00B0C");
                    pollingRateText = QString::number(status.latestReading->pollingRateHz, 'f', 1) +
                        QStringLiteral(" Hz");
                }
                notifier.publishSerialStatus(
                    static_cast<int>(node.id.value),
                    connectionText,
                    temperatureText,
                    pollingRateText);
            }
        }
    }

    if (runtime.getInteractionContext().isInitialized()) {
        if (auto selectedNode = runtime.getInteractionContext().markerInteractionController().takePendingSelectedNodeId()) {
            notifier.notifyGraphSelectionChanged(NodeGraphNodeId{*selectedNode});
        }
    }
}
