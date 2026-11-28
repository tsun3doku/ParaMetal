#include "ViewportMailbox.hpp"

#include <QtCore/QtGlobal>

#include <algorithm>

void ViewportMailbox::requestWireframeMode(app::WireframeMode mode) {
    requestedWireframeMode = mode;
    wireframeDirty = true;
}

void ViewportMailbox::requestGridEnabled(bool enabled) {
    requestedGridEnabled = enabled;
    gridDirty = true;
}

void ViewportMailbox::requestBackgroundMode(app::BackgroundMode mode) {
    requestedBackgroundMode = mode;
    backgroundModeDirty = true;
}

void ViewportMailbox::requestNavigationCubeVisible(bool visible) {
    requestedNavigationCubeVisible = visible;
    navigationCubeVisibleDirty = true;
}

void ViewportMailbox::requestAxisLabelsVisible(bool visible) {
    requestedAxisLabelsVisible = visible;
    axisLabelsVisibleDirty = true;
}

void ViewportMailbox::requestFocusWorldOrigin() {
    focusWorldOriginPending = true;
}

void ViewportMailbox::requestProjectionMode(CameraProjectionMode mode) {
    requestedProjectionMode = mode;
    projectionModeDirty = true;
}

void ViewportMailbox::requestCameraFov(float degrees) {
    requestedCameraFov = degrees;
    cameraFovDirty = true;
}

void ViewportMailbox::requestCameraZoomSpeed(float speed) {
    requestedCameraZoomSpeed = speed;
    cameraZoomSpeedDirty = true;
}

void ViewportMailbox::requestCameraPanSpeed(float speed) {
    requestedCameraPanSpeed = speed;
    cameraPanSpeedDirty = true;
}

void ViewportMailbox::requestCameraNearClip(float meters) {
    requestedCameraNearClip = meters;
    cameraCullDirty = true;
}

void ViewportMailbox::requestCullLinesOnly(bool linesOnly) {
    requestedCullLinesOnly = linesOnly;
    cameraCullDirty = true;
}

void ViewportMailbox::requestTimelinePlaying(bool playing) {
    pendingScrubFrame.store(noPendingScrubFrame, std::memory_order_relaxed);
    requestedTimelineStep = 0;
    timelineResetPending = false;
    requestedTimelinePlaying = playing;
    timelinePlayingDirty = true;
}

void ViewportMailbox::requestTimelineReset() {
    pendingScrubFrame.store(noPendingScrubFrame, std::memory_order_relaxed);
    requestedTimelineStep = 0;
    timelinePlayingDirty = false;
    timelineResetPending = true;
}

void ViewportMailbox::requestTimelineScrub(uint32_t frame) {
    requestedTimelineStep = 0;
    timelinePlayingDirty = false;
    timelineResetPending = false;
    pendingScrubFrame.store(frame, std::memory_order_relaxed);
}

void ViewportMailbox::requestTimelineStep(int delta) {
    pendingScrubFrame.store(noPendingScrubFrame, std::memory_order_relaxed);
    timelinePlayingDirty = false;
    timelineResetPending = false;
    requestedTimelineStep += delta;
}

void ViewportMailbox::requestTimelineRange(uint32_t frameCount, float fps) {
    Q_ASSERT(frameCount > 0);
    Q_ASSERT(fps > 0.0f);
    requestedTimelineFrameCount = frameCount;
    requestedTimelineFps = fps;
    timelineRangeDirty = true;
}

void ViewportMailbox::requestSelection(
    const std::vector<NodeGraphNodeId>& nodeIds,
    int primaryNodeId) {
    requestedSelectedNodes = nodeIds;
    requestedPrimaryNode = std::max(0, primaryNodeId);
    selectionDirty = true;
}

void ViewportMailbox::requestProbePicking(int nodeId, int probeIndex) {
    requestedProbePickingNode = nodeId;
    requestedProbePickingIndex = probeIndex;
    probePickingDirty = true;
}

void ViewportMailbox::requestBoundaryFaceEditing(int nodeId, bool armed) {
    requestedBoundaryFaceNode = std::max(0, nodeId);
    requestedBoundaryFaceArmed = armed;
    boundaryFaceDirty = true;
}

void ViewportMailbox::requestNodeAction(const NodeActionMessage& message) {
    if (message.type == NodeActionMessage::Type::Begin &&
        (message.nodeId.value == 0 ||
         message.nodeInstanceId == 0 ||
         message.actionId == 0)) {
        return;
    }
    pendingNodeActions.push_back(message);
}

void ViewportMailbox::requestViewportMeshSelection(bool active) {
    requestedViewportMeshSelection = active;
    viewportMeshSelectionDirty = true;
}

void ViewportMailbox::requestSubselectionElementType(int elementType) {
    requestedSubselectionElementType = std::clamp(
        elementType,
        static_cast<int>(MeshElementType::Vertex),
        static_cast<int>(MeshElementType::Face));
    subselectionElementTypeDirty = true;
}

void ViewportMailbox::requestHeatPaletteRange(float minimum, float maximum) {
    requestedHeatPaletteMin = minimum;
    requestedHeatPaletteMax = maximum;
    heatPaletteRangeDirty = true;
}
void ViewportMailbox::requestHeatPalette(int palette) { requestedHeatPalette = palette; heatPaletteDirty = true; }
void ViewportMailbox::requestVelocityPaletteRange(float minimum, float maximum) {
    requestedVelocityPaletteMin = minimum;
    requestedVelocityPaletteMax = maximum;
    velocityPaletteRangeDirty = true;
}
void ViewportMailbox::requestVelocityPalette(int palette) { requestedVelocityPalette = palette; velocityPaletteDirty = true; }
void ViewportMailbox::requestWorldUnit(int unit) { requestedWorldUnit = unit; worldUnitDirty = true; }

void ViewportMailbox::applyViewportProjectState(const ProjectFile::Viewport& viewport) {
    appliedViewportProjectState = viewport;
    viewportProjectStateDirty = true;
    requestedBackgroundMode = viewport.backgroundMode;
    requestedNavigationCubeVisible = viewport.navigationCubeVisible;
    requestedAxisLabelsVisible = viewport.axisLabelsVisible;
    backgroundModeDirty = true;
    navigationCubeVisibleDirty = true;
    axisLabelsVisibleDirty = true;
}

void ViewportMailbox::requestCurrentViewportProjectState() {
    currentViewportProjectStateRequested = true;
}

void ViewportMailbox::replaceGraphState(const NodeGraphState& graphState) {
    cachedGraphState = graphState;
    graphStateInitialized = true;
    graphStateDirty = true;
    pendingGraphDeltas.clear();
    pendingNodeActions.clear();
}

void ViewportMailbox::appendGraphDelta(const NodeGraphDelta& delta) {
    Q_ASSERT(graphStateInitialized);
    if (!graphStateInitialized) return;
    const bool applied = applyNodeGraphDelta(cachedGraphState, delta);
    Q_ASSERT(applied);
    if (!applied) return;
    pendingGraphDeltas.push_back(delta);
}

bool ViewportMailbox::takeWireframeMode(app::WireframeMode& mode, bool force) {
    if (!force && !wireframeDirty) return false;
    mode = requestedWireframeMode;
    wireframeDirty = false;
    return true;
}

bool ViewportMailbox::takeGridEnabled(bool& enabled, bool force) {
    if (!force && !gridDirty) return false;
    enabled = requestedGridEnabled;
    gridDirty = false;
    return true;
}

bool ViewportMailbox::takeBackgroundMode(app::BackgroundMode& mode, bool force) {
    if (!force && !backgroundModeDirty) return false;
    mode = requestedBackgroundMode;
    backgroundModeDirty = false;
    return true;
}

bool ViewportMailbox::takeNavigationCubeVisible(bool& visible, bool force) {
    if (!force && !navigationCubeVisibleDirty) return false;
    visible = requestedNavigationCubeVisible;
    navigationCubeVisibleDirty = false;
    return true;
}

bool ViewportMailbox::takeAxisLabelsVisible(bool& visible, bool force) {
    if (!force && !axisLabelsVisibleDirty) return false;
    visible = requestedAxisLabelsVisible;
    axisLabelsVisibleDirty = false;
    return true;
}

bool ViewportMailbox::takeFocusWorldOrigin() {
    if (!focusWorldOriginPending) return false;
    focusWorldOriginPending = false;
    return true;
}

bool ViewportMailbox::takeProjectionMode(CameraProjectionMode& mode) {
    if (!projectionModeDirty) return false;
    mode = requestedProjectionMode;
    projectionModeDirty = false;
    return true;
}

bool ViewportMailbox::takeCameraFov(float& degrees) {
    if (!cameraFovDirty) return false;
    degrees = requestedCameraFov;
    cameraFovDirty = false;
    return true;
}

bool ViewportMailbox::takeCameraZoomSpeed(float& speed) {
    if (!cameraZoomSpeedDirty) return false;
    speed = requestedCameraZoomSpeed;
    cameraZoomSpeedDirty = false;
    return true;
}

bool ViewportMailbox::takeCameraPanSpeed(float& speed) {
    if (!cameraPanSpeedDirty) return false;
    speed = requestedCameraPanSpeed;
    cameraPanSpeedDirty = false;
    return true;
}

bool ViewportMailbox::takeCameraCull(float& meters, bool& linesOnly, bool force) {
    if (!force && !cameraCullDirty) return false;
    meters = requestedCameraNearClip;
    linesOnly = requestedCullLinesOnly;
    cameraCullDirty = false;
    return true;
}

bool ViewportMailbox::takeTimelinePlaying(bool& playing, bool force) {
    if (!force && !timelinePlayingDirty) return false;
    playing = requestedTimelinePlaying;
    timelinePlayingDirty = false;
    return true;
}

bool ViewportMailbox::takeTimelineReset() {
    if (!timelineResetPending) return false;
    timelineResetPending = false;
    return true;
}

int ViewportMailbox::takeTimelineStep() {
    const int step = requestedTimelineStep;
    requestedTimelineStep = 0;
    return step;
}

uint32_t ViewportMailbox::takeTimelineScrub() {
    return pendingScrubFrame.exchange(noPendingScrubFrame, std::memory_order_relaxed);
}

bool ViewportMailbox::takeTimelineRange(uint32_t& frameCount, float& fps, bool force) {
    if (!force && !timelineRangeDirty) return false;
    frameCount = requestedTimelineFrameCount;
    fps = requestedTimelineFps;
    timelineRangeDirty = false;
    return true;
}

bool ViewportMailbox::takeSelection(
    std::vector<NodeGraphNodeId>& nodeIds,
    int& primaryNodeId,
    bool force) {
    if (!force && !selectionDirty) return false;
    nodeIds = requestedSelectedNodes;
    primaryNodeId = requestedPrimaryNode;
    selectionDirty = false;
    return true;
}

bool ViewportMailbox::takeProbePicking(int& outNodeId, int& outProbeIndex) {
    if (!probePickingDirty) return false;
    outNodeId = requestedProbePickingNode;
    outProbeIndex = requestedProbePickingIndex;
    probePickingDirty = false;
    return true;
}

bool ViewportMailbox::takeBoundaryFaceEditing(int& outNodeId, bool& outArmed, bool force) {
    if (!force && !boundaryFaceDirty) return false;
    outNodeId = requestedBoundaryFaceNode;
    outArmed = requestedBoundaryFaceArmed;
    boundaryFaceDirty = false;
    return true;
}

void ViewportMailbox::takeNodeActions(std::vector<NodeActionMessage>& outMessages) {
    outMessages.swap(pendingNodeActions);
}

void ViewportMailbox::clearNodeActions() {
    pendingNodeActions.clear();
}

bool ViewportMailbox::takeViewportMeshSelection(bool& outActive, bool force) {
    if (!force && !viewportMeshSelectionDirty) return false;
    outActive = requestedViewportMeshSelection;
    viewportMeshSelectionDirty = false;
    return true;
}

bool ViewportMailbox::takeSubselectionElementType(int& outElementType, bool force) {
    if (!force && !subselectionElementTypeDirty) return false;
    outElementType = requestedSubselectionElementType;
    subselectionElementTypeDirty = false;
    return true;
}

bool ViewportMailbox::takeHeatPaletteRange(float& minimum, float& maximum, bool force) {
    if (!heatPaletteRangeDirty && !force) return false;
    minimum = requestedHeatPaletteMin; maximum = requestedHeatPaletteMax; heatPaletteRangeDirty = false; return true;
}
bool ViewportMailbox::takeHeatPalette(int& palette, bool force) {
    if (!heatPaletteDirty && !force) return false;
    palette = requestedHeatPalette; heatPaletteDirty = false; return true;
}
bool ViewportMailbox::takeVelocityPaletteRange(float& minimum, float& maximum, bool force) {
    if (!velocityPaletteRangeDirty && !force) return false;
    minimum = requestedVelocityPaletteMin; maximum = requestedVelocityPaletteMax; velocityPaletteRangeDirty = false; return true;
}
bool ViewportMailbox::takeVelocityPalette(int& palette, bool force) {
    if (!velocityPaletteDirty && !force) return false;
    palette = requestedVelocityPalette; velocityPaletteDirty = false; return true;
}
bool ViewportMailbox::takeWorldUnit(int& unit, bool force) {
    if (!worldUnitDirty && !force) return false;
    unit = requestedWorldUnit;
    worldUnitDirty = false;
    return true;
}

bool ViewportMailbox::takeAppliedViewportProjectState(ProjectFile::Viewport& viewport) {
    if (!viewportProjectStateDirty) return false;
    viewport = appliedViewportProjectState;
    viewportProjectStateDirty = false;
    return true;
}

bool ViewportMailbox::takeCurrentViewportProjectState() {
    if (!currentViewportProjectStateRequested) return false;
    currentViewportProjectStateRequested = false;
    return true;
}

const NodeGraphState* ViewportMailbox::graphReplacement(bool force) const {
    if (!graphStateInitialized || (!force && !graphStateDirty)) return nullptr;
    return &cachedGraphState;
}

void ViewportMailbox::graphReplacementApplied() {
    graphStateDirty = false;
    pendingGraphDeltas.clear();
}

void ViewportMailbox::requestExport(int nodeId) {
    std::lock_guard<std::mutex> lock(exportMutex);
    pendingExportNodes.push_back(nodeId);
}

std::vector<int> ViewportMailbox::takePendingExports() {
    std::lock_guard<std::mutex> lock(exportMutex);
    std::vector<int> result;
    result.swap(pendingExportNodes);
    return result;
}

void ViewportMailbox::requestExportProducts(int nodeId, uint64_t requestId, uint64_t graphRevision) {
    if (nodeId <= 0 || requestId == 0) return;
    std::lock_guard<std::mutex> lock(exportMutex);
    pendingExportProductRequests.push_back({nodeId, requestId, graphRevision});
}

std::vector<ExportProductsRequest> ViewportMailbox::takePendingExportProducts() {
    std::lock_guard<std::mutex> lock(exportMutex);
    std::vector<ExportProductsRequest> result;
    result.swap(pendingExportProductRequests);
    return result;
}
