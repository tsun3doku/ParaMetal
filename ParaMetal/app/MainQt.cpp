#include "MainQt.h"
#include "App.h"
#include "nodegraph/NodeGraphBridge.hpp"
#include "nodegraph/NodeGraphSave.hpp"
#include "nodegraph/ui/scene/NodeGraphEditorWidget.hpp"
#include "py/PyBridge.hpp"
#include "py/PyTerminalWidget.hpp"
#include "scene/Camera.hpp"
#include "scene/CameraController.hpp"
#include "util/UiTheme.hpp"
#include "VulkanWindow.hpp"

#include <QAction>
#include <QApplication>
#include <QCloseEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSettings>
#include <QSizePolicy>
#include <QSplitter>
#include <QSplitterHandle>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>

#ifdef Q_OS_WIN
#define NOMINMAX
#include <windows.h>
#endif

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("ParaMetal");
    resize(1600, 900);
    setMinimumSize(700, 400);
    setStyleSheet(QString::fromStdString(ui::appStyleSheet()));

    createMenuBar();
    createNodeGraphEditorWidget();

    QWidget* centralHost = new QWidget(this);
    QHBoxLayout* hostLayout = new QHBoxLayout(centralHost);
    hostLayout->setContentsMargins(0, 0, 0, 0);
    hostLayout->setSpacing(0);

    mainSplitter = new QSplitter(Qt::Horizontal, centralHost);
    ui::configureSplitter(*mainSplitter);
    mainSplitter->setOpaqueResize(true);
    mainSplitter->setChildrenCollapsible(false);
#ifdef Q_OS_WIN
    mainSplitter->setAttribute(Qt::WA_NativeWindow, true);
#endif    
    hostLayout->addWidget(mainSplitter);

    if (nodeGraphEditor) {
        nodeGraphEditor->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
        mainSplitter->addWidget(nodeGraphEditor);
    }

    QWidget* viewportHost = new QWidget(mainSplitter);
    QVBoxLayout* viewportLayout = new QVBoxLayout(viewportHost);
    viewportLayout->setContentsMargins(0, 0, 0, 0);
    viewportLayout->setSpacing(0);
    viewportWindow = new VulkanWindow();
    viewportContainer = QWidget::createWindowContainer(viewportWindow, viewportHost);
    viewportContainer->setFocusPolicy(Qt::StrongFocus);
    viewportContainer->setMinimumSize(320, 240);
    viewportLayout->addWidget(viewportContainer);
    viewportHost->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    mainSplitter->addWidget(viewportHost);

    mainSplitter->setStretchFactor(0, 0);
    mainSplitter->setStretchFactor(1, 1);
    mainSplitter->setSizes({700, 900});

    setCentralWidget(centralHost);
    connect(mainSplitter, &QSplitter::splitterMoved, this, [this](int, int) {
        raiseNativeSplitterHandles();
    });

    QTimer* uiSyncTimer = new QTimer(this);
    uiSyncTimer->setInterval(100);
    connect(uiSyncTimer, &QTimer::timeout, this, &MainWindow::syncNodeGraphBridge);
    uiSyncTimer->start();
    
    QTimer::singleShot(0, this, [this]() {
        raiseNativeSplitterHandles();        
#ifdef Q_OS_WIN
        if (viewportWindow) {
            const HWND vulkanHwnd = reinterpret_cast<HWND>(viewportWindow->winId());
            if (vulkanHwnd) {
                LONG_PTR exStyle = GetWindowLongPtr(vulkanHwnd, GWL_EXSTYLE);
                SetWindowLongPtr(vulkanHwnd, GWL_EXSTYLE, exStyle | WS_EX_NOREDIRECTIONBITMAP);
            }
        }
#endif
    });
}

MainWindow::~MainWindow() {
}

void MainWindow::setApp(App* application) {
    app = application;
    if (viewportWindow) {
        viewportWindow->setApp(app);
    }
    if (nodeGraphEditor) {
        boundRuntimeQuery = app ? app->runtimeQuery() : nullptr;
        nodeGraphEditor->setRuntimeQuery(boundRuntimeQuery);
        boundSceneController = app ? app->getSceneController() : nullptr;
        nodeGraphEditor->setSceneController(boundSceneController);
        boundModelSelection = app ? app->getModelSelection() : nullptr;
        nodeGraphEditor->setModelSelection(boundModelSelection);
    }
    syncNodeGraphBridge();
    if (NodeGraphBridge* bridge = app ? app->getNodeGraphBridge() : nullptr) {
        lastSavedRevision = bridge->getRevision();
        setProjectModified(false);
    }
}

void MainWindow::raiseNativeSplitterHandles() {
#ifdef Q_OS_WIN
    if (!mainSplitter) {
        return;
    }

    for (int i = 1; i < mainSplitter->count(); ++i) {
        QSplitterHandle* handle = mainSplitter->handle(i);
        if (!handle) {
            continue;
        }

        handle->setAttribute(Qt::WA_NativeWindow, true);
        const WId handleId = handle->winId();
        if (!handleId) {
            continue;
        }

        SetWindowPos(
            reinterpret_cast<HWND>(handleId),
            HWND_TOP,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    }
#endif
}

void MainWindow::createMenuBar() {
    QMenuBar* menuBar = new QMenuBar(this);
    setMenuBar(menuBar);

    createFileMenu(menuBar);

    QMenu* viewMenu = menuBar->addMenu("&View");

    nodeGraphAction = new QAction("Node &Graph", this);
    nodeGraphAction->setCheckable(true);
    nodeGraphAction->setChecked(true);
    connect(nodeGraphAction, &QAction::toggled, this, [this](bool checked) {
        setNodeGraphVisible(checked);
    });
    viewMenu->addAction(nodeGraphAction);

    pyTerminalAction = new QAction("Python &Terminal", this);
    pyTerminalAction->setCheckable(true);
    pyTerminalAction->setChecked(true);
    connect(pyTerminalAction, &QAction::toggled, this, [this](bool checked) {
        setPyTerminalVisible(checked);
    });
    viewMenu->addAction(pyTerminalAction);
}

void MainWindow::createFileMenu(QMenuBar* menuBar) {
    QMenu* fileMenu = menuBar->addMenu("&File");

    QAction* newAction = new QAction("&New", this);
    newAction->setShortcut(QKeySequence::New);
    connect(newAction, &QAction::triggered, this, [this]() {
        newProject();
    });
    fileMenu->addAction(newAction);

    QAction* openAction = new QAction("&Open...", this);
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, [this]() {
        openProject();
    });
    fileMenu->addAction(openAction);

    fileMenu->addSeparator();

    saveAction = new QAction("&Save", this);
    saveAction->setShortcut(QKeySequence::Save);
    connect(saveAction, &QAction::triggered, this, [this]() {
        saveProject();
    });
    fileMenu->addAction(saveAction);

    saveAsAction = new QAction("Save &As...", this);
    saveAsAction->setShortcut(QKeySequence::SaveAs);
    connect(saveAsAction, &QAction::triggered, this, [this]() {
        saveProjectAs();
    });
    fileMenu->addAction(saveAsAction);

    fileMenu->addSeparator();

    recentFilesMenu = fileMenu->addMenu("Recent &Files");
    rebuildRecentFilesMenu();

    fileMenu->addSeparator();

    QAction* exitAction = new QAction("E&xit", this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &MainWindow::onExit);
    fileMenu->addAction(exitAction);

    updateWindowTitle();
}

void MainWindow::createNodeGraphEditorWidget() {
    nodeGraphEditor = new NodeGraphEditorWidget(this);
    nodeGraphEditor->setObjectName("NodeGraphEditorWidget");
    nodeGraphEditor->setRuntimeQuery(app ? app->runtimeQuery() : nullptr);

    syncNodeGraphBridge();
}

void MainWindow::syncNodeGraphBridge() {
    if (app) {
        const auto errors = app->consumeSimulationErrors();
        for (const auto& error : errors) {
            QMessageBox::critical(
                this,
                "Simulation Validation Failed",
                QString::fromStdString(error.message));
        }
    }

    if (!nodeGraphEditor) {
        return;
    }

    const RuntimeQuery* runtimeQuery = app ? app->runtimeQuery() : nullptr;
    if (runtimeQuery != boundRuntimeQuery) {
        nodeGraphEditor->setRuntimeQuery(runtimeQuery);
        boundRuntimeQuery = runtimeQuery;
    }

    const SceneController* sceneController = app ? app->getSceneController() : nullptr;
    if (sceneController != boundSceneController) {
        nodeGraphEditor->setSceneController(sceneController);
        boundSceneController = sceneController;
    }

    ModelSelection* modelSelection = app ? app->getModelSelection() : nullptr;
    if (modelSelection != boundModelSelection) {
        nodeGraphEditor->setModelSelection(modelSelection);
        boundModelSelection = modelSelection;
    }

    if (!app) {
        nodeGraphEditor->syncSelection();
        return;
    }

    NodeGraphBridge* bridge = app->getNodeGraphBridge();
    if (bridge == boundNodeGraphBridge) {
        if (bridge && bridge->getRevision() != lastSavedRevision) {
            setProjectModified(true);
        }
        nodeGraphEditor->syncSelection();
        return;
    }

    nodeGraphEditor->setBridge(bridge);
    boundNodeGraphBridge = bridge;
    if (bridge) {
        lastSavedRevision = bridge->getRevision();
        setProjectModified(false);
    }
    pybridge::setBridge(bridge);
    if (bridge) {
        if (PyTerminalWidget* term = nodeGraphEditor->getPyTerminal()) {
            term->initializeInterpreter();
        }
    }
    nodeGraphEditor->syncSelection();
}

void MainWindow::setNodeGraphVisible(bool visible) {
    if (!mainSplitter || !nodeGraphEditor) {
        return;
    }

    QList<int> sizes = mainSplitter->sizes();
    if (sizes.size() < 2) {
        return;
    }

    if (visible) {
        nodeGraphEditor->show();
        if (sizes[0] <= 1) {
            sizes[0] = std::max(nodeGraphEditor->minimumWidth(), lastNodeGraphWidth);
            mainSplitter->setSizes(sizes);
        }
        return;
    }

    lastNodeGraphWidth = std::max(nodeGraphEditor->minimumWidth(), sizes[0]);
    nodeGraphEditor->hide();
}

void MainWindow::setPyTerminalVisible(bool visible) {
    if (!nodeGraphEditor) {
        return;
    }
    PyTerminalWidget* term = nodeGraphEditor->getPyTerminal();
    if (!term) {
        return;
    }
    if (visible) {
        term->show();
    } else {
        term->hide();
    }
}

bool MainWindow::newProject() {
    if (!promptSaveIfModified()) {
        return false;
    }

    if (NodeGraphBridge* bridge = app ? app->getNodeGraphBridge() : nullptr) {
        bridge->clear();
    }
    if (nodeGraphEditor) {
        nodeGraphEditor->refreshGraph();
    }

    currentProjectPath.clear();
    if (NodeGraphBridge* bridge = app ? app->getNodeGraphBridge() : nullptr) {
        lastSavedRevision = bridge->getRevision();
    }
    setProjectModified(false);
    return true;
}

bool MainWindow::openProject() {
    if (!promptSaveIfModified()) {
        return false;
    }

    const QString path = QFileDialog::getOpenFileName(
        this,
        "Open ParaMetal Project",
        currentProjectPath.isEmpty() ? QString() : QFileInfo(currentProjectPath).absolutePath(),
        "ParaMetal Projects (*.pm);;All Files (*)");
    if (path.isEmpty()) {
        return false;
    }

    return openProjectPath(path);
}

bool MainWindow::openProjectPath(const QString& path) {
    NodeGraphBridge* bridge = app ? app->getNodeGraphBridge() : nullptr;
    if (!bridge) {
        QMessageBox::critical(this, "Open Project", "Node graph is not available.");
        return false;
    }

    NodeGraphSave::Data data;
    QString error;
    if (!NodeGraphSave::load(data, path, &error)) {
        QMessageBox::critical(this, "Open Project", error);
        return false;
    }

    std::string graphError;
    if (!bridge->loadState(data.graph, data.nextNodeId, data.nextSocketId, data.nextEdgeId, graphError)) {
        QMessageBox::critical(this, "Open Project", QString::fromStdString(graphError));
        return false;
    }

    if (CameraController* cameraController = app ? app->getCameraController() : nullptr) {
        cameraController->setCameraState(
            data.viewport.lookAt,
            data.viewport.orientation,
            data.viewport.radius,
            data.viewport.fov);
    }

    currentProjectPath = path;
    lastSavedRevision = bridge->getRevision();
    setProjectModified(false);
    addToRecentFiles(path);
    if (nodeGraphEditor) {
        nodeGraphEditor->refreshGraph();
    }
    return true;
}

bool MainWindow::saveProject() {
    if (currentProjectPath.isEmpty()) {
        return saveProjectAs();
    }
    return saveProjectToPath(currentProjectPath);
}

bool MainWindow::saveProjectAs() {
    const QString path = QFileDialog::getSaveFileName(
        this,
        "Save ParaMetal Project",
        currentProjectPath.isEmpty() ? QString("untitled.pm") : currentProjectPath,
        "ParaMetal Projects (*.pm);;All Files (*)");
    if (path.isEmpty()) {
        return false;
    }

    QString savePath = path;
    if (QFileInfo(savePath).suffix().isEmpty()) {
        savePath += ".pm";
    }
    return saveProjectToPath(savePath);
}

bool MainWindow::saveProjectToPath(const QString& path) {
    NodeGraphBridge* bridge = app ? app->getNodeGraphBridge() : nullptr;
    if (!bridge) {
        QMessageBox::critical(this, "Save Project", "Node graph is not available.");
        return false;
    }

    NodeGraphSave::Data data;
    data.graph = bridge->state();
    bridge->getNextIds(data.nextNodeId, data.nextSocketId, data.nextEdgeId);

    if (CameraController* cameraController = app ? app->getCameraController() : nullptr) {
        const Camera& camera = cameraController->getCamera();
        data.viewport.lookAt = camera.getLookAt();
        data.viewport.orientation = camera.getOrientation();
        data.viewport.radius = camera.getRadius();
        data.viewport.fov = camera.getBaseFov();
    }

    QString error;
    if (!NodeGraphSave::save(data, path, &error)) {
        QMessageBox::critical(this, "Save Project", error);
        return false;
    }

    currentProjectPath = path;
    lastSavedRevision = bridge->getRevision();
    setProjectModified(false);
    addToRecentFiles(path);
    return true;
}

bool MainWindow::promptSaveIfModified() {
    if (!isModified) {
        return true;
    }

    const QMessageBox::StandardButton result = QMessageBox::question(
        this,
        "Unsaved Changes",
        "Save changes to the current ParaMetal project?",
        QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
        QMessageBox::Save);

    if (result == QMessageBox::Cancel) {
        return false;
    }
    if (result == QMessageBox::Save) {
        return saveProject();
    }
    return true;
}

void MainWindow::updateWindowTitle() {
    QString title = "ParaMetal";
    if (!currentProjectPath.isEmpty()) {
        title += " - " + QFileInfo(currentProjectPath).fileName();
    }
    if (isModified) {
        title += "*";
    }
    setWindowTitle(title);

    if (saveAction) {
        saveAction->setEnabled(isModified);
    }
    if (saveAsAction) {
        saveAsAction->setEnabled(true);
    }
}

void MainWindow::setProjectModified(bool modified) {
    if (isModified == modified) {
        updateWindowTitle();
        return;
    }
    isModified = modified;
    updateWindowTitle();
}

void MainWindow::addToRecentFiles(const QString& path) {
    QStringList files = QSettings().value("recentFiles").toStringList();
    files.removeAll(path);
    files.prepend(path);
    while (files.size() > 10) {
        files.removeLast();
    }
    QSettings().setValue("recentFiles", files);
    rebuildRecentFilesMenu();
}

void MainWindow::rebuildRecentFilesMenu() {
    if (!recentFilesMenu) {
        return;
    }

    recentFilesMenu->clear();
    const QStringList files = QSettings().value("recentFiles").toStringList();
    for (const QString& path : files) {
        QAction* action = recentFilesMenu->addAction(QFileInfo(path).fileName());
        action->setToolTip(path);
        connect(action, &QAction::triggered, this, [this, path]() {
            if (!promptSaveIfModified()) {
                return;
            }
            if (!QFileInfo::exists(path)) {
                QMessageBox::warning(this, "Open Project", "Recent project file no longer exists.");
                QStringList files = QSettings().value("recentFiles").toStringList();
                files.removeAll(path);
                QSettings().setValue("recentFiles", files);
                rebuildRecentFilesMenu();
                return;
            }
            openProjectPath(path);
        });
    }
    if (files.empty()) {
        QAction* emptyAction = recentFilesMenu->addAction("(No Recent Files)");
        emptyAction->setEnabled(false);
    }
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (!promptSaveIfModified()) {
        event->ignore();
        return;
    }

    if (app) {
        app->shutdown();
    }

    event->accept();
}

void MainWindow::onExit() {
    close();
}

int main(int argc, char** argv) {
    QApplication qapp(argc, argv);
    QApplication::setApplicationName("ParaMetal");
    QApplication::setOrganizationName("ParaMetal");

    MainWindow mainWindow;
    App app;
    mainWindow.setApp(&app);
    mainWindow.show();

    return qapp.exec();
}
