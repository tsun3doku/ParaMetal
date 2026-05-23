#include "MainQt.h"
#include "App.h"
#include "nodegraph/ui/scene/NodeGraphEditorWidget.hpp"
#include "py/PyBridge.hpp"
#include "py/PyTerminalWidget.hpp"
#include "util/UiTheme.hpp"
#include "VulkanWindow.hpp"

#include <QAction>
#include <QApplication>
#include <QCloseEvent>
#include <QHBoxLayout>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSizePolicy>
#include <QSplitter>
#include <QSplitterHandle>
#include <QString>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>

#ifdef Q_OS_WIN
#define NOMINMAX
#include <windows.h>
#endif

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("HeatSpectra");
    resize(1600, 900);
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
        nodeGraphEditor->setMinimumWidth(160);
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
    mainSplitter->setSizes({600, 1000});

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

    QMenu* fileMenu = menuBar->addMenu("&File");

    fileMenu->addSeparator();

    QAction* exitAction = new QAction("E&xit", this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &MainWindow::onExit);
    fileMenu->addAction(exitAction);

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
        nodeGraphEditor->syncSelection();
        return;
    }

    nodeGraphEditor->setBridge(bridge);
    boundNodeGraphBridge = bridge;
    pybridge::setBridge(bridge);
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

void MainWindow::closeEvent(QCloseEvent* event) {
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

    MainWindow mainWindow;
    App app;
    mainWindow.setApp(&app);
    mainWindow.show();

    return qapp.exec();
}
