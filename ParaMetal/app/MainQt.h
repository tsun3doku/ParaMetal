#pragma once

#include <QMainWindow>
#include <QString>

#include <cstdint>

class App;
class QAction;
class ModelSelection;
class NodeGraphController;
class NodeGraphEditorWidget;
class NodeGraph;
class PyTerminalWidget;
class RuntimeQuery;
class SceneController;
class TimelineNodeController;
class QCloseEvent;
class QWidget;
class VulkanWindow;
class QSplitter;
class QMenu;
class QMenuBar;
class TimelineWidget;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);

    void setApp(App* application);

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void createMenuBar();
    void createFileMenu(QMenuBar* menuBar);
    void createNodeGraphEditorWidget();
    void syncNodeGraph();
    void setNodeGraphVisible(bool visible);
    void setPyTerminalVisible(bool visible);
    void raiseNativeSplitterHandles();
    void onExit();
    bool newProject();
    bool openProject();
    bool openProjectPath(const QString& path);
    bool saveProject();
    bool saveProjectAs();
    bool saveProjectToPath(const QString& path);
    bool promptSaveIfModified();
    void updateWindowTitle();
    void setProjectModified(bool modified);
    void addToRecentFiles(const QString& path);
    void rebuildRecentFilesMenu();

    App* app = nullptr;

    QString currentProjectPath;
    uint64_t lastSavedRevision = 0;
    bool isModified = false;
    QAction* saveAction = nullptr;
    QAction* saveAsAction = nullptr;
    QMenu* recentFilesMenu = nullptr;
    QAction* nodeGraphAction = nullptr;
    QAction* pyTerminalAction = nullptr;
    VulkanWindow* viewportWindow = nullptr;
    QWidget* viewportContainer = nullptr;
    NodeGraphEditorWidget* nodeGraphEditor = nullptr;
    PyTerminalWidget* pyTerminal = nullptr;
    NodeGraph* boundNodeGraph = nullptr;
    const SceneController* boundSceneController = nullptr;
    ModelSelection* boundModelSelection = nullptr;
    const RuntimeQuery* boundRuntimeQuery = nullptr;
    NodeGraphController* boundNodeGraphController = nullptr;
    QSplitter* mainSplitter = nullptr;
    TimelineWidget* timelineWidget = nullptr;
    TimelineNodeController* timelineNodeController = nullptr;
};
