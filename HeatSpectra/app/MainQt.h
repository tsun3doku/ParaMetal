#pragma once

#include <QMainWindow>

class App;
class QAction;
class ModelSelection;
class NodeGraphEditorWidget;
class NodeGraphBridge;
class PyTerminalWidget;
class RuntimeQuery;
class SceneController;
class QCloseEvent;
class QWidget;
class VulkanWindow;
class QSplitter;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

    void setApp(App* application);

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void createMenuBar();
    void createNodeGraphEditorWidget();
    void syncNodeGraphBridge();
    void setNodeGraphVisible(bool visible);
    void setPyTerminalVisible(bool visible);
    void raiseNativeSplitterHandles();
    void onExit();

    App* app = nullptr;

    QAction* nodeGraphAction = nullptr;
    QAction* pyTerminalAction = nullptr;
    VulkanWindow* viewportWindow = nullptr;
    QWidget* viewportContainer = nullptr;
    NodeGraphEditorWidget* nodeGraphEditor = nullptr;
    NodeGraphBridge* boundNodeGraphBridge = nullptr;
    const SceneController* boundSceneController = nullptr;
    ModelSelection* boundModelSelection = nullptr;
    const RuntimeQuery* boundRuntimeQuery = nullptr;
    QSplitter* mainSplitter = nullptr;
    int lastNodeGraphWidth = 320;
};
