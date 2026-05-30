#pragma once

#include "PySyntaxHighlighter.hpp"

#include <QWidget>
#include <QStringList>
#include <memory>

class QPlainTextEdit;
class QLineEdit;
class QEvent;
class NodeGraphActionStrip;
class PyInterpreter;

class PyTerminalWidget : public QWidget {
    Q_OBJECT
public:
    explicit PyTerminalWidget(QWidget* parent = nullptr);
    ~PyTerminalWidget();

    bool initializeInterpreter();
    void shutdownInterpreter();

signals:
    void defaultGraphRequested();

protected:
    bool eventFilter(QObject* obj, QEvent* event) override;

private slots:
    void onInputReturnPressed();
    void onInputHistoryUp();
    void onInputHistoryDown();

private:
    void appendOutput(const QString& text);
    void appendPrompt(const QString& prompt);
    void executeBuffer();

    QPlainTextEdit* outputEdit = nullptr;
    QLineEdit* inputEdit = nullptr;
    NodeGraphActionStrip* sampleGraphStrip = nullptr;
    PySyntaxHighlighter* highlighter = nullptr;

    std::unique_ptr<PyInterpreter> interpreter;
    QStringList commandHistory;
    int historyIndex = -1;
    QString currentBuffer;
    bool waitingForMoreInput = false;
};
