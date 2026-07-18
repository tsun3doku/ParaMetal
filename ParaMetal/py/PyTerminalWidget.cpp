#include "PyTerminalWidget.hpp"

#include "PyInterpreter.hpp"
#include "nodegraph/ui/widgets/NodeGraphActionStrip.hpp"
#include "nodegraph/ui/widgets/NodeGraphWidgetStyle.hpp"
#include "ui/UiIconRegistry.hpp"
#include "ui/UiTypography.hpp"

#include <QPlainTextEdit>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QKeyEvent>
#include <QScrollBar>
#include <QFrame>
#include <QLabel>
#include <QWidget>

PyTerminalWidget::PyTerminalWidget(QWidget* parent)
    : QWidget(parent) {

    nodegraphwidgets::applyNodePanelStyle(this);
    setMinimumWidth(320);

    QVBoxLayout* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    // Title bar
    QFrame* titleBar = new QFrame(this);
    nodegraphwidgets::styleTitleBar(titleBar);
    QHBoxLayout* titleLayout = new QHBoxLayout(titleBar);
    titleLayout->setContentsMargins(10, 0, 0, 0);
    titleLayout->setSpacing(6);

    QLabel* titleIcon = new QLabel(titleBar);
    titleIcon->setFixedSize(20, 20);
    titleIcon->setStyleSheet(QStringLiteral("border: none; background: transparent;"));
    const QPixmap icon = ui::IconRegistry::screenSpacePixmapForFolder(
        QStringLiteral("Terminal"),
        20.0);
    if (!icon.isNull()) {
        titleIcon->setPixmap(icon);
    }
    titleLayout->addWidget(titleIcon);

    QLabel* titleLabel = new QLabel(QStringLiteral("Console"), titleBar);
    nodegraphwidgets::styleTitleLabel(titleLabel);
    titleLayout->addWidget(titleLabel);
    titleLayout->addStretch(1);
    rootLayout->addWidget(titleBar);

    // Output pane
    outputEdit = new QPlainTextEdit(this);
    outputEdit->setReadOnly(true);
    outputEdit->setLineWrapMode(QPlainTextEdit::WidgetWidth);
    outputEdit->setFrameStyle(QFrame::NoFrame);
    outputEdit->setFont(ui::UiTypography::font(ui::TextRole::Console, ui::FontFamily::Monospace));
    outputEdit->setStyleSheet(QStringLiteral(
        "QPlainTextEdit {"
        "  border: none;"
        "  background: %1;"
        "  padding: 6px 10px;"
        "  border-radius: 0px;"
        "}"
        "QPlainTextEdit:focus {"
        "  border: none;"
        "}"
    ).arg(nodegraphwidgets::colorPanelBackground.name()));
    rootLayout->addWidget(outputEdit, 1);

    highlighter = new PySyntaxHighlighter(outputEdit->document());

    // Separator between output and input
    QFrame* separator = new QFrame(this);
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Plain);
    separator->setFixedHeight(1);
    separator->setStyleSheet(QStringLiteral(
        "QFrame { background: %1; border: none; }"
    ).arg(nodegraphwidgets::colorPanelCardBorder.name()));
    rootLayout->addWidget(separator);

    QWidget* sampleGraphStripHost = new QWidget(this);
    sampleGraphStripHost->setObjectName(QStringLiteral("SampleGraphStripHost"));
    sampleGraphStripHost->setAttribute(Qt::WA_StyledBackground, true);
    sampleGraphStripHost->setStyleSheet(QStringLiteral(
        "QWidget#SampleGraphStripHost {"
        "  background: %1;"
        "}"
    ).arg(nodegraphwidgets::colorPanelBackground.name()));
    QVBoxLayout* sampleGraphStripLayout = new QVBoxLayout(sampleGraphStripHost);
    sampleGraphStripLayout->setContentsMargins(10, 14, 10, 14);
    sampleGraphStripLayout->setSpacing(0);

    sampleGraphStrip = new NodeGraphActionStrip(
        QStringLiteral("Sample graph available"),
        QStringLiteral("default_graph() creates a sample graph"),
        QStringLiteral("Run"),
        QStringLiteral("textures/preview.png"),
        sampleGraphStripHost);
    sampleGraphStripLayout->addWidget(sampleGraphStrip);
    rootLayout->addWidget(sampleGraphStripHost);

    QFrame* inputSeparator = new QFrame(this);
    inputSeparator->setFrameShape(QFrame::HLine);
    inputSeparator->setFrameShadow(QFrame::Plain);
    inputSeparator->setFixedHeight(1);
    inputSeparator->setStyleSheet(QStringLiteral(
        "QFrame { background: %1; border: none; }"
    ).arg(nodegraphwidgets::colorPanelCardBorder.name()));
    rootLayout->addWidget(inputSeparator);

    // Input line
    inputEdit = new QLineEdit(this);
    inputEdit->setPlaceholderText(QStringLiteral(">>>"));
    inputEdit->setFont(ui::UiTypography::font(ui::TextRole::Console, ui::FontFamily::Monospace));
    inputEdit->setStyleSheet(QStringLiteral(
        "QLineEdit {"
        "  background: %1;"
        "  color: %2;"
        "  border: none;"
        "  border-radius: 0px;"
        "  padding: 8px 10px;"
        "}"
        "QLineEdit:focus {"
        "  border: none;"
        "}"
        "QLineEdit::placeholder {"
        "  color: rgb(150,150,150);"
        "}"
    ).arg(nodegraphwidgets::colorPanelBackground.name(),
          nodegraphwidgets::colorTextInput.name()));
    rootLayout->addWidget(inputEdit);

    connect(inputEdit, &QLineEdit::returnPressed, this, &PyTerminalWidget::onInputReturnPressed);
    connect(sampleGraphStrip, &NodeGraphActionStrip::triggered, this, [this]() {
        sampleGraphStrip->hide();
        emit defaultGraphRequested();
    });
    connect(sampleGraphStrip, &NodeGraphActionStrip::dismissed, sampleGraphStrip, &NodeGraphActionStrip::hide);

    // Custom key handling for history navigation
    inputEdit->installEventFilter(this);
}

PyTerminalWidget::~PyTerminalWidget() = default;

bool PyTerminalWidget::eventFilter(QObject* obj, QEvent* event) {
    if (obj == inputEdit && event->type() == QEvent::KeyPress) {
        QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
        if (keyEvent->key() == Qt::Key_Up) {
            onInputHistoryUp();
            return true;
        }
        if (keyEvent->key() == Qt::Key_Down) {
            onInputHistoryDown();
            return true;
        }
        if (keyEvent->key() == Qt::Key_Tab) {
            // Insert 4 spaces
            int pos = inputEdit->cursorPosition();
            inputEdit->setText(inputEdit->text().insert(pos, QStringLiteral("    ")));
            inputEdit->setCursorPosition(pos + 4);
            return true;
        }
        if (keyEvent->key() == Qt::Key_Escape) {
            // Cancel the current input buffer and reset to a fresh prompt
            inputEdit->clear();
            currentBuffer.clear();
            waitingForMoreInput = false;
            appendOutput(QStringLiteral("^C\n"));
            appendPrompt(QStringLiteral(">>> "));
            return true;
        }
    }
    return QWidget::eventFilter(obj, event);
}

bool PyTerminalWidget::initializeInterpreter() {
    if (interpreter) {
        return true;
    }

    interpreter = std::make_unique<PyInterpreter>();

    if (!interpreter->initialize()) {
        appendOutput("Failed to initialize Python interpreter.\n");
        interpreter.reset();
        return false;
    }
    std::string ver = interpreter->pythonVersion();
    appendOutput(QStringLiteral("ParaMetal Console\n"));
    appendOutput(QStringLiteral("Python %1\n").arg(QString::fromStdString(ver)));
    appendOutput(QStringLiteral("parametal imported as pm\n\n"));
    appendOutput(QStringLiteral("Help: api(), registry()\n\n"));

    appendPrompt(">>> ");
    return true;
}

void PyTerminalWidget::shutdownInterpreter() {
    if (interpreter) {
        interpreter->shutdown();
        interpreter.reset();
    }
}

void PyTerminalWidget::onInputReturnPressed() {
    QString input = inputEdit->text();
    inputEdit->clear();

    if (!interpreter && !initializeInterpreter()) {
        return;
    }

    if (!waitingForMoreInput) {
        // New command
        currentBuffer = input;
        appendPrompt(">>> " + input);
    } else {
        // Continuation line
        currentBuffer += "\n" + input;
        appendPrompt("... " + input);
    }

    executeBuffer();
}

void PyTerminalWidget::executeBuffer() {
    bool more = interpreter->runSource(currentBuffer.toStdString());

    // Flush any buffered output from Python's execution
    std::string out = interpreter->consumeOutput();
    std::string err = interpreter->consumeError();
    if (!out.empty()) {
        appendOutput(QString::fromStdString(out));
    }
    if (!err.empty()) {
        appendOutput(QString::fromStdString(err));
    }

    if (more) {
        waitingForMoreInput = true;
        return;
    }

    if (!currentBuffer.isEmpty()) {
        commandHistory.append(currentBuffer);
        historyIndex = commandHistory.size();
    }
    currentBuffer.clear();
    waitingForMoreInput = false;
    appendPrompt(">>> ");
}

void PyTerminalWidget::onInputHistoryUp() {
    if (commandHistory.isEmpty() || historyIndex <= 0) {
        return;
    }
    historyIndex--;
    inputEdit->setText(commandHistory[historyIndex]);
    inputEdit->setCursorPosition(inputEdit->text().length());
}

void PyTerminalWidget::onInputHistoryDown() {
    if (commandHistory.isEmpty() || historyIndex >= commandHistory.size() - 1) {
        inputEdit->clear();
        historyIndex = commandHistory.size();
        return;
    }
    historyIndex++;
    inputEdit->setText(commandHistory[historyIndex]);
    inputEdit->setCursorPosition(inputEdit->text().length());
}

void PyTerminalWidget::appendOutput(const QString& text) {
    outputEdit->insertPlainText(text);
    QScrollBar* sb = outputEdit->verticalScrollBar();
    sb->setValue(sb->maximum());
}

void PyTerminalWidget::appendPrompt(const QString& prompt) {
    outputEdit->insertPlainText(prompt + "\n");
    QScrollBar* sb = outputEdit->verticalScrollBar();
    sb->setValue(sb->maximum());
}
