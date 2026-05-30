#pragma once

#include <QWidget>

class QLabel;
class QPushButton;
class QString;

class NodeGraphActionStrip : public QWidget {
    Q_OBJECT
public:
    explicit NodeGraphActionStrip(
        const QString& title,
        const QString& description,
        const QString& actionText,
        const QString& iconFolder,
        QWidget* parent = nullptr);

signals:
    void triggered();
    void dismissed();

private:
    QLabel* iconLabel = nullptr;
    QLabel* titleLabel = nullptr;
    QLabel* descriptionLabel = nullptr;
    QPushButton* actionButton = nullptr;
    QPushButton* dismissButton = nullptr;
};
