#pragma once

#include <QWidget>

class QLabel;
class QPixmap;
class QPushButton;
class QString;

class NodeGraphActionStrip : public QWidget {
    Q_OBJECT
public:
    explicit NodeGraphActionStrip(
        const QString& title,
        const QString& description,
        const QString& actionText,
        const QString& previewPath,
        QWidget* parent = nullptr);

signals:
    void triggered();
    void dismissed();

private:
    static QPixmap loadPreview(const QString& relativePath);

    QLabel* previewLabel = nullptr;
    QLabel* titleLabel = nullptr;
    QLabel* descriptionLabel = nullptr;
    QPushButton* actionButton = nullptr;
    QPushButton* dismissButton = nullptr;
};
