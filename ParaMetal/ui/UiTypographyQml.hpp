#pragma once

#include <QObject>
#include <QFont>
#include <QString>

namespace ui {

class UiTypographyQml final : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString fontFamily READ fontFamily CONSTANT)
    Q_PROPERTY(QString monoFamily READ monoFamily CONSTANT)
    Q_PROPERTY(int titleFontSize READ titleFontSize CONSTANT)
    Q_PROPERTY(int regularFontSize READ regularFontSize CONSTANT)
    Q_PROPERTY(int regularFontWeight READ regularFontWeight CONSTANT)
    Q_PROPERTY(int descriptionFontSize READ descriptionFontSize CONSTANT)
    Q_PROPERTY(int consoleFontSize READ consoleFontSize CONSTANT)
    Q_PROPERTY(int nodeTitleFontSize READ nodeTitleFontSize CONSTANT)
    Q_PROPERTY(QFont regularFont READ regularFont CONSTANT)
    Q_PROPERTY(QFont descriptionFont READ descriptionFont CONSTANT)
    Q_PROPERTY(QFont nodeTitleFont READ nodeTitleFont CONSTANT)

public:
    explicit UiTypographyQml(QObject* parent = nullptr) : QObject(parent) {}

    QString fontFamily() const;
    QString monoFamily() const;
    int titleFontSize() const;
    int regularFontSize() const;
    int regularFontWeight() const;
    int descriptionFontSize() const;
    int consoleFontSize() const;
    int nodeTitleFontSize() const;
    QFont regularFont() const;
    QFont descriptionFont() const;
    QFont nodeTitleFont() const;
};

}
