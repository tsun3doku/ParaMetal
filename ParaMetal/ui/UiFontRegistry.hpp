#pragma once

#include <QString>
#include <QStringList>

class QGuiApplication;

namespace ui {

class UiFontRegistry {
public:
    UiFontRegistry() = delete;

    static void installBundledFonts(QGuiApplication& application);

private:
    static QString resolveBundledAsset(const QString& relativePath);
    static QString loadBundledFontFamily(const QStringList& relativePaths);
};

}
