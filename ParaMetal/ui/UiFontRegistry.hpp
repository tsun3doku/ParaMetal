#pragma once

#include <QString>
#include <QStringList>

class QApplication;

namespace ui {

class UiFontRegistry {
public:
    UiFontRegistry() = delete;

    static void installBundledFonts(QApplication& application);

private:
    static QString resolveBundledAsset(const QString& relativePath);
    static QString loadBundledFontFamily(const QStringList& relativePaths);
};

}
