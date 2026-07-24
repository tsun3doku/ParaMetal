#include "UiFontRegistry.hpp"
#include "UiTypography.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QFont>
#include <QFontDatabase>
#include <QGuiApplication>

namespace ui {

QString UiFontRegistry::resolveBundledAsset(const QString& relativePath) {
    const QStringList candidates = {
        QDir(QCoreApplication::applicationDirPath()).filePath(relativePath),
        relativePath,
        QStringLiteral("ParaMetal/") + relativePath,
        QStringLiteral("../") + relativePath,
        QStringLiteral("../../") + relativePath,
        QStringLiteral("../ParaMetal/") + relativePath,
        QStringLiteral("../../ParaMetal/") + relativePath
    };

    for (const QString& candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return candidate;
        }
    }
    return {};
}

QString UiFontRegistry::loadBundledFontFamily(const QStringList& relativePaths) {
    QString family;
    for (const QString& relativePath : relativePaths) {
        const QString fontPath = resolveBundledAsset(relativePath);
        if (fontPath.isEmpty()) {
            continue;
        }

        const int fontId = QFontDatabase::addApplicationFont(fontPath);
        if (fontId < 0 || !family.isEmpty()) {
            continue;
        }

        const QStringList families = QFontDatabase::applicationFontFamilies(fontId);
        if (!families.isEmpty()) {
            family = families.first();
        }
    }
    return family;
}

void UiFontRegistry::installBundledFonts(QGuiApplication& application) {
    const QString outfitFamily = loadBundledFontFamily({
        QStringLiteral("fonts/Outfit/Outfit-Light.ttf"),
        QStringLiteral("fonts/Outfit/Outfit-Regular.ttf"),
        QStringLiteral("fonts/Outfit/Outfit-Medium.ttf"),
        QStringLiteral("fonts/Outfit/Outfit-SemiBold.ttf"),
        QStringLiteral("fonts/Outfit/Outfit-Bold.ttf")
    });
    loadBundledFontFamily({
        QStringLiteral("fonts/AzeretMono/AzeretMono-Regular.ttf"),
        QStringLiteral("fonts/AzeretMono/AzeretMono-Bold.ttf"),
        QStringLiteral("fonts/AzeretMono/AzeretMono-Italic.ttf"),
        QStringLiteral("fonts/AzeretMono/AzeretMono-BoldItalic.ttf")
    });

    if (!outfitFamily.isEmpty()) {
        QFont applicationFont = UiTypography::font(TextRole::Regular);
        applicationFont.setFamily(outfitFamily);
        application.setFont(applicationFont);
    }
}

}
