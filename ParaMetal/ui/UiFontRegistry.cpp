#include "UiFontRegistry.hpp"
#include "UiTypography.hpp"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QFont>
#include <QFontDatabase>

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

void UiFontRegistry::installBundledFonts(QApplication& application) {
    const QString urbanistFamily = loadBundledFontFamily({
        QStringLiteral("fonts/Urbanist/Urbanist-Light.ttf"),
        QStringLiteral("fonts/Urbanist/Urbanist-Regular.ttf"),
        QStringLiteral("fonts/Urbanist/Urbanist-Medium.ttf"),
        QStringLiteral("fonts/Urbanist/Urbanist-SemiBold.ttf"),
        QStringLiteral("fonts/Urbanist/Urbanist-Bold.ttf"),
        QStringLiteral("fonts/Urbanist/Urbanist-LightItalic.ttf"),
        QStringLiteral("fonts/Urbanist/Urbanist-Italic.ttf"),
        QStringLiteral("fonts/Urbanist/Urbanist-MediumItalic.ttf"),
        QStringLiteral("fonts/Urbanist/Urbanist-SemiBoldItalic.ttf"),
        QStringLiteral("fonts/Urbanist/Urbanist-BoldItalic.ttf")
    });
    loadBundledFontFamily({
        QStringLiteral("fonts/AzeretMono/AzeretMono-Regular.ttf"),
        QStringLiteral("fonts/AzeretMono/AzeretMono-Bold.ttf"),
        QStringLiteral("fonts/AzeretMono/AzeretMono-Italic.ttf"),
        QStringLiteral("fonts/AzeretMono/AzeretMono-BoldItalic.ttf")
    });

    if (!urbanistFamily.isEmpty()) {
        QFont applicationFont = UiTypography::font(TextRole::Regular);
        applicationFont.setFamily(urbanistFamily);
        application.setFont(applicationFont);
    }
}

}
