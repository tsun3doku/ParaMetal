#include "UiIconRegistry.hpp"

#include <algorithm>
#include <cmath>

namespace ui {

std::vector<std::filesystem::path> IconRegistry::iconFolderCandidates(const std::string& folder) {
    return {
        std::filesystem::path("textures/icons") / folder,
        std::filesystem::path("ParaMetal/textures/icons") / folder,
        std::filesystem::path("../textures/icons") / folder,
        std::filesystem::path("../../textures/icons") / folder,
        std::filesystem::path("../ParaMetal/textures/icons") / folder,
        std::filesystem::path("../../ParaMetal/textures/icons") / folder
    };
}

int IconRegistry::parseIconWidth(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    if (name.size() < 2 || name.back() != 'w') {
        return 0;
    }

    try {
        return std::stoi(name.substr(0, name.size() - 1));
    } catch (...) {
        return 0;
    }
}

std::vector<std::pair<int, std::filesystem::path>> IconRegistry::iconVariantsForFolder(
    const QString& folder) {
    std::vector<std::pair<int, std::filesystem::path>> variants;
    const std::string folderName = folder.toStdString();

    for (const auto& folderPath : iconFolderCandidates(folderName)) {
        if (!std::filesystem::exists(folderPath) || !std::filesystem::is_directory(folderPath)) {
            continue;
        }

        for (const auto& sizeEntry : std::filesystem::directory_iterator(folderPath)) {
            if (!sizeEntry.is_directory()) {
                continue;
            }

            const int variantWidth = parseIconWidth(sizeEntry.path());
            if (variantWidth <= 0) {
                continue;
            }

            for (const auto& fileEntry : std::filesystem::directory_iterator(sizeEntry.path())) {
                if (!fileEntry.is_regular_file()) {
                    continue;
                }

                const std::filesystem::path extension = fileEntry.path().extension();
                if (extension == ".png" || extension == ".PNG") {
                    variants.emplace_back(variantWidth, fileEntry.path());
                    break;
                }
            }
        }
    }

    std::sort(variants.begin(), variants.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    return variants;
}

QString IconRegistry::nearestIconPathForFolder(const QString& folder, qreal targetPixelWidth) {
    const auto variants = iconVariantsForFolder(folder);
    if (variants.empty()) {
        return QString();
    }

    const int targetWidth = iconWidthCacheKey(targetPixelWidth);
    for (const auto& variant : variants) {
        if (variant.first >= targetWidth) {
            return QString::fromStdString(variant.second.string());
        }
    }
    return QString::fromStdString(variants.back().second.string());
}

QString IconRegistry::exactIconPathForFolder(const QString& folder, int pixelWidth) {
    const auto variants = iconVariantsForFolder(folder);
    const auto match = std::find_if(variants.begin(), variants.end(), [pixelWidth](const auto& variant) {
        return variant.first == pixelWidth;
    });
    return match == variants.end() ? QString() : QString::fromStdString(match->second.string());
}

QPixmap IconRegistry::pixmapForPath(const QString& iconPath) {
    if (iconPath.isEmpty()) {
        return QPixmap();
    }

    const std::string cacheKey = iconPath.toStdString();
    const auto cacheIt = pixmapCache.find(cacheKey);
    if (cacheIt != pixmapCache.end()) {
        return cacheIt->second;
    }

    QPixmap pixmap(iconPath);
    if (pixmap.isNull()) {
        return QPixmap();
    }

    pixmapCache.emplace(cacheKey, pixmap);
    return pixmap;
}

QPixmap IconRegistry::pixmapForFolder(const QString& folder, qreal targetPixelWidth) {
    return pixmapForPath(nearestIconPathForFolder(folder, targetPixelWidth));
}

QPixmap IconRegistry::screenSpacePixmapForFolder(const QString& folder,
                                                 qreal logicalWidth,
                                                 int sourcePixelWidth) {
    if (logicalWidth <= 0.0 || sourcePixelWidth <= 0) {
        return QPixmap();
    }

    QPixmap pixmap = pixmapForPath(exactIconPathForFolder(folder, sourcePixelWidth));
    if (pixmap.isNull()) {
        return QPixmap();
    }

    pixmap.setDevicePixelRatio(static_cast<qreal>(sourcePixelWidth) / logicalWidth);
    return pixmap;
}

int IconRegistry::iconWidthCacheKey(qreal targetPixelWidth) {
    return std::max(1, static_cast<int>(std::lround(targetPixelWidth)));
}

} // namespace ui
