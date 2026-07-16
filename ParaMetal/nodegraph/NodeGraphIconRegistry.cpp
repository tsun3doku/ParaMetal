#include "NodeGraphIconRegistry.hpp"

#include <algorithm>
#include <cmath>

const char* NodeGraphIconRegistry::iconFolderForType(const NodeTypeId& typeId) {
    if (typeId == "contact") return "Contact";
    if (typeId == "heat_solve") return "HeatSystem";
    if (typeId == "model") return "Model";
    if (typeId == "heat_model") return "HeatModel";
    if (typeId == "remesh") return "Remesh";
    if (typeId == "transform") return "Transform";
    if (typeId == "voronoi") return "VoronoiSystem";
    if (typeId == "mesh_points") return "Points";
    return nullptr;
}

std::vector<std::filesystem::path> NodeGraphIconRegistry::iconFolderCandidates(const std::string& folder) {
    return {
        std::filesystem::path("textures/icons") / folder,
        std::filesystem::path("ParaMetal/textures/icons") / folder,
        std::filesystem::path("../textures/icons") / folder,
        std::filesystem::path("../../textures/icons") / folder,
        std::filesystem::path("../ParaMetal/textures/icons") / folder,
        std::filesystem::path("../../ParaMetal/textures/icons") / folder
    };
}

int NodeGraphIconRegistry::parseIconWidth(const std::filesystem::path& path) {
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

std::vector<std::pair<int, std::filesystem::path>> NodeGraphIconRegistry::iconVariantsForFolder(const QString& folder) {
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

    std::sort(
        variants.begin(),
        variants.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });
    return variants;
}

QString NodeGraphIconRegistry::nearestIconPathForFolder(const QString& folder, qreal targetPixelWidth) {
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

QString NodeGraphIconRegistry::exactIconPathForFolder(const QString& folder, int pixelWidth) {
    const auto variants = iconVariantsForFolder(folder);
    const auto match = std::find_if(
        variants.begin(),
        variants.end(),
        [pixelWidth](const auto& variant) {
            return variant.first == pixelWidth;
        });
    return match == variants.end()
        ? QString()
        : QString::fromStdString(match->second.string());
}

QPixmap NodeGraphIconRegistry::pixmapForPath(const QString& iconPath) {
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

QPixmap NodeGraphIconRegistry::nodePixmapForType(const NodeTypeId& typeId, qreal targetPixelWidth) {
    const char* folder = iconFolderForType(typeId);
    if (!folder) {
        return QPixmap();
    }
    return pixmapForPath(nearestIconPathForFolder(QString::fromUtf8(folder), targetPixelWidth));
}

QPixmap NodeGraphIconRegistry::screenSpaceNodePixmapForType(const NodeTypeId& typeId, qreal logicalWidth) {
    const char* folder = iconFolderForType(typeId);
    if (!folder) {
        return QPixmap();
    }
    return screenSpacePixmapForFolder(QString::fromUtf8(folder), logicalWidth);
}

QPixmap NodeGraphIconRegistry::screenSpacePixmapForFolder(const QString& folder, qreal logicalWidth) {
    if (logicalWidth <= 0.0) {
        return QPixmap();
    }

    QPixmap pixmap = pixmapForPath(exactIconPathForFolder(folder, screenSpaceIconSourceWidth));
    if (pixmap.isNull()) {
        return QPixmap();
    }

    pixmap.setDevicePixelRatio(static_cast<qreal>(screenSpaceIconSourceWidth) / logicalWidth);
    return pixmap;
}

int NodeGraphIconRegistry::iconWidthCacheKey(qreal targetPixelWidth) {
    return std::max(1, static_cast<int>(std::lround(targetPixelWidth)));
}
