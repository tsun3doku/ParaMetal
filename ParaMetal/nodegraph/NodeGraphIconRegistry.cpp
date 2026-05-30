#include "NodeGraphIconRegistry.hpp"

#include <algorithm>
#include <filesystem>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

const char* NodeGraphIconRegistry::iconFolderForType(const NodeTypeId& typeId) {
    if (typeId == "contact") return "Contact";
    if (typeId == "heat_solve") return "HeatSystem";
    if (typeId == "model") return "Model";
    if (typeId == "heat_model") return "HeatModel";
    if (typeId == "remesh") return "Remesh";
    if (typeId == "transform") return "Transform";
    if (typeId == "voronoi") return "VoronoiSystem";
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

QString NodeGraphIconRegistry::iconPathForFolder(const QString& folder, qreal targetPixelWidth) {
    std::vector<std::pair<int, std::filesystem::path>> variants;
    const std::string folderName = folder.toStdString();
    const int targetWidth = iconWidthCacheKey(targetPixelWidth);

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

    if (variants.empty()) {
        return QString();
    }

    std::sort(
        variants.begin(),
        variants.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

    for (const auto& variant : variants) {
        if (variant.first >= targetWidth) {
            return QString::fromStdString(variant.second.string());
        }
    }

    return QString::fromStdString(variants.back().second.string());
}

bool IconCacheKey::operator==(const IconCacheKey& other) const noexcept {
    return typeId == other.typeId && iconWidthKey == other.iconWidthKey;
}

std::size_t IconCacheKeyHash::operator()(const IconCacheKey& key) const noexcept {
    const std::size_t typeHash = std::hash<NodeTypeId>{}(key.typeId);
    const std::size_t widthHash = std::hash<int>{}(key.iconWidthKey);
    return typeHash ^ (widthHash + 0x9e3779b9 + (typeHash << 6) + (typeHash >> 2));
}

QString NodeGraphIconRegistry::iconPathForType(const NodeTypeId& typeId, qreal targetPixelWidth) {
    const char* folder = iconFolderForType(typeId);
    if (!folder) {
        return QString();
    }

    return iconPathForFolder(QString::fromUtf8(folder), targetPixelWidth);
}

QPixmap NodeGraphIconRegistry::iconForType(const NodeTypeId& typeId, qreal targetPixelWidth) {
    const QString iconPath = iconPathForType(typeId, targetPixelWidth);
    if (iconPath.isEmpty()) {
        return QPixmap();
    }

    const std::filesystem::path iconFilePath(iconPath.toStdString());
    const IconCacheKey cacheKey{
        typeId,
        parseIconWidth(iconFilePath.parent_path().filename())
    };
    const auto cacheIt = iconCache.find(cacheKey);
    if (cacheIt != iconCache.end()) {
        return cacheIt->second;
    }

    QPixmap pixmap(iconPath);
    if (pixmap.isNull()) {
        return QPixmap();
    }

    iconCache.emplace(cacheKey, pixmap);
    return pixmap;
}

int NodeGraphIconRegistry::iconWidthCacheKey(qreal targetPixelWidth) {
    return std::max(1, static_cast<int>(std::lround(targetPixelWidth)));
}
