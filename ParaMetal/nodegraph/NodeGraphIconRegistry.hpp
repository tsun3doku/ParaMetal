#pragma once

#include "NodeGraphTypes.hpp"

#include <filesystem>
#include <unordered_map>
#include <utility>
#include <vector>

#include <QPixmap>
#include <QString>

class NodeGraphIconRegistry {
public:
    NodeGraphIconRegistry() = delete;

    static QPixmap nodePixmapForType(const NodeTypeId& typeId, qreal targetPixelWidth);
    static QPixmap screenSpaceNodePixmapForType(const NodeTypeId& typeId, qreal logicalWidth);
    static QPixmap screenSpacePixmapForFolder(const QString& folder, qreal logicalWidth);

private:
    static constexpr int screenSpaceIconSourceWidth = 128;

    static const char* iconFolderForType(const NodeTypeId& typeId);
    static std::vector<std::filesystem::path> iconFolderCandidates(const std::string& folder);
    static std::vector<std::pair<int, std::filesystem::path>> iconVariantsForFolder(const QString& folder);
    static QString nearestIconPathForFolder(const QString& folder, qreal targetPixelWidth);
    static QString exactIconPathForFolder(const QString& folder, int pixelWidth);
    static QPixmap pixmapForPath(const QString& iconPath);
    static int iconWidthCacheKey(qreal targetPixelWidth);
    static int parseIconWidth(const std::filesystem::path& path);
    inline static std::unordered_map<std::string, QPixmap> pixmapCache{};
};
