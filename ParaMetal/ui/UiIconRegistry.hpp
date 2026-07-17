#pragma once

#include <QPixmap>
#include <QString>

#include <filesystem>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ui {

class IconRegistry {
public:
    IconRegistry() = delete;

    static QPixmap pixmapForFolder(const QString& folder, qreal targetPixelWidth);
    static QPixmap screenSpacePixmapForFolder(const QString& folder,
                                              qreal logicalWidth,
                                              int sourcePixelWidth = 128);

private:
    static std::vector<std::filesystem::path> iconFolderCandidates(const std::string& folder);
    static std::vector<std::pair<int, std::filesystem::path>> iconVariantsForFolder(const QString& folder);
    static QString nearestIconPathForFolder(const QString& folder, qreal targetPixelWidth);
    static QString exactIconPathForFolder(const QString& folder, int pixelWidth);
    static QPixmap pixmapForPath(const QString& iconPath);
    static int iconWidthCacheKey(qreal targetPixelWidth);
    static int parseIconWidth(const std::filesystem::path& path);

    inline static std::unordered_map<std::string, QPixmap> pixmapCache{};
};

} // namespace ui
