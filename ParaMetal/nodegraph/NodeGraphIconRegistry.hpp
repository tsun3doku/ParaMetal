#pragma once

#include "NodeGraphTypes.hpp"

#include <cstddef>
#include <filesystem>
#include <unordered_map>
#include <vector>

#include <QPixmap>
#include <QString>

struct IconCacheKey {
    NodeTypeId typeId{};
    int iconWidthKey = 0;

    bool operator==(const IconCacheKey& other) const noexcept;
};

struct IconCacheKeyHash {
    std::size_t operator()(const IconCacheKey& key) const noexcept;
};

class NodeGraphIconRegistry {
public:
    NodeGraphIconRegistry() = delete;

    static QPixmap iconForType(const NodeTypeId& typeId, qreal targetPixelWidth);
    static QString iconPathForFolder(const QString& folder, qreal targetPixelWidth);

private:
    static const char* iconFolderForType(const NodeTypeId& typeId);
    static std::vector<std::filesystem::path> iconFolderCandidates(const std::string& folder);
    static QString iconPathForType(const NodeTypeId& typeId, qreal targetPixelWidth);
    static int iconWidthCacheKey(qreal targetPixelWidth);
    static int parseIconWidth(const std::filesystem::path& path);
    inline static std::unordered_map<IconCacheKey, QPixmap, IconCacheKeyHash> iconCache{};
};
