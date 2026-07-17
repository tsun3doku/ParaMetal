#pragma once

#include "NodeGraphTypes.hpp"

#include <QPixmap>

class NodeGraphIconRegistry {
public:
    NodeGraphIconRegistry() = delete;

    static QPixmap nodePixmapForType(const NodeTypeId& typeId, qreal targetPixelWidth);
    static QPixmap screenSpaceNodePixmapForType(const NodeTypeId& typeId, qreal logicalWidth);

private:
    static const char* iconFolderForType(const NodeTypeId& typeId);
};
