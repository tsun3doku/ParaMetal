#include "BoundaryFaceUiModel.hpp"

#include <algorithm>

void BoundaryFaceUiModel::setPosition(double x, double y) {
    if (posX == x && posY == y) return;
    posX = x;
    posY = y;
    emit stateChanged();
}

void BoundaryFaceUiModel::applyEditingNode(int nodeId) {
    nodeId = std::max(0, nodeId);
    if (nodeIdValue == nodeId) return;
    nodeIdValue = nodeId;
    faceValue = -1;
    emit stateChanged();
}

void BoundaryFaceUiModel::applySelectedFace(int nodeId, int face) {
    if (nodeIdValue != nodeId) return;
    face = (face >= 0 && face < 6) ? face : -1;
    if (faceValue == face) return;
    faceValue = face;
    emit stateChanged();
}
