#pragma once

#include <QObject>

class BoundaryFaceUiModel final : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool visible READ visible NOTIFY stateChanged)
    Q_PROPERTY(int nodeId READ nodeId NOTIFY stateChanged)
    Q_PROPERTY(int face READ face NOTIFY stateChanged)
    Q_PROPERTY(double x READ x NOTIFY stateChanged)
    Q_PROPERTY(double y READ y NOTIFY stateChanged)

public:
    explicit BoundaryFaceUiModel(QObject* parent = nullptr) : QObject(parent) {}

    bool visible() const { return nodeIdValue > 0 && faceValue >= 0; }
    int nodeId() const { return nodeIdValue; }
    int face() const { return faceValue; }
    double x() const { return posX; }
    double y() const { return posY; }

    Q_INVOKABLE void setPosition(double x, double y);

public slots:
    void applyEditingNode(int nodeId);
    void applySelectedFace(int nodeId, int face);

signals:
    void stateChanged();

private:
    int nodeIdValue = 0;
    int faceValue = -1;
    double posX = -1.0;
    double posY = -1.0;
};
