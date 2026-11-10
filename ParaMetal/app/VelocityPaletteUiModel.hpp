#pragma once

#include "UiRuntimeTypes.hpp"

#include <QObject>

class VelocityPaletteUiModel final : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool visible READ visible NOTIFY stateChanged)
    Q_PROPERTY(int paletteId READ paletteId NOTIFY stateChanged)
    Q_PROPERTY(int units READ units NOTIFY stateChanged)
    Q_PROPERTY(double minimum READ minimum NOTIFY stateChanged)
    Q_PROPERTY(double maximum READ maximum NOTIFY stateChanged)
    Q_PROPERTY(double x READ x NOTIFY stateChanged)
    Q_PROPERTY(double y READ y NOTIFY stateChanged)

public:
    explicit VelocityPaletteUiModel(QObject* parent = nullptr) : QObject(parent) {}

    bool visible() const { return state.visible; }
    int paletteId() const { return state.paletteId; }
    int units() const { return state.units; }
    double minimum() const { return state.minimum; }
    double maximum() const { return state.maximum; }
    double x() const { return state.x; }
    double y() const { return state.y; }

    Q_INVOKABLE void setPaletteId(int value);
    Q_INVOKABLE void setUnits(int value);
    Q_INVOKABLE void setRange(double minimum, double maximum);
    Q_INVOKABLE void setPosition(double x, double y);

public slots:
    void applyVisibility(bool visible);

signals:
    void stateChanged();
    void paletteRequested(int palette);
    void rangeRequested(float minimum, float maximum);

private:
    VelocityPaletteUiState state{};
};
