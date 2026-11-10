#include "VelocityPaletteUiModel.hpp"

#include <algorithm>
#include <cmath>

void VelocityPaletteUiModel::setPaletteId(int value) {
    value = std::clamp(value, 0, 3);
    if (state.paletteId == value) return;
    state.paletteId = value;
    emit stateChanged();
    emit paletteRequested(value);
}

void VelocityPaletteUiModel::setUnits(int value) {
    value = std::clamp(value, 0, 2);
    if (state.units == value) return;
    state.units = value;
    emit stateChanged();
}

void VelocityPaletteUiModel::setRange(double minimum, double maximum) {
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || maximum <= minimum) return;
    state.minimum = minimum;
    state.maximum = maximum;
    emit stateChanged();
    emit rangeRequested(static_cast<float>(minimum), static_cast<float>(maximum));
}

void VelocityPaletteUiModel::setPosition(double newX, double newY) {
    if (state.x == newX && state.y == newY) return;
    state.x = newX;
    state.y = newY;
    emit stateChanged();
}

void VelocityPaletteUiModel::applyVisibility(bool value) {
    if (state.visible == value) return;
    state.visible = value;
    emit stateChanged();
}
