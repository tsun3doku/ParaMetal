#pragma once

#include "NodePanelBase.hpp"

#include <array>

class QLineEdit;
class QLabel;

class NodePointsPanel final : public NodePanelBase {
public:
    explicit NodePointsPanel(QWidget* parent = nullptr);

protected:
    void refreshFromNode() override;

private:
    void applySettings();

    QLineEdit* countEdit = nullptr;
    std::array<QLineEdit*, 3> dimEdits{};
};
