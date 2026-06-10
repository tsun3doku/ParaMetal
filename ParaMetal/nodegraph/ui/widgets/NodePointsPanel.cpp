#include "NodePointsPanel.hpp"

#include "nodegraph/NodeGraphBridge.hpp"
#include "nodegraph/NodeGraphEditor.hpp"
#include "nodegraph/NodePointsParams.hpp"
#include "NodeGraphWidgetStyle.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QVBoxLayout>

NodePointsPanel::NodePointsPanel(QWidget* parent)
    : NodePanelBase(parent) {
    QVBoxLayout* layout = static_cast<QVBoxLayout*>(this->layout());

    // Point Count
    QHBoxLayout* countRow = new QHBoxLayout();
    countRow->addWidget(new QLabel("Point Count:", this));
    countEdit = nodegraphwidgets::createNumericEdit(this, 1.0, 10000000.0, 0);
    countEdit->setText("1000");
    countRow->addWidget(countEdit, 1);
    layout->addLayout(countRow);

    // Dimensions
    QHBoxLayout* dimRow = new QHBoxLayout();
    dimRow->addWidget(new QLabel("Dimensions:", this));
    const char* axisLabels[3] = {"X", "Y", "Z"};
    for (std::size_t i = 0; i < dimEdits.size(); ++i) {
        dimRow->addWidget(new QLabel(axisLabels[i], this));
        dimEdits[i] = nodegraphwidgets::createNumericEdit(this, 0.0, 1000.0, 6);
        dimEdits[i]->setText("1.00");
        dimRow->addWidget(dimEdits[i], 1);
    }
    layout->addLayout(dimRow);

    layout->addStretch();

    connect(countEdit, &QLineEdit::editingFinished, this, [this]() { applySettings(); });
    for (QLineEdit* edit : dimEdits) {
        connect(edit, &QLineEdit::editingFinished, this, [this]() { applySettings(); });
    }
}

void NodePointsPanel::applySettings() {
    if (isSyncing() || !canEdit()) {
        return;
    }

    PointsNodeParams params{};
    params.pointCount = static_cast<uint32_t>(countEdit->text().toDouble());
    params.dimX = static_cast<float>(dimEdits[0]->text().toDouble());
    params.dimY = static_cast<float>(dimEdits[1]->text().toDouble());
    params.dimZ = static_cast<float>(dimEdits[2]->text().toDouble());

    if (params.pointCount == 0) {
        setStatus("Point count must be at least 1.");
        return;
    }

    NodeGraphEditor editor(bridge());
    if (!writePointsNodeParams(editor, currentNodeId(), params)) {
        setStatus("Failed to update point cloud settings.");
        return;
    }

    setStatus(QString("Point cloud: %1 points, dim [%2, %3, %4]")
                  .arg(params.pointCount)
                  .arg(QString::number(params.dimX, 'f', 4))
                  .arg(QString::number(params.dimY, 'f', 4))
                  .arg(QString::number(params.dimZ, 'f', 4)));
}

void NodePointsPanel::refreshFromNode() {
    if (!canEdit()) {
        return;
    }

    NodeGraphNode node{};
    if (!loadCurrentNode(node)) {
        return;
    }

    const PointsNodeParams params = readPointsNodeParams(node);
    setSyncing(true);
    countEdit->setText(QString::number(params.pointCount));
    dimEdits[0]->setText(QString::number(params.dimX, 'f', 4));
    dimEdits[1]->setText(QString::number(params.dimY, 'f', 4));
    dimEdits[2]->setText(QString::number(params.dimZ, 'f', 4));
    setSyncing(false);
}
