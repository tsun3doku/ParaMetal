#include "NodeGraphNavHints.hpp"

#include "nodegraph/NodeGraphIconRegistry.hpp"
#include "nodegraph/ui/widgets/NodeGraphWidgetStyle.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QString>

const NodeGraphNavHints::NavHint NodeGraphNavHints::navHints[] = {
    { "NodeGraph_nav/mouse_middle", "Pan" },
    { "NodeGraph_nav/mouse_right",  "Menu" },
    { "NodeGraph_nav/node_right",   "Display" },
    { "NodeGraph_nav/node_left",    "Freeze" },
};

void NodeGraphNavHints::applyClickThrough(QWidget* widget) {
    if (widget) {
        widget->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    }
}

QWidget* NodeGraphNavHints::createNavHintItem(const NavHint& hint, QWidget* parent) {
    QWidget* item = new QWidget(parent);
    applyClickThrough(item);

    QLabel* iconLabel = new QLabel(item);
    iconLabel->setFixedSize(nodegraphwidgets::navHintIconSize, nodegraphwidgets::navHintIconSize);
    iconLabel->setAlignment(Qt::AlignCenter);
    applyClickThrough(iconLabel);

    const QPixmap icon = NodeGraphIconRegistry::screenSpacePixmapForFolder(
        QString::fromUtf8(hint.iconFolder),
        nodegraphwidgets::navHintIconSize);
    if (!icon.isNull()) {
        iconLabel->setPixmap(icon);
    }

    QLabel* textLabel = new QLabel(QString::fromUtf8(hint.text), item);
    textLabel->setStyleSheet(
        QStringLiteral("color: %1; background: transparent;")
            .arg(nodegraphwidgets::colorNavHintText.name()));
    applyClickThrough(textLabel);

    QHBoxLayout* layout = new QHBoxLayout(item);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(nodegraphwidgets::navHintIconTextGap);
    layout->addWidget(iconLabel);
    layout->addWidget(textLabel);

    return item;
}

NodeGraphNavHints::NodeGraphNavHints(QWidget* parent)
    : QWidget(parent) {
    setObjectName(QStringLiteral("NodeGraphNavHints"));
    setAttribute(Qt::WA_StyledBackground, true);
    applyClickThrough(this);
    setStyleSheet(QStringLiteral("background: transparent; border: none;"));

    QHBoxLayout* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(nodegraphwidgets::navHintItemSpacing);

    for (const NavHint& hint : navHints) {
        layout->addWidget(createNavHintItem(hint, this));
    }
}
