#include "NodeGraphActionStrip.hpp"

#include "nodegraph/NodeGraphIconRegistry.hpp"
#include "nodegraph/ui/widgets/NodeGraphWidgetStyle.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QSizePolicy>
#include <QString>
#include <QVBoxLayout>

NodeGraphActionStrip::NodeGraphActionStrip(
    const QString& title,
    const QString& description,
    const QString& actionText,
    const QString& iconFolder,
    QWidget* parent)
    : QWidget(parent) {

    setObjectName(QStringLiteral("NodeGraphActionStrip"));
    setAttribute(Qt::WA_StyledBackground, true);
    setFixedHeight(nodegraphwidgets::actionStripHeight);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setStyleSheet(nodegraphwidgets::actionStripStyleSheet());

    QHBoxLayout* rootLayout = new QHBoxLayout(this);
    rootLayout->setContentsMargins(12, 8, 8, 8);
    rootLayout->setSpacing(10);

    iconLabel = new QLabel(this);
    iconLabel->setFixedSize(nodegraphwidgets::actionStripIconSize, nodegraphwidgets::actionStripIconSize);
    iconLabel->setAlignment(Qt::AlignCenter);
    const QPixmap icon = NodeGraphIconRegistry::screenSpacePixmapForFolder(
        iconFolder,
        nodegraphwidgets::actionStripIconSize);
    if (!icon.isNull()) {
        iconLabel->setPixmap(icon);
    }
    rootLayout->addWidget(iconLabel, 0, Qt::AlignTop);

    QVBoxLayout* textLayout = new QVBoxLayout();
    textLayout->setContentsMargins(0, 0, 0, 0);
    textLayout->setSpacing(3);

    titleLabel = new QLabel(title, this);
    titleLabel->setObjectName(QStringLiteral("ActionStripTitle"));
    titleLabel->setWordWrap(false);
    textLayout->addWidget(titleLabel);

    descriptionLabel = new QLabel(description, this);
    descriptionLabel->setObjectName(QStringLiteral("ActionStripDescription"));
    descriptionLabel->setWordWrap(true);
    textLayout->addWidget(descriptionLabel);

    rootLayout->addLayout(textLayout, 1);

    actionButton = new QPushButton(actionText, this);
    actionButton->setObjectName(QStringLiteral("ActionStripButton"));
    actionButton->setCursor(Qt::PointingHandCursor);
    actionButton->setFixedWidth(56);
    rootLayout->addWidget(actionButton, 0, Qt::AlignVCenter);

    dismissButton = new QPushButton(QStringLiteral("x"), this);
    dismissButton->setObjectName(QStringLiteral("ActionStripDismiss"));
    dismissButton->setCursor(Qt::PointingHandCursor);
    dismissButton->setFixedSize(nodegraphwidgets::actionStripDismissSize, nodegraphwidgets::actionStripDismissSize);
    rootLayout->addWidget(dismissButton, 0, Qt::AlignVCenter);

    connect(actionButton, &QPushButton::clicked, this, &NodeGraphActionStrip::triggered);
    connect(dismissButton, &QPushButton::clicked, this, &NodeGraphActionStrip::dismissed);
}
