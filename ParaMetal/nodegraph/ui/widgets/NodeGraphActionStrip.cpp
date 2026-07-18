#include "NodeGraphActionStrip.hpp"

#include "nodegraph/ui/widgets/NodeGraphWidgetStyle.hpp"
#include "ui/UiIconRegistry.hpp"
#include "ui/UiTypography.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QSize>
#include <QSizePolicy>
#include <QString>
#include <QStringList>
#include <QVBoxLayout>

QPixmap NodeGraphActionStrip::loadPreview(const QString& relativePath) {
    const QStringList candidates = {
        QDir(QCoreApplication::applicationDirPath()).filePath(relativePath),
        relativePath,
        QStringLiteral("ParaMetal/") + relativePath,
        QStringLiteral("../") + relativePath,
        QStringLiteral("../../") + relativePath,
        QStringLiteral("../ParaMetal/") + relativePath,
        QStringLiteral("../../ParaMetal/") + relativePath
    };

    for (const QString& candidate : candidates) {
        QPixmap preview(candidate);
        if (!preview.isNull()) {
            return preview;
        }
    }
    return {};
}

NodeGraphActionStrip::NodeGraphActionStrip(
    const QString& title,
    const QString& description,
    const QString& actionText,
    const QString& previewPath,
    QWidget* parent)
    : QWidget(parent) {

    setObjectName(QStringLiteral("NodeGraphActionStrip"));
    setAttribute(Qt::WA_StyledBackground, true);
    setFont(ui::UiTypography::font(ui::TextRole::Regular));
    setFixedHeight(nodegraphwidgets::actionStripHeight);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setStyleSheet(nodegraphwidgets::actionStripStyleSheet());

    QHBoxLayout* rootLayout = new QHBoxLayout(this);
    rootLayout->setContentsMargins(12, 8, 8, 8);
    rootLayout->setSpacing(10);

    previewLabel = new QLabel(this);
    previewLabel->setFixedSize(nodegraphwidgets::actionStripPreviewSize, nodegraphwidgets::actionStripPreviewSize);
    previewLabel->setAlignment(Qt::AlignCenter);
    previewLabel->setToolTip(QStringLiteral("Sample graph preview"));
    const QPixmap preview = loadPreview(previewPath);
    if (!preview.isNull()) {
        previewLabel->setPixmap(preview.scaled(
            nodegraphwidgets::actionStripPreviewSize,
            nodegraphwidgets::actionStripPreviewSize,
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation));
    }
    rootLayout->addWidget(previewLabel, 0, Qt::AlignTop);

    QVBoxLayout* detailsLayout = new QVBoxLayout();
    detailsLayout->setContentsMargins(0, 0, 0, 0);
    detailsLayout->setSpacing(5);

    QHBoxLayout* titleLayout = new QHBoxLayout();
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(2);

    titleLabel = new QLabel(title, this);
    titleLabel->setObjectName(QStringLiteral("ActionStripTitle"));
    nodegraphwidgets::styleTitleLabel(titleLabel);
    titleLabel->setWordWrap(true);
    titleLayout->addWidget(titleLabel, 1, Qt::AlignTop);

    dismissButton = new QPushButton(this);
    dismissButton->setObjectName(QStringLiteral("ActionStripDismiss"));
    dismissButton->setAccessibleName(QStringLiteral("Dismiss sample graph"));
    dismissButton->setCursor(Qt::PointingHandCursor);
    dismissButton->setFixedSize(nodegraphwidgets::actionStripDismissSize, nodegraphwidgets::actionStripDismissSize);
    const QPixmap dismissPixmap = ui::IconRegistry::screenSpacePixmapForFolder(
        QStringLiteral("Menu/x"),
        nodegraphwidgets::actionStripDismissIconSize);
    if (!dismissPixmap.isNull()) {
        dismissButton->setIcon(QIcon(dismissPixmap));
        dismissButton->setIconSize(QSize(
            nodegraphwidgets::actionStripDismissIconSize,
            nodegraphwidgets::actionStripDismissIconSize));
    }
    titleLayout->addWidget(dismissButton, 0, Qt::AlignTop);
    detailsLayout->addLayout(titleLayout);

    descriptionLabel = new QLabel(description, this);
    descriptionLabel->setObjectName(QStringLiteral("ActionStripDescription"));
    descriptionLabel->setFont(ui::UiTypography::font(ui::TextRole::Description));
    descriptionLabel->setWordWrap(true);
    detailsLayout->addWidget(descriptionLabel);
    detailsLayout->addStretch(1);

    actionButton = new QPushButton(actionText, this);
    actionButton->setObjectName(QStringLiteral("ActionStripButton"));
    actionButton->setCursor(Qt::PointingHandCursor);
    actionButton->setFixedWidth(56);
    detailsLayout->addWidget(actionButton, 0, Qt::AlignLeft);

    rootLayout->addLayout(detailsLayout, 1);

    connect(actionButton, &QPushButton::clicked, this, &NodeGraphActionStrip::triggered);
    connect(dismissButton, &QPushButton::clicked, this, &NodeGraphActionStrip::dismissed);
}
