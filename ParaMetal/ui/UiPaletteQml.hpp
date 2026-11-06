#pragma once

#include <QObject>
#include <QColor>

namespace ui {

class UiPaletteQml final : public QObject {
    Q_OBJECT
    Q_PROPERTY(QColor selectionHighlight READ selectionHighlight CONSTANT)
    Q_PROPERTY(QColor nodeSelectionHighlight READ nodeSelectionHighlight CONSTANT)
    Q_PROPERTY(QColor probeMarkerRed READ probeMarkerRed CONSTANT)

    Q_PROPERTY(QColor windowBackground READ windowBackground CONSTANT)
    Q_PROPERTY(QColor panelBackground READ panelBackground CONSTANT)
    Q_PROPERTY(QColor cardBackground READ cardBackground CONSTANT)
    Q_PROPERTY(QColor inputBackground READ inputBackground CONSTANT)
    Q_PROPERTY(QColor checkboxBackground READ checkboxBackground CONSTANT)
    Q_PROPERTY(QColor checkboxBorder READ checkboxBorder CONSTANT)

    Q_PROPERTY(QColor border READ border CONSTANT)
    Q_PROPERTY(QColor subtleBorder READ subtleBorder CONSTANT)

    Q_PROPERTY(QColor text READ text CONSTANT)
    Q_PROPERTY(QColor headingText READ headingText CONSTANT)
    Q_PROPERTY(QColor secondaryText READ secondaryText CONSTANT)
    Q_PROPERTY(QColor mutedText READ mutedText CONSTANT)
    Q_PROPERTY(QColor onAccentText READ onAccentText CONSTANT)
    Q_PROPERTY(QColor activeTabText READ activeTabText CONSTANT)
    Q_PROPERTY(QColor navigationText READ navigationText CONSTANT)

    Q_PROPERTY(QColor accent READ accent CONSTANT)
    Q_PROPERTY(QColor interactiveAccent READ interactiveAccent CONSTANT)
    Q_PROPERTY(QColor interactiveHover READ interactiveHover CONSTANT)
    Q_PROPERTY(QColor hover READ hover CONSTANT)
    Q_PROPERTY(QColor selected READ selected CONSTANT)

    Q_PROPERTY(QColor toolNormal READ toolNormal CONSTANT)
    Q_PROPERTY(QColor toolHover READ toolHover CONSTANT)
    Q_PROPERTY(QColor toolPressed READ toolPressed CONSTANT)
    Q_PROPERTY(QColor toolSelectedPressed READ toolSelectedPressed CONSTANT)
    Q_PROPERTY(QColor toolBorder READ toolBorder CONSTANT)
    Q_PROPERTY(QColor sliderTrack READ sliderTrack CONSTANT)
    Q_PROPERTY(QColor sliderHandle READ sliderHandle CONSTANT)
    Q_PROPERTY(QColor sliderHandleBorder READ sliderHandleBorder CONSTANT)
    Q_PROPERTY(QColor sourceButton READ sourceButton CONSTANT)
    Q_PROPERTY(QColor sourceButtonHover READ sourceButtonHover CONSTANT)
    Q_PROPERTY(QColor sourceButtonBorder READ sourceButtonBorder CONSTANT)

    Q_PROPERTY(QColor menuBarBackground READ menuBarBackground CONSTANT)
    Q_PROPERTY(QColor menuText READ menuText CONSTANT)
    Q_PROPERTY(QColor menuDisabledText READ menuDisabledText CONSTANT)
    Q_PROPERTY(QColor menuSeparator READ menuSeparator CONSTANT)

    Q_PROPERTY(QColor timelineBackground READ timelineBackground CONSTANT)
    Q_PROPERTY(QColor timelineTrack READ timelineTrack CONSTANT)
    Q_PROPERTY(QColor timelineRecorded READ timelineRecorded CONSTANT)
    Q_PROPERTY(QColor timelineTick READ timelineTick CONSTANT)
    Q_PROPERTY(QColor timelineText READ timelineText CONSTANT)
    Q_PROPERTY(QColor timelinePlayhead READ timelinePlayhead CONSTANT)
    Q_PROPERTY(QColor timelineFieldBackground READ timelineFieldBackground CONSTANT)

    Q_PROPERTY(QColor welcomeBackdrop READ welcomeBackdrop CONSTANT)
    Q_PROPERTY(QColor welcomeCardBackground READ welcomeCardBackground CONSTANT)
    Q_PROPERTY(QColor welcomeTitleBar READ welcomeTitleBar CONSTANT)
    Q_PROPERTY(QColor welcomeTitleBarText READ welcomeTitleBarText CONSTANT)
    Q_PROPERTY(QColor welcomeTitleBarBorder READ welcomeTitleBarBorder CONSTANT)
    Q_PROPERTY(QColor welcomeOptionBackground READ welcomeOptionBackground CONSTANT)
    Q_PROPERTY(QColor welcomeOptionHover READ welcomeOptionHover CONSTANT)
    Q_PROPERTY(QColor welcomeCardBorder READ welcomeCardBorder CONSTANT)
    Q_PROPERTY(QColor welcomeText READ welcomeText CONSTANT)
    Q_PROPERTY(QColor welcomeSubtext READ welcomeSubtext CONSTANT)

public:
    explicit UiPaletteQml(QObject* parent = nullptr) : QObject(parent) {}

    QColor selectionHighlight() const;
    QColor nodeSelectionHighlight() const;
    QColor probeMarkerRed() const;

    QColor windowBackground() const;
    QColor panelBackground() const;
    QColor cardBackground() const;
    QColor inputBackground() const;
    QColor checkboxBackground() const;
    QColor checkboxBorder() const;

    QColor border() const;
    QColor subtleBorder() const;

    QColor text() const;
    QColor headingText() const;
    QColor secondaryText() const;
    QColor mutedText() const;
    QColor onAccentText() const;
    QColor activeTabText() const;
    QColor navigationText() const;

    QColor accent() const;
    QColor interactiveAccent() const;
    QColor interactiveHover() const;
    QColor hover() const;
    QColor selected() const;

    QColor toolNormal() const;
    QColor toolHover() const;
    QColor toolPressed() const;
    QColor toolSelectedPressed() const;
    QColor toolBorder() const;
    QColor sliderTrack() const;
    QColor sliderHandle() const;
    QColor sliderHandleBorder() const;
    QColor sourceButton() const;
    QColor sourceButtonHover() const;
    QColor sourceButtonBorder() const;

    QColor menuBarBackground() const;
    QColor menuText() const;
    QColor menuDisabledText() const;
    QColor menuSeparator() const;

    QColor timelineBackground() const;
    QColor timelineTrack() const;
    QColor timelineRecorded() const;
    QColor timelineTick() const;
    QColor timelineText() const;
    QColor timelinePlayhead() const;
    QColor timelineFieldBackground() const;

    QColor welcomeBackdrop() const;
    QColor welcomeCardBackground() const;
    QColor welcomeTitleBar() const;
    QColor welcomeTitleBarText() const;
    QColor welcomeTitleBarBorder() const;
    QColor welcomeOptionBackground() const;
    QColor welcomeOptionHover() const;
    QColor welcomeCardBorder() const;
    QColor welcomeText() const;
    QColor welcomeSubtext() const;
};

} // namespace ui
