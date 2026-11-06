#include "UiPalette.hpp"

namespace ui {

QColor UiPalette::selectionHighlight() {
    return QColor(QString::fromLatin1(SelectionHighlightHex));
}

glm::vec3 UiPalette::selectionHighlightLinear() {
    // sRGB to linear space for outline pass
    return glm::vec3(
        std::pow(SelectionHighlightSrgb.r, 2.2f),
        std::pow(SelectionHighlightSrgb.g, 2.2f),
        std::pow(SelectionHighlightSrgb.b, 2.2f)
    );
}

glm::vec4 UiPalette::selectionHighlightSrgb() {
    return SelectionHighlightSrgb;
}

QColor UiPalette::nodeSelectionHighlight() {
    return QColor(72, 183, 255);
}

QColor UiPalette::probeMarkerRed() {
    return QColor(QString::fromLatin1(ProbeMarkerRedHex));
}

glm::vec4 UiPalette::probeMarkerRedSrgb() {
    return ProbeMarkerRedSrgb;
}

QColor UiPalette::windowBackground() { return QColor(QString::fromLatin1(WindowBackgroundHex)); }
QColor UiPalette::panelBackground() { return QColor(QString::fromLatin1(PanelBackgroundHex)); }
QColor UiPalette::cardBackground() { return QColor(QString::fromLatin1(CardBackgroundHex)); }
QColor UiPalette::inputBackground() { return QColor(QString::fromLatin1(InputBackgroundHex)); }
QColor UiPalette::checkboxBackground() { return QColor(QString::fromLatin1(CheckboxBackgroundHex)); }
QColor UiPalette::checkboxBorder() { return QColor(QString::fromLatin1(CheckboxBorderHex)); }

QColor UiPalette::border() { return QColor(QString::fromLatin1(BorderHex)); }
QColor UiPalette::subtleBorder() { return QColor(QString::fromLatin1(SubtleBorderHex)); }

QColor UiPalette::text() { return QColor(QString::fromLatin1(TextHex)); }
QColor UiPalette::headingText() { return QColor(QString::fromLatin1(HeadingTextHex)); }
QColor UiPalette::secondaryText() { return QColor(QString::fromLatin1(SecondaryTextHex)); }
QColor UiPalette::mutedText() { return QColor(QString::fromLatin1(MutedTextHex)); }
QColor UiPalette::onAccentText() { return QColor(QString::fromLatin1(OnAccentTextHex)); }
QColor UiPalette::activeTabText() { return QColor(QString::fromLatin1(ActiveTabTextHex)); }
QColor UiPalette::navigationText() { return QColor(QString::fromLatin1(NavigationTextHex)); }

QColor UiPalette::accent() { return QColor(QString::fromLatin1(AccentHex)); }
QColor UiPalette::interactiveAccent() { return QColor(QString::fromLatin1(InteractiveAccentHex)); }
QColor UiPalette::interactiveHover() { return QColor(QString::fromLatin1(InteractiveHoverHex)); }
QColor UiPalette::hover() { return QColor(QString::fromLatin1(HoverHex)); }
QColor UiPalette::selected() { return QColor(QString::fromLatin1(SelectedHex)); }

QColor UiPalette::toolNormal() { return QColor(QString::fromLatin1(ToolNormalHex)); }
QColor UiPalette::toolHover() { return QColor(QString::fromLatin1(ToolHoverHex)); }
QColor UiPalette::toolPressed() { return QColor(QString::fromLatin1(ToolPressedHex)); }
QColor UiPalette::toolSelectedPressed() { return QColor(QString::fromLatin1(ToolSelectedPressedHex)); }
QColor UiPalette::toolBorder() { return QColor(QString::fromLatin1(ToolBorderHex)); }
QColor UiPalette::sliderTrack() { return QColor(QString::fromLatin1(SliderTrackHex)); }
QColor UiPalette::sliderHandle() { return QColor(QString::fromLatin1(SliderHandleHex)); }
QColor UiPalette::sliderHandleBorder() { return QColor(QString::fromLatin1(SliderHandleBorderHex)); }
QColor UiPalette::sourceButton() { return QColor(QString::fromLatin1(SourceButtonHex)); }
QColor UiPalette::sourceButtonHover() { return QColor(QString::fromLatin1(SourceButtonHoverHex)); }
QColor UiPalette::sourceButtonBorder() { return QColor(QString::fromLatin1(SourceButtonBorderHex)); }

QColor UiPalette::menuBarBackground() { return QColor(QString::fromLatin1(MenuBarBackgroundHex)); }
QColor UiPalette::menuText() { return QColor(QString::fromLatin1(MenuTextHex)); }
QColor UiPalette::menuDisabledText() { return QColor(QString::fromLatin1(MenuDisabledTextHex)); }
QColor UiPalette::menuSeparator() { return QColor(QString::fromLatin1(MenuSeparatorHex)); }

QColor UiPalette::timelineBackground() { return QColor(QString::fromLatin1(TimelineBackgroundHex)); }
QColor UiPalette::timelineTrack() { return QColor(QString::fromLatin1(TimelineTrackHex)); }
QColor UiPalette::timelineRecorded() { return QColor(QString::fromLatin1(TimelineRecordedHex)); }
QColor UiPalette::timelineTick() { return QColor(QString::fromLatin1(TimelineTickHex)); }
QColor UiPalette::timelineText() { return QColor(QString::fromLatin1(TimelineTextHex)); }
QColor UiPalette::timelinePlayhead() { return QColor(QString::fromLatin1(TimelinePlayheadHex)); }
QColor UiPalette::timelineFieldBackground() { return QColor(QString::fromLatin1(TimelineFieldBackgroundHex)); }

QColor UiPalette::welcomeBackdrop() { return QColor::fromRgbF(0.04f, 0.04f, 0.05f, 0.80f); }
QColor UiPalette::welcomeCardBackground() { return QColor(QString::fromLatin1(WelcomeCardBackgroundHex)); }
QColor UiPalette::welcomeTitleBar() { return QColor(QString::fromLatin1(WelcomeTitleBarHex)); }
QColor UiPalette::welcomeTitleBarText() { return QColor(QString::fromLatin1(WelcomeTitleBarTextHex)); }
QColor UiPalette::welcomeTitleBarBorder() { return QColor(QString::fromLatin1(WelcomeTitleBarBorderHex)); }
QColor UiPalette::welcomeOptionBackground() { return QColor(QString::fromLatin1(WelcomeOptionBackgroundHex)); }
QColor UiPalette::welcomeOptionHover() { return QColor(QString::fromLatin1(WelcomeOptionHoverHex)); }
QColor UiPalette::welcomeCardBorder() { return QColor(QString::fromLatin1(WelcomeCardBorderHex)); }
QColor UiPalette::welcomeText() { return QColor(QString::fromLatin1(WelcomeTextHex)); }
QColor UiPalette::welcomeSubtext() { return QColor(QString::fromLatin1(WelcomeSubtextHex)); }

} // namespace ui
