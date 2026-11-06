#pragma once

#include <QColor>
#include <QString>
#include <cmath>
#include <glm/glm.hpp>

namespace ui {

class UiPalette {
public:
    UiPalette() = delete;

    // Viewport selection highlight 
    inline static constexpr const char* SelectionHighlightHex = "#f6a53c";
    inline static constexpr glm::vec4 SelectionHighlightSrgb{0.964705f, 0.647058f, 0.235294f, 1.0f};

    // NodeGraph node selection highlight (Theme Blue #48B7FF / rgb(72, 183, 255))
    inline static constexpr const char* NodeSelectionHighlightHex = "#48b7ff";
    inline static constexpr glm::vec4 NodeSelectionHighlightSrgb{72.0f / 255.0f, 183.0f / 255.0f, 1.0f, 1.0f};

    // Probe marker
    inline static constexpr const char* ProbeMarkerRedHex = "#ea0b0b";
    inline static constexpr glm::vec4 ProbeMarkerRedSrgb{0.92f, 0.04f, 0.04f, 1.0f};

    // UI accents and BGs
    inline static constexpr const char* WindowBackgroundHex = "#1e1e1e";
    inline static constexpr const char* PanelBackgroundHex = "#2e2e2e";
    inline static constexpr const char* CardBackgroundHex = "#3d3d3d";
    inline static constexpr const char* InputBackgroundHex = "#545454";
    inline static constexpr const char* CheckboxBackgroundHex = "#545454";
    inline static constexpr const char* CheckboxBorderHex = "#444444";

    // Borders
    inline static constexpr const char* BorderHex = "#444444";
    inline static constexpr const char* SubtleBorderHex = "#444444";

    // Text
    inline static constexpr const char* TextHex = "#f0f0f0";
    inline static constexpr const char* HeadingTextHex = "#ffffff";
    inline static constexpr const char* SecondaryTextHex = "#b0b0b0";
    inline static constexpr const char* MutedTextHex = "#808080";
    inline static constexpr const char* OnAccentTextHex = "#ffffff";
    inline static constexpr const char* ActiveTabTextHex = "#ffffff";
    inline static constexpr const char* NavigationTextHex = "#a0a0a0";

    // Interactive
    inline static constexpr const char* AccentHex = "#3578ff";
    inline static constexpr const char* InteractiveAccentHex = "#3578ff";
    inline static constexpr const char* InteractiveHoverHex = "#4a88ff";
    inline static constexpr const char* HoverHex = "#4a4a4a";
    inline static constexpr const char* SelectedHex = "#545454";

    // Tools
    inline static constexpr const char* ToolNormalHex = "#343434";
    inline static constexpr const char* ToolHoverHex = "#464646";
    inline static constexpr const char* ToolPressedHex = "#282828";
    inline static constexpr const char* ToolSelectedPressedHex = "#2868e0";
    inline static constexpr const char* ToolBorderHex = "#444444";
    inline static constexpr const char* SliderTrackHex = "#808080";
    inline static constexpr const char* SliderHandleHex = "#e0e0e0";
    inline static constexpr const char* SliderHandleBorderHex = "#303030";
    inline static constexpr const char* SourceButtonHex = "#4a4a4a";
    inline static constexpr const char* SourceButtonHoverHex = "#585858";
    inline static constexpr const char* SourceButtonBorderHex = "#444444";

    // Menu
    inline static constexpr const char* MenuBarBackgroundHex = "#1e1e1e";
    inline static constexpr const char* MenuTextHex = "#f0f0f0";
    inline static constexpr const char* MenuDisabledTextHex = "#707070";
    inline static constexpr const char* MenuSeparatorHex = "#383838";

    // Timeline
    inline static constexpr const char* TimelineBackgroundHex = "#343434";
    inline static constexpr const char* TimelineTrackHex = "#808080";
    inline static constexpr const char* TimelineRecordedHex = "#416d9c";
    inline static constexpr const char* TimelineTickHex = "#909090";
    inline static constexpr const char* TimelineTextHex = "#c0c0c0";
    inline static constexpr const char* TimelinePlayheadHex = "#2e7eff";
    inline static constexpr const char* TimelineFieldBackgroundHex = "#404040";

    // Welcome screen
    inline static constexpr const char* WelcomeCardBackgroundHex = "#2e2e2e";
    inline static constexpr const char* WelcomeTitleBarHex = "#d0d0d0";
    inline static constexpr const char* WelcomeTitleBarTextHex = "#202020";
    inline static constexpr const char* WelcomeTitleBarBorderHex = "#a0a0a0";
    inline static constexpr const char* WelcomeOptionBackgroundHex = "#1e1e1e";
    inline static constexpr const char* WelcomeOptionHoverHex = "#383838";
    inline static constexpr const char* WelcomeCardBorderHex = "#444444";
    inline static constexpr const char* WelcomeTextHex = "#ffffff";
    inline static constexpr const char* WelcomeSubtextHex = "#b0b0b0";

    static QColor selectionHighlight();
    static glm::vec3 selectionHighlightLinear();
    static glm::vec4 selectionHighlightSrgb();

    static QColor nodeSelectionHighlight();

    static QColor probeMarkerRed();
    static glm::vec4 probeMarkerRedSrgb();

    static QColor windowBackground();
    static QColor panelBackground();
    static QColor cardBackground();
    static QColor inputBackground();
    static QColor checkboxBackground();
    static QColor checkboxBorder();

    static QColor border();
    static QColor subtleBorder();

    static QColor text();
    static QColor headingText();
    static QColor secondaryText();
    static QColor mutedText();
    static QColor onAccentText();
    static QColor activeTabText();
    static QColor navigationText();

    static QColor accent();
    static QColor interactiveAccent();
    static QColor interactiveHover();
    static QColor hover();
    static QColor selected();

    static QColor toolNormal();
    static QColor toolHover();
    static QColor toolPressed();
    static QColor toolSelectedPressed();
    static QColor toolBorder();
    static QColor sliderTrack();
    static QColor sliderHandle();
    static QColor sliderHandleBorder();
    static QColor sourceButton();
    static QColor sourceButtonHover();
    static QColor sourceButtonBorder();

    static QColor menuBarBackground();
    static QColor menuText();
    static QColor menuDisabledText();
    static QColor menuSeparator();

    static QColor timelineBackground();
    static QColor timelineTrack();
    static QColor timelineRecorded();
    static QColor timelineTick();
    static QColor timelineText();
    static QColor timelinePlayhead();
    static QColor timelineFieldBackground();

    static QColor welcomeBackdrop();
    static QColor welcomeCardBackground();
    static QColor welcomeTitleBar();
    static QColor welcomeTitleBarText();
    static QColor welcomeTitleBarBorder();
    static QColor welcomeOptionBackground();
    static QColor welcomeOptionHover();
    static QColor welcomeCardBorder();
    static QColor welcomeText();
    static QColor welcomeSubtext();
};

} // namespace ui
