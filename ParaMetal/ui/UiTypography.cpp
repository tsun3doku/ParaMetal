#include "UiTypography.hpp"

namespace ui {

QString UiTypography::familyName(FontFamily family) {
    switch (family) {
    case FontFamily::Monospace:
        return QStringLiteral("Azeret Mono");
    case FontFamily::Interface:
    default:
        return QStringLiteral("Outfit");
    }
}

const TypographySpec& UiTypography::spec(TextRole role) {
    switch (role) {
    case TextRole::Title:
        return Title;
    case TextRole::Description:
        return Description;
    case TextRole::Console:
        return Console;
    case TextRole::TimelineBubble:
        return TimelineBubble;
    case TextRole::NodeTitle:
        return NodeTitle;
    case TextRole::Regular:
    default:
        return Regular;
    }
}

QFont UiTypography::font(TextRole role, FontFamily family) {
    const TypographySpec& typography = spec(role);
    QFont result(familyName(family));
    result.setPixelSize(typography.pixelSize);
    result.setWeight(typography.weight);
    result.setLetterSpacing(QFont::AbsoluteSpacing, typography.letterSpacing);
    result.setHintingPreference(QFont::PreferVerticalHinting);
    return result;
}

}
