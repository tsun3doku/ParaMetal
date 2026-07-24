#include "UiTypographyQml.hpp"

#include "UiTypography.hpp"

namespace ui {

QString UiTypographyQml::fontFamily() const {
    return UiTypography::familyName(FontFamily::Interface);
}

QString UiTypographyQml::monoFamily() const {
    return UiTypography::familyName(FontFamily::Monospace);
}

int UiTypographyQml::titleFontSize() const {
    return UiTypography::spec(TextRole::Title).pixelSize;
}

int UiTypographyQml::regularFontSize() const {
    return UiTypography::spec(TextRole::Regular).pixelSize;
}

int UiTypographyQml::regularFontWeight() const {
    return static_cast<int>(UiTypography::spec(TextRole::Regular).weight);
}

int UiTypographyQml::descriptionFontSize() const {
    return UiTypography::spec(TextRole::Description).pixelSize;
}

int UiTypographyQml::consoleFontSize() const {
    return UiTypography::spec(TextRole::Console).pixelSize;
}

int UiTypographyQml::nodeTitleFontSize() const {
    return UiTypography::spec(TextRole::NodeTitle).pixelSize;
}

QFont UiTypographyQml::regularFont() const {
    return UiTypography::font(TextRole::Regular);
}

QFont UiTypographyQml::descriptionFont() const {
    return UiTypography::font(TextRole::Description);
}

QFont UiTypographyQml::nodeTitleFont() const {
    return UiTypography::font(TextRole::NodeTitle);
}


}
