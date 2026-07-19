#pragma once

#include <QFont>
#include <QString>

namespace ui {

enum class FontFamily {
    Interface,
    Monospace
};

enum class TextRole {
    Title,
    Regular,
    Description,
    Console,
    TimelineBubble,
    NodeTitle
};

struct TypographySpec {
    int pixelSize;
    QFont::Weight weight;
    qreal letterSpacing;
};

class UiTypography {
public:
    UiTypography() = delete;

    inline static constexpr TypographySpec Title{12, QFont::Normal, 0.75};
    inline static constexpr TypographySpec Regular{12, QFont::Light, 0.0};
    inline static constexpr TypographySpec Description{12, QFont::Light, 0.0};
    inline static constexpr TypographySpec Console{11, QFont::Light, 0.0};
    inline static constexpr TypographySpec TimelineBubble{Description.pixelSize, QFont::DemiBold, Description.letterSpacing};
    inline static constexpr TypographySpec NodeTitle{Regular.pixelSize, QFont::Medium, Regular.letterSpacing};

    static QString familyName(FontFamily family);
    static const TypographySpec& spec(TextRole role);
    static QFont font(TextRole role, FontFamily family = FontFamily::Interface);
};

}
