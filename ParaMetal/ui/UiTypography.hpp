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

    inline static constexpr TypographySpec Title{12, QFont::Normal, 0.1};
    inline static constexpr TypographySpec Regular{13, QFont::Normal, 0.0};
    inline static constexpr TypographySpec Description{13, QFont::Light, 0.0};
    inline static constexpr TypographySpec Console{11, QFont::Light, 0.0};
    inline static constexpr TypographySpec NodeTitle{Regular.pixelSize, QFont::Light, Regular.letterSpacing};

    static QString familyName(FontFamily family);
    static const TypographySpec& spec(TextRole role);
    static QFont font(TextRole role, FontFamily family = FontFamily::Interface);
};

}
