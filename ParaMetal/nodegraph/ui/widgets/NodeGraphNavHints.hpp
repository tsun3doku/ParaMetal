#pragma once

#include <QWidget>

class NodeGraphNavHints : public QWidget {
public:
    explicit NodeGraphNavHints(QWidget* parent = nullptr);

private:
    struct NavHint {
        const char* iconFolder;
        const char* text;
    };

    static const NavHint navHints[];
    static void applyClickThrough(QWidget* widget);
    static QWidget* createNavHintItem(const NavHint& hint, QWidget* parent);
};
