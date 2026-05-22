#pragma once

#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextDocument>

class PySyntaxHighlighter : public QSyntaxHighlighter {
    Q_OBJECT
public:
    explicit PySyntaxHighlighter(QTextDocument* parent);

protected:
    void highlightBlock(const QString& text) override;

private:
    struct HighlightingRule {
        QRegularExpression pattern;
        QTextCharFormat format;
    };
    QVector<HighlightingRule> rules;

    QTextCharFormat keywordFormat;
    QTextCharFormat builtinFormat;
    QTextCharFormat stringFormat;
    QTextCharFormat commentFormat;
    QTextCharFormat numberFormat;
    QTextCharFormat promptFormat;
};
