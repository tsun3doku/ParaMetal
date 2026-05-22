#include "PySyntaxHighlighter.hpp"

#include <QRegularExpression>

PySyntaxHighlighter::PySyntaxHighlighter(QTextDocument* parent)
    : QSyntaxHighlighter(parent) {

    keywordFormat.setForeground(QColor(198, 120, 221));
    keywordFormat.setFontWeight(QFont::Bold);
    const QStringList keywordPatterns = {
        QStringLiteral("\\band\\b"), QStringLiteral("\\bas\\b"), QStringLiteral("\\bassert\\b"),
        QStringLiteral("\\bbreak\\b"), QStringLiteral("\\bclass\\b"), QStringLiteral("\\bcontinue\\b"),
        QStringLiteral("\\bdef\\b"), QStringLiteral("\\bdel\\b"), QStringLiteral("\\belif\\b"),
        QStringLiteral("\\belse\\b"), QStringLiteral("\\bexcept\\b"), QStringLiteral("\\bFalse\\b"),
        QStringLiteral("\\bfinally\\b"), QStringLiteral("\\bfor\\b"), QStringLiteral("\\bfrom\\b"),
        QStringLiteral("\\bglobal\\b"), QStringLiteral("\\bif\\b"), QStringLiteral("\\bimport\\b"),
        QStringLiteral("\\bin\\b"), QStringLiteral("\\bis\\b"), QStringLiteral("\\blambda\\b"),
        QStringLiteral("\\bNone\\b"), QStringLiteral("\\bnonlocal\\b"), QStringLiteral("\\bnot\\b"),
        QStringLiteral("\\bor\\b"), QStringLiteral("\\bpass\\b"), QStringLiteral("\\braise\\b"),
        QStringLiteral("\\breturn\\b"), QStringLiteral("\\bTrue\\b"), QStringLiteral("\\btry\\b"),
        QStringLiteral("\\bwhile\\b"), QStringLiteral("\\bwith\\b"), QStringLiteral("\\byield\\b"),
    };
    for (const QString& pattern : keywordPatterns) {
        HighlightingRule rule;
        rule.pattern = QRegularExpression(pattern);
        rule.format = keywordFormat;
        rules.append(rule);
    }

    builtinFormat.setForeground(QColor(97, 175, 239));
    const QStringList builtinPatterns = {
        QStringLiteral("\\bprint\\b"), QStringLiteral("\\blen\\b"), QStringLiteral("\\brange\\b"),
        QStringLiteral("\\benumerate\\b"), QStringLiteral("\\bzip\\b"), QStringLiteral("\\bmap\\b"),
        QStringLiteral("\\bfilter\\b"), QStringLiteral("\\bsum\\b"), QStringLiteral("\\bmin\\b"),
        QStringLiteral("\\bmax\\b"), QStringLiteral("\\bsorted\\b"), QStringLiteral("\\bopen\\b"),
        QStringLiteral("\\bint\\b"), QStringLiteral("\\bfloat\\b"), QStringLiteral("\\bstr\\b"),
        QStringLiteral("\\blist\\b"), QStringLiteral("\\bdict\\b"), QStringLiteral("\\btuple\\b"),
        QStringLiteral("\\bset\\b"), QStringLiteral("\\btype\\b"), QStringLiteral("\\bisinstance\\b"),
        QStringLiteral("\\bhasattr\\b"), QStringLiteral("\\bgetattr\\b"), QStringLiteral("\\bsetattr\\b"),
    };
    for (const QString& pattern : builtinPatterns) {
        HighlightingRule rule;
        rule.pattern = QRegularExpression(pattern);
        rule.format = builtinFormat;
        rules.append(rule);
    }

    stringFormat.setForeground(QColor(152, 195, 121));
    HighlightingRule singleQuoteRule;
    singleQuoteRule.pattern = QRegularExpression(QStringLiteral("'[^']*'"));
    singleQuoteRule.format = stringFormat;
    rules.append(singleQuoteRule);

    HighlightingRule doubleQuoteRule;
    doubleQuoteRule.pattern = QRegularExpression(QStringLiteral("\"[^\"]*\""));
    doubleQuoteRule.format = stringFormat;
    rules.append(doubleQuoteRule);

    commentFormat.setForeground(QColor(92, 99, 112));
    HighlightingRule commentRule;
    commentRule.pattern = QRegularExpression(QStringLiteral("#[^\n]*"));
    commentRule.format = commentFormat;
    rules.append(commentRule);

    numberFormat.setForeground(QColor(97, 175, 239));
    HighlightingRule numberRule;
    numberRule.pattern = QRegularExpression(QStringLiteral("\\b[0-9]+\\.?[0-9]*\\b"));
    numberRule.format = numberFormat;
    rules.append(numberRule);

    promptFormat.setForeground(QColor(150, 150, 150));
    promptFormat.setFontWeight(QFont::Bold);
    HighlightingRule promptRule;
    promptRule.pattern = QRegularExpression(QStringLiteral("^(>>>|\\.\\.\\.)"));
    promptRule.format = promptFormat;
    rules.append(promptRule);
}

void PySyntaxHighlighter::highlightBlock(const QString& text) {
    for (const HighlightingRule& rule : rules) {
        QRegularExpressionMatchIterator matchIterator = rule.pattern.globalMatch(text);
        while (matchIterator.hasNext()) {
            QRegularExpressionMatch match = matchIterator.next();
            setFormat(match.capturedStart(), match.capturedLength(), rule.format);
        }
    }
}
