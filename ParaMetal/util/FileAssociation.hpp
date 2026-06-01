#pragma once

#include <QString>

class FileAssociation {
public:
    static bool isRegistered();
    static bool registerFileAssociation(QString* outError = nullptr);
    static bool unregisterFileAssociation(QString* outError = nullptr);
};