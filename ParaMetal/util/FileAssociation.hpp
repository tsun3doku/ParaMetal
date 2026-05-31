#pragma once

#include <QString>

namespace fileassociation {

bool isRegistered();
bool registerFileAssociation(QString* outError = nullptr);
bool unregisterFileAssociation(QString* outError = nullptr);

} // namespace fileassociation
