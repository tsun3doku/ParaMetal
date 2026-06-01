#ifdef Q_OS_WIN

#include "FileAssociation.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QSettings>
#include <QString>

#define NOMINMAX
#include <windows.h>
#include <shlobj.h>
#pragma comment(lib, "shell32.lib")

static constexpr const char* Extension = ".pm";
static constexpr const char* ProgId = "ParaMetal.Project";
static constexpr const char* FriendlyName = "ParaMetal Project";

static QString exePath() {
    return QDir::toNativeSeparators(QCoreApplication::applicationFilePath());
}

static bool writeKey(const QString& path, const QString& name, const QString& value, QString* outError) {
    QSettings settings(path, QSettings::NativeFormat);
    settings.setValue(name, value);
    if (settings.status() != QSettings::NoError) {
        if (outError) {
            *outError = "Failed to write registry key: " + path + "\\" + name;
        }
        return false;
    }
    return true;
}

static void notifyExplorer() {
    SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
}

bool FileAssociation::isRegistered() {
    QSettings settings("HKEY_CURRENT_USER\\Software\\Classes\\" + QString(Extension), QSettings::NativeFormat);
    return settings.value("Default").toString() == ProgId;
}

bool FileAssociation::registerFileAssociation(QString* outError) {
    const QString appPath = exePath();
    const QString iconPath = appPath + ",0";
    const QString openCmd = "\"" + appPath + "\" \"%1\"";

    if (!writeKey("HKEY_CURRENT_USER\\Software\\Classes\\" + QString(Extension), "Default", ProgId, outError)) {
        return false;
    }
    if (!writeKey("HKEY_CURRENT_USER\\Software\\Classes\\" + QString(ProgId), "Default", FriendlyName, outError)) {
        return false;
    }
    if (!writeKey("HKEY_CURRENT_USER\\Software\\Classes\\" + QString(ProgId) + "\\DefaultIcon", "Default", iconPath, outError)) {
        return false;
    }
    if (!writeKey("HKEY_CURRENT_USER\\Software\\Classes\\" + QString(ProgId) + "\\shell\\open\\command", "Default", openCmd, outError)) {
        return false;
    }

    notifyExplorer();
    return true;
}

bool FileAssociation::unregisterFileAssociation(QString* outError) {
    QSettings extSettings("HKEY_CURRENT_USER\\Software\\Classes\\" + QString(Extension), QSettings::NativeFormat);
    extSettings.remove("");
    if (extSettings.status() != QSettings::NoError) {
        if (outError) {
            *outError = "Failed to remove registry key for " + QString(Extension) + ".";
        }
        return false;
    }

    QSettings progIdSettings("HKEY_CURRENT_USER\\Software\\Classes\\" + QString(ProgId), QSettings::NativeFormat);
    progIdSettings.remove("");
    if (progIdSettings.status() != QSettings::NoError) {
        if (outError) {
            *outError = "Failed to remove registry key for " + QString(ProgId) + ".";
        }
        return false;
    }

    notifyExplorer();
    return true;
}

#endif