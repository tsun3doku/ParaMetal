#ifdef Q_OS_LINUX

#include "FileAssociation.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QImageWriter>
#include <QProcess>
#include <QStandardPaths>
#include <QString>
#include <QTextStream>

static constexpr const char* DesktopFileName = "parametal.desktop";
static constexpr const char* MimePackageFileName = "application-x-parametal.xml";
static constexpr const char* IconBaseName = "parametal";
static constexpr const char* SourceIconSubPath = "resources/Logo/app_icon.png";

static const int IconSizes[] = {48, 128, 256, 512};

static QString xdgDataHome() {
    return QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation);
}

static QString sourceIconPath() {
    return QDir(QCoreApplication::applicationDirPath()).filePath(SourceIconSubPath);
}

static bool ensureDir(const QString& path) {
    return QDir().mkpath(path);
}

static bool writeTextFile(const QString& path, const QString& content, QString* outError) {
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (outError) {
            *outError = "Failed to open " + path + " for writing.";
        }
        return false;
    }
    QTextStream stream(&file);
    stream << content;
    file.close();
    return true;
}

static bool installIcons(QString* outError) {
    const QString sourcePath = sourceIconPath();
    if (!QFile::exists(sourcePath)) {
        if (outError) {
            *outError = "Source icon not found at " + sourcePath;
        }
        return false;
    }

    QImage sourceImage(sourcePath);
    if (sourceImage.isNull()) {
        if (outError) {
            *outError = "Failed to load source icon from " + sourcePath;
        }
        return false;
    }

    const QString iconsBase = xdgDataHome() + "/icons/hicolor";

    for (int size : IconSizes) {
        QString dir = QString("%1/%2x%2/apps").arg(iconsBase).arg(size);
        if (!ensureDir(dir)) {
            if (outError) {
                *outError = "Failed to create directory: " + dir;
            }
            return false;
        }

        QImage scaled = sourceImage.scaled(size, size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        QString destPath = dir + "/" + IconBaseName + ".png";
        QImageWriter writer(destPath, "png");
        if (!writer.write(scaled)) {
            if (outError) {
                *outError = "Failed to write icon to " + destPath + ": " + writer.errorString();
            }
            return false;
        }
    }
    return true;
}

static bool writeDesktopFile(QString* outError) {
    const QString appPath = QDir::toNativeSeparators(QCoreApplication::applicationFilePath());
    const QString content = QString(
        "[Desktop Entry]\n"
        "Name=ParaMetal\n"
        "Comment=Transient heat transfer simulator\n"
        "Exec=%1 %%f\n"
        "Icon=%2\n"
        "Type=Application\n"
        "Terminal=false\n"
        "MimeType=application/x-parametal;\n"
        "Categories=Graphics;Engineering;Science;\n"
    ).arg(appPath, IconBaseName);

    const QString path = xdgDataHome() + "/applications/" + DesktopFileName;
    if (!ensureDir(QFileInfo(path).absolutePath())) {
        if (outError) {
            *outError = "Failed to create applications directory.";
        }
        return false;
    }
    return writeTextFile(path, content, outError);
}

static bool writeMimePackage(QString* outError) {
    const QString content =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<mime-type type=\"application/x-parametal\">\n"
        "  <comment>ParaMetal Project</comment>\n"
        "  <glob pattern=\"*.pm\"/>\n"
        "</mime-type>\n";

    const QString path = xdgDataHome() + "/mime/packages/" + MimePackageFileName;
    if (!ensureDir(QFileInfo(path).absolutePath())) {
        if (outError) {
            *outError = "Failed to create mime packages directory.";
        }
        return false;
    }
    return writeTextFile(path, content, outError);
}

static bool runUpdateCommand(const QString& command, const QString& arg) {
    QProcess process;
    process.start(command, QStringList{arg});
    if (!process.waitForFinished(5000)) {
        return false;
    }
    return process.exitCode() == 0;
}

static void updateDatabases() {
    runUpdateCommand("update-desktop-database", xdgDataHome() + "/applications");
    runUpdateCommand("update-mime-database", xdgDataHome() + "/mime");
}

static bool removeFileIfExists(const QString& path) {
    if (!QFile::exists(path)) {
        return true;
    }
    return QFile::remove(path);
}

bool FileAssociation::isRegistered() {
    const QString desktopPath = xdgDataHome() + "/applications/" + DesktopFileName;
    if (!QFile::exists(desktopPath)) {
        return false;
    }
    QFile file(desktopPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    const QString content = QString::fromUtf8(file.readAll());
    return content.contains("MimeType=application/x-parametal");
}

bool FileAssociation::registerFileAssociation(QString* outError) {
    if (!installIcons(outError)) {
        return false;
    }
    if (!writeDesktopFile(outError)) {
        return false;
    }
    if (!writeMimePackage(outError)) {
        return false;
    }
    updateDatabases();
    return true;
}

bool FileAssociation::unregisterFileAssociation(QString* outError) {
    bool ok = true;
    QString accumulatedError;

    auto tryRemove = [&](const QString& path) {
        if (!removeFileIfExists(path)) {
            accumulatedError += "Failed to remove " + path + "\n";
            ok = false;
        }
    };

    const QString dataHome = xdgDataHome();
    tryRemove(dataHome + "/applications/" + DesktopFileName);
    tryRemove(dataHome + "/mime/packages/" + MimePackageFileName);

    for (int size : IconSizes) {
        tryRemove(QString("%1/icons/hicolor/%2x%2/apps/%3.png")
                  .arg(dataHome).arg(size).arg(IconBaseName));
    }

    updateDatabases();

    if (!ok && outError) {
        *outError = accumulatedError.trimmed();
    }
    return ok;
}

#endif