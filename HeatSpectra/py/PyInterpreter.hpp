#pragma once

#include <string>

class PyInterpreter {
public:
    PyInterpreter();
    ~PyInterpreter();

    bool initialize();
    void shutdown();
    bool isInitialized() const;

    bool runSource(const std::string& source);
    std::string consumeOutput();
    std::string consumeError();
    std::string pythonVersion() const;

private:
    void* impl = nullptr;
};
