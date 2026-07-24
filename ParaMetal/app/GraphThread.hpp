#pragma once

#include <QThread>

class GraphHost;

class GraphThread final {
public:
    GraphThread();
    ~GraphThread();

    GraphThread(const GraphThread&) = delete;
    GraphThread& operator=(const GraphThread&) = delete;

    GraphHost* host() const { return graphHost; }
    void start();
    void stop();

private:
    QThread thread;
    GraphHost* graphHost = nullptr;
    bool running = false;
};
