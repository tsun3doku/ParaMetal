#include "GraphThread.hpp"

#include "GraphHost.hpp"

#include <QMetaObject>

GraphThread::GraphThread() {
    graphHost = new GraphHost();
    graphHost->moveToThread(&thread);
    QObject::connect(&thread, &QThread::started, graphHost, &GraphHost::initialize);
}

GraphThread::~GraphThread() {
    stop();
}

void GraphThread::start() {
    if (running) return;
    Q_ASSERT(graphHost);
    if (!graphHost) return;
    running = true;
    thread.start();
}

void GraphThread::stop() {
    if (!running) {
        delete graphHost;
        graphHost = nullptr;
        return;
    }
    QMetaObject::invokeMethod(graphHost, &GraphHost::shutdown, Qt::BlockingQueuedConnection);
    thread.quit();
    thread.wait();
    delete graphHost;
    graphHost = nullptr;
    running = false;
}
