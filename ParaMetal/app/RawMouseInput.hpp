#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>

class RawMouseInput {
public:
    struct Sample {
        int64_t timestampQpc;
        float dx;  
        float dy;
    };

    RawMouseInput() = default;
    ~RawMouseInput();
    RawMouseInput(const RawMouseInput&) = delete;
    RawMouseInput& operator=(const RawMouseInput&) = delete;

    bool initialize();
    void shutdown();
    bool isInitialized() const { return running.load(std::memory_order_acquire); }

    size_t drain(Sample* out, size_t maxCount);

    void handleRawInputMessage(void* rawHandle);

private:
    void threadMain();
    bool pushSample(int64_t timestampQpc, float dx, float dy);

    static constexpr uint32_t Capacity = 512;
    std::thread thread;
    std::atomic<void*> messageWindow{nullptr};
    void* readyEvent = nullptr;
    std::atomic<bool> running{false};
    float pointerScale = 1.0f;
    Sample buffer[Capacity]{};
    std::atomic<uint32_t> head{0};
    std::atomic<uint32_t> tail{0};
};
