#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct GpuTimingStats;

class FrameStats {
public:
    std::vector<std::string> buildTimingLines(const GpuTimingStats* graphicsTiming, std::optional<float> computeGpuMs);
    void capitalizeLines(std::vector<std::string>& lines) const;

    // Simulation GPU timing for presentation only
    void setComputeGpuMs(std::optional<float> ms) { computeGpuMs = ms; }
    std::optional<float> getComputeGpuMs() const { return computeGpuMs; }
    float getSwapFpsValue() const { return swapFpsValue.load(std::memory_order_relaxed); }

    // Presentation telemetry: swap cadence and sim step rate.
    void notePresentedFrame();
    void setSimStepTotal(uint64_t total);

private:
    static char capitalizeChar(unsigned char ch);
    static void capitalizeLine(std::string& line);
    static uint32_t countWithinWindow(
        const std::chrono::high_resolution_clock::time_point* times,
        uint32_t writeIndex,
        uint32_t count,
        std::chrono::high_resolution_clock::time_point now,
        std::chrono::seconds window);

    std::optional<float> computeGpuMs;

    static constexpr uint32_t HistorySize = 256;

    std::chrono::high_resolution_clock::time_point swapTimes[HistorySize];
    uint32_t swapWriteIndex = 0;
    uint32_t swapCount = 0;
    std::atomic<float> swapFpsValue{0.0f};
    std::atomic<float> swapP95Ms{0.0f};

    struct SimSample {
        std::chrono::high_resolution_clock::time_point time;
        uint64_t total = 0;
    };
    SimSample simSamples[HistorySize];
    uint32_t simWriteIndex = 0;
    uint32_t simCount = 0;
    std::atomic<float> simFpsValue{0.0f};
};
