#include "FrameStats.hpp"

#include "render/SceneRenderer.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>

char FrameStats::capitalizeChar(unsigned char ch) {
    return static_cast<char>(std::toupper(ch));
}

void FrameStats::capitalizeLine(std::string& line) {
    for (char& ch : line) {
        ch = capitalizeChar(static_cast<unsigned char>(ch));
    }
}

void FrameStats::capitalizeLines(std::vector<std::string>& lines) const {
    for (std::string& line : lines) {
        capitalizeLine(line);
    }
}

uint32_t FrameStats::countWithinWindow(
    const std::chrono::high_resolution_clock::time_point* times,
    uint32_t writeIndex,
    uint32_t count,
    std::chrono::high_resolution_clock::time_point now,
    std::chrono::seconds window) {
    const auto boundary = now - window;
    uint32_t inWindow = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t index = (writeIndex + HistorySize - 1 - i) % HistorySize;
        if (times[index] < boundary) {
            break;
        }
        ++inWindow;
    }
    return inWindow;
}

void FrameStats::notePresentedFrame() {
    const auto now = std::chrono::high_resolution_clock::now();
    swapTimes[swapWriteIndex] = now;
    swapWriteIndex = (swapWriteIndex + 1) % HistorySize;
    if (swapCount < HistorySize) {
        ++swapCount;
    }

    const uint32_t inSecond = countWithinWindow(swapTimes, swapWriteIndex, swapCount, now, std::chrono::seconds(1));
    if (inSecond >= 2) {
        const uint32_t oldestIndex = (swapWriteIndex + HistorySize - inSecond) % HistorySize;
        const double spanMs = std::chrono::duration<double, std::milli>(now - swapTimes[oldestIndex]).count();
        if (spanMs > 0.5) {
            swapFpsValue.store(static_cast<float>((inSecond - 1) * 1000.0 / spanMs), std::memory_order_relaxed);
        }
    }

    const uint32_t inP95 = countWithinWindow(swapTimes, swapWriteIndex, swapCount, now, std::chrono::seconds(3));
    if (inP95 >= 8) {
        float intervals[HistorySize];
        const uint32_t intervalCount = inP95 - 1;
        for (uint32_t i = 0; i < intervalCount; ++i) {
            const uint32_t newer = (swapWriteIndex + HistorySize - 1 - i) % HistorySize;
            const uint32_t older = (swapWriteIndex + HistorySize - 2 - i) % HistorySize;
            intervals[i] = std::chrono::duration<float, std::milli>(swapTimes[newer] - swapTimes[older]).count();
        }
        std::sort(intervals, intervals + intervalCount);
        const uint32_t p95Index = std::min(intervalCount - 1, static_cast<uint32_t>(intervalCount * 0.95f));
        swapP95Ms.store(intervals[p95Index], std::memory_order_relaxed);
    }
}

void FrameStats::setSimStepTotal(uint64_t total) {
    const auto now = std::chrono::high_resolution_clock::now();
    simSamples[simWriteIndex] = { now, total };
    simWriteIndex = (simWriteIndex + 1) % HistorySize;
    if (simCount < HistorySize) {
        ++simCount;
    }

    const auto boundary = now - std::chrono::seconds(1);
    uint32_t inWindow = 0;
    for (uint32_t i = 0; i < simCount; ++i) {
        const uint32_t index = (simWriteIndex + HistorySize - 1 - i) % HistorySize;
        if (simSamples[index].time < boundary) {
            break;
        }
        ++inWindow;
    }
    if (inWindow >= 2) {
        const uint32_t oldestIndex = (simWriteIndex + HistorySize - inWindow) % HistorySize;
        const double spanSec = std::chrono::duration<double>(now - simSamples[oldestIndex].time).count();
        if (spanSec > 0.05 && total >= simSamples[oldestIndex].total) {
            simFpsValue.store(static_cast<float>((total - simSamples[oldestIndex].total) / spanSec), std::memory_order_relaxed);
        }
    }
}

std::vector<std::string> FrameStats::buildTimingLines(const GpuTimingStats* graphicsTiming, std::optional<float> computeGpuMs) {
    std::vector<std::string> lines;
    if (graphicsTiming) {
        lines.reserve(graphicsTiming->passTimings.size() + 4);
        for (const GpuPassTiming& passTiming : graphicsTiming->passTimings) {
            std::ostringstream passLine;
            passLine << std::fixed << std::setprecision(2) << passTiming.name << ": " << passTiming.ms << " ms";
            lines.push_back(passLine.str());
        }
    }
    else {
        lines.reserve(4);
    }

    std::ostringstream computeLine;
    if (computeGpuMs.has_value()) {
        // Sum of per-node GPU work across potentially overlapping nodes
        computeLine << std::fixed << std::setprecision(2) << "GPU COMPUTE: " << computeGpuMs.value() << " ms";
    }
    else {
        computeLine << "GPU COMPUTE: -- ms";
    }
    lines.push_back(computeLine.str());

    std::ostringstream graphicsLine;
    if (graphicsTiming) {
        graphicsLine << std::fixed << std::setprecision(2) << "GPU GRAPHICS: " << graphicsTiming->totalMs << " ms";
    }
    else {
        graphicsLine << "GPU GRAPHICS: -- ms";
    }
    lines.push_back(graphicsLine.str());

    std::ostringstream simLine;
    simLine << std::fixed << std::setprecision(1)
            << "SIM: " << simFpsValue.load(std::memory_order_relaxed) << " FPS";
    lines.push_back(simLine.str());

    std::ostringstream p95Line;
    p95Line << std::fixed << std::setprecision(1)
            << "FRAME P95: " << swapP95Ms.load(std::memory_order_relaxed) << " ms";
    lines.push_back(p95Line.str());

    return lines;
}
