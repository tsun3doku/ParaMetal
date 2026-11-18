#pragma once

#include "CudaEvent.cuh"

#include <cuda_runtime.h>

#include <utility>

namespace cudaUtils {

enum class GpuTimerStatus {
    NotAvailable,
    Pending,
    Ready,
    Failed,
};

// Best-effort GPU interval timer. Timing is purely diagnostic, so failures are
// logged and cleared instead of being left in the thread's sticky error state.
class GpuTimer {
public:
    GpuTimer() = default;

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    GpuTimer(GpuTimer&& other) noexcept
        : beginEvent_(std::move(other.beginEvent_)),
          endEvent_(std::move(other.endEvent_)),
          beginRecorded_(std::exchange(other.beginRecorded_, false)),
          endRecorded_(std::exchange(other.endRecorded_, false)) {}

    GpuTimer& operator=(GpuTimer&& other) noexcept {
        if (this != &other) {
            reset();
            beginEvent_ = std::move(other.beginEvent_);
            endEvent_ = std::move(other.endEvent_);
            beginRecorded_ = std::exchange(other.beginRecorded_, false);
            endRecorded_ = std::exchange(other.endRecorded_, false);
        }
        return *this;
    }

    bool create() {
        if (!reset()) return false;
        if (!beginEvent_.create() || !endEvent_.create()) {
            reset();
            return false;
        }
        return true;
    }

    bool reset() {
        beginRecorded_ = false;
        endRecorded_ = false;
        const bool beginOk = beginEvent_.reset();
        const bool endOk = endEvent_.reset();
        return beginOk && endOk;
    }

    bool begin(cudaStream_t stream) {
        const cudaError_t result = cudaEventRecord(beginEvent_.handle(), stream);
        if (result != cudaSuccess) {
            clearError(result, "record GPU timer begin event");
            beginRecorded_ = false;
            endRecorded_ = false;
            return false;
        }
        beginRecorded_ = true;
        endRecorded_ = false;
        return true;
    }

    bool end(cudaStream_t stream) {
        if (!beginRecorded_) return false;
        const cudaError_t result = cudaEventRecord(endEvent_.handle(), stream);
        if (result != cudaSuccess) {
            clearError(result, "record GPU timer end event");
            beginRecorded_ = false;
            endRecorded_ = false;
            return false;
        }
        endRecorded_ = true;
        return true;
    }

    // Returns Ready with a valid interval once the recorded end event completes.
    // Pending means the interval is still in flight; the interval is consumed on
    // Ready, so the next begin() starts a fresh measurement.
    GpuTimerStatus pollElapsed(float& milliseconds) {
        if (!beginRecorded_ || !endRecorded_) return GpuTimerStatus::NotAvailable;

        const cudaError_t query = cudaEventQuery(endEvent_.handle());
        if (query == cudaErrorNotReady) return GpuTimerStatus::Pending;
        if (query != cudaSuccess) {
            clearError(query, "query GPU timer end event");
            beginRecorded_ = false;
            endRecorded_ = false;
            return GpuTimerStatus::Failed;
        }

        float elapsedMs = 0.0f;
        const cudaError_t elapsed = cudaEventElapsedTime(&elapsedMs, beginEvent_.handle(), endEvent_.handle());
        if (elapsed != cudaSuccess) {
            clearError(elapsed, "measure GPU timer interval");
            beginRecorded_ = false;
            endRecorded_ = false;
            return GpuTimerStatus::Failed;
        }

        milliseconds = elapsedMs;
        beginRecorded_ = false;
        endRecorded_ = false;
        return GpuTimerStatus::Ready;
    }

private:
    static void clearError(cudaError_t result, const char* operation) {
        cudaTrace(result, operation);
        (void)cudaGetLastError();
    }

    CudaEvent beginEvent_;
    CudaEvent endEvent_;
    bool beginRecorded_ = false;
    bool endRecorded_ = false;
};

} // namespace cudaUtils
