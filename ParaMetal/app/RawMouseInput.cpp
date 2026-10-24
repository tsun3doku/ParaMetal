#include "RawMouseInput.hpp"

#include <windows.h>

static constexpr wchar_t RawMouseWindowClass[] = L"ParaMetalRawMouseInputWindow";

static float pointerSpeedToMultiplier(UINT speed) {
    if (speed < 3) {
        return static_cast<float>(speed) / 32.0f;
    }
    if (speed < 11) {
        return static_cast<float>(speed - 2) / 8.0f;
    }
    return static_cast<float>(speed - 6) / 4.0f;
}

static LRESULT CALLBACK rawMouseWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_INPUT: {
        RawMouseInput* self = reinterpret_cast<RawMouseInput*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (self != nullptr) {
            self->handleRawInputMessage(reinterpret_cast<void*>(lParam));
        }
        break;
    }
    case WM_CLOSE:
        PostQuitMessage(0);
        return 0;
    default:
        break;
    }
    return DefWindowProcW(hwnd, message, wParam, lParam);
}

RawMouseInput::~RawMouseInput() {
    shutdown();
}

bool RawMouseInput::initialize() {
    shutdown();
    UINT pointerSpeed = 10;
    if (SystemParametersInfoW(SPI_GETMOUSESPEED, 0, &pointerSpeed, 0) != TRUE) {
        pointerSpeed = 10;
    }
    pointerScale = pointerSpeedToMultiplier(pointerSpeed);
    readyEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (readyEvent == nullptr) {
        return false;
    }
    thread = std::thread(&RawMouseInput::threadMain, this);
    WaitForSingleObject(static_cast<HANDLE>(readyEvent), 2000);
    CloseHandle(static_cast<HANDLE>(readyEvent));
    readyEvent = nullptr;
    return running.load(std::memory_order_acquire);
}

void RawMouseInput::shutdown() {
    if (thread.joinable()) {
        HWND window = static_cast<HWND>(messageWindow.load(std::memory_order_acquire));
        if (window != nullptr) {
            PostMessageW(window, WM_CLOSE, 0, 0);
        }
        thread.join();
    }
    running.store(false, std::memory_order_release);
    messageWindow.store(nullptr, std::memory_order_release);
    head.store(0, std::memory_order_relaxed);
    tail.store(0, std::memory_order_relaxed);
}

void RawMouseInput::threadMain() {
    const HINSTANCE instance = GetModuleHandleW(nullptr);
    WNDCLASSW windowClass{};
    windowClass.lpfnWndProc = rawMouseWindowProc;
    windowClass.hInstance = instance;
    windowClass.lpszClassName = RawMouseWindowClass;
    RegisterClassW(&windowClass);

    HWND window = CreateWindowExW(0, RawMouseWindowClass, L"", 0, 0, 0, 0, 0,
                                  HWND_MESSAGE, nullptr, instance, this);
    if (window != nullptr) {
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
        RAWINPUTDEVICE device{};
        device.usUsagePage = 0x01;
        device.usUsage = 0x02;
        device.dwFlags = RIDEV_INPUTSINK;
        device.hwndTarget = window;
        if (RegisterRawInputDevices(&device, 1, sizeof(device)) == TRUE) {
            messageWindow.store(window, std::memory_order_release);
            running.store(true, std::memory_order_release);
        }
    }

    if (readyEvent != nullptr) {
        SetEvent(static_cast<HANDLE>(readyEvent));
    }

    if (running.load(std::memory_order_acquire)) {
        MSG message{};
        while (GetMessageW(&message, nullptr, 0, 0) > 0) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }

    if (window != nullptr) {
        DestroyWindow(window);
    }
    messageWindow.store(nullptr, std::memory_order_release);
}

void RawMouseInput::handleRawInputMessage(void* rawHandle) {
    RAWINPUT raw{};
    UINT size = sizeof(raw);
    if (GetRawInputData(static_cast<HRAWINPUT>(rawHandle), RID_INPUT, &raw, &size, sizeof(RAWINPUTHEADER)) != size) {
        return;
    }
    if (raw.header.dwType != RIM_TYPEMOUSE) {
        return;
    }
    const RAWMOUSE& mouse = raw.data.mouse;
    if ((mouse.usFlags & MOUSE_MOVE_ABSOLUTE) != 0 || (mouse.lLastX == 0 && mouse.lLastY == 0)) {
        return;
    }
    LARGE_INTEGER timestamp{};
    QueryPerformanceCounter(&timestamp);
    pushSample(timestamp.QuadPart,
               static_cast<float>(mouse.lLastX) * pointerScale,
               static_cast<float>(mouse.lLastY) * pointerScale);
}

bool RawMouseInput::pushSample(int64_t timestampQpc, float dx, float dy) {
    const uint32_t currentHead = head.load(std::memory_order_relaxed);
    const uint32_t currentTail = tail.load(std::memory_order_acquire);
    if (currentHead - currentTail >= Capacity) {
        return false;
    }
    buffer[currentHead % Capacity] = Sample{timestampQpc, dx, dy};
    head.store(currentHead + 1, std::memory_order_release);
    return true;
}

size_t RawMouseInput::drain(Sample* out, size_t maxCount) {
    uint32_t currentTail = tail.load(std::memory_order_relaxed);
    const uint32_t currentHead = head.load(std::memory_order_acquire);
    size_t count = 0;
    while (currentTail != currentHead && count < maxCount) {
        out[count++] = buffer[currentTail % Capacity];
        ++currentTail;
    }
    tail.store(currentTail, std::memory_order_release);
    return count;
}
