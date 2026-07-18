#include "SerialPort.hpp"

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <string>

static std::string windowsError(const char* operation) {
    const DWORD error = GetLastError();
    return std::string(operation) + " failed (Windows error " + std::to_string(error) + ")";
}

class WindowsSerialPort final : public SerialPort {
public:
    ~WindowsSerialPort() override { close(); }

    bool open(const SerialPortConfig& config, std::string& outError) override {
        close();
        if (config.portName.empty()) {
            outError = "No serial port selected";
            return false;
        }

        const std::string devicePath = "\\\\.\\" + config.portName;
        handle = CreateFileA(devicePath.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            outError = windowsError("Open serial port");
            return false;
        }

        DCB dcb{};
        dcb.DCBlength = sizeof(dcb);
        if (!GetCommState(handle, &dcb)) {
            outError = windowsError("Read serial configuration");
            close();
            return false;
        }
        dcb.BaudRate = config.baudRate;
        dcb.ByteSize = 8;
        dcb.Parity = NOPARITY;
        dcb.StopBits = ONESTOPBIT;
        dcb.fBinary = TRUE;
        dcb.fParity = FALSE;
        dcb.fOutxCtsFlow = FALSE;
        dcb.fOutxDsrFlow = FALSE;
        dcb.fDtrControl = DTR_CONTROL_ENABLE;
        dcb.fDsrSensitivity = FALSE;
        dcb.fOutX = FALSE;
        dcb.fInX = FALSE;
        dcb.fRtsControl = RTS_CONTROL_ENABLE;
        if (!SetCommState(handle, &dcb)) {
            outError = windowsError("Configure serial port");
            close();
            return false;
        }

        COMMTIMEOUTS timeouts{};
        timeouts.ReadIntervalTimeout = MAXDWORD;
        if (!SetCommTimeouts(handle, &timeouts)) {
            outError = windowsError("Configure serial timeouts");
            close();
            return false;
        }
        SetupComm(handle, 4096, 4096);
        PurgeComm(handle, PURGE_RXCLEAR | PURGE_TXCLEAR);
        outError.clear();
        return true;
    }

    void close() override {
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
            handle = INVALID_HANDLE_VALUE;
        }
    }

    bool isOpen() const override { return handle != INVALID_HANDLE_VALUE; }

    size_t readAvailable(uint8_t* destination, size_t capacity, std::string& outError) override {
        if (!isOpen() || !destination || capacity == 0) return 0;
        DWORD errors = 0;
        COMSTAT status{};
        if (!ClearCommError(handle, &errors, &status)) {
            outError = windowsError("Query serial port");
            return 0;
        }
        const DWORD requested = static_cast<DWORD>(std::min<size_t>(capacity, status.cbInQue));
        if (requested == 0) return 0;
        DWORD bytesRead = 0;
        if (!ReadFile(handle, destination, requested, &bytesRead, nullptr)) {
            outError = windowsError("Read serial port");
            return 0;
        }
        outError.clear();
        return bytesRead;
    }

private:
    HANDLE handle = INVALID_HANDLE_VALUE;
};

std::vector<SerialPortInfo> SerialPort::enumeratePorts() {
    std::vector<SerialPortInfo> ports;
    wchar_t target[512]{};
    for (uint32_t index = 1; index <= 256; ++index) {
        const std::wstring name = L"COM" + std::to_wstring(index);
        if (QueryDosDeviceW(name.c_str(), target, static_cast<DWORD>(std::size(target))) != 0) {
            const std::string portName = "COM" + std::to_string(index);
            ports.push_back({portName, portName});
        }
    }
    return ports;
}

std::unique_ptr<SerialPort> SerialPort::create() {
    return std::make_unique<WindowsSerialPort>();
}
