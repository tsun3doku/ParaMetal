#pragma once

class IViewportInteraction {
public:
    virtual ~IViewportInteraction() = default;

    virtual bool handlePointerPress(float mouseX, float mouseY, int button = 1, bool shiftPressed = false) = 0;
    virtual bool handlePointerMove(float mouseX, float mouseY) = 0;
    virtual bool handlePointerRelease(float mouseX, float mouseY, int button = 1) = 0;
    virtual bool hasPointerCapture() const = 0;
};
