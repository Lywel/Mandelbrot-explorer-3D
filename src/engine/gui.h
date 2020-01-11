#pragma once
#include <string>
#include <sstream>

class GUI
{
public:
    enum class Event
    {
        None,
        Quit,
        Forward,
        Backward,
        Left,
        Right
    };
    virtual void set_pixels(const void* pixels) = 0;
    virtual Event get_events(int* mouse_x, int* mouse_y) = 0;
    virtual void render() = 0;

    std::ostringstream infos;
};
