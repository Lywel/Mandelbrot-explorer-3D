#pragma once
#include "../color.h"
#include "../engine/gui.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <SDL.h>
#include <SDL_ttf.h>

class Window : public GUI
{
public:
    Window(int _width, int _height, const char* _title);
    ~Window();

    void set_pixels(const void* pixels) override;
    GUI::Event get_events(int* mouse_x, int* mouse_y) override;
    void render() override;

    const int width, height;
    const char* title;

private:
    void render_infos();

    TTF_Font* info_font = NULL;
    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;
    SDL_Texture* texture = NULL;

    Uint32 startclock = 0;
    Uint32 deltaclock = 0;
    Uint32 currentFPS = 0;
};
