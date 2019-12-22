#pragma once
#include "color.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>

#include <SDL.h>
#include <SDL_ttf.h>

class Window
{
public:
    Window(int _width, int _height, const char* _title);
    ~Window();

    void render(void* pixels);
    bool input_pool();

    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;
    SDL_Texture* texture = NULL;

    const int width, height;
    const char* title;

private:
    void render_infos(const std::string &str);

    TTF_Font* info_font = NULL;

    Uint32 startclock = 0;
    Uint32 deltaclock = 0;
    Uint32 currentFPS = 0;
};