#include "window.h"

Window::Window(int _width, int _height, const char* _title)
    : width(_width), height(_height), title(_title)
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cerr << "[ERROR] Could not initialize SDL: "
            << SDL_GetError()
            << std::endl;
        exit(1);
    }
    if (TTF_Init() < 0)
    {
        std::cerr << "[ERROR] Could not initialize TTF: "
            << SDL_GetError()
            << std::endl;
        exit(1);
    }
    if (!(window = SDL_CreateWindow(title, 0, 0, width, height, SDL_WINDOW_OPENGL )))//| SDL_WINDOW_FULLSCREEN)))
    {
        std::cerr << "[ERROR] Could not initialize window: "
            << SDL_GetError()
            << std::endl;
        exit(1);
    }
    if (!(renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED)))
    {
        std::cerr << "[ERROR] Could not initialize renderer: "
            << SDL_GetError()
            << std::endl;
        exit(1);
    }
    if (!(texture = SDL_CreateTexture(renderer,
                    SDL_PIXELFORMAT_ABGR8888,
                    SDL_TEXTUREACCESS_STREAMING, width, height)))
    {
        std::cerr << "[ERROR] Could not initialize texture: "
            << SDL_GetError()
            << std::endl;
        exit(1);
    }

    if (!(info_font = TTF_OpenFont("assets/Sans.ttf", 18)))
    {
        std::cerr << "[ERROR] Could not load ttf font: "
            << SDL_GetError()
            << std::endl;
        exit(1);
    }

    SDL_SetRelativeMouseMode(SDL_TRUE);

    startclock = SDL_GetTicks();
}

Window::~Window()
{
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

GUI::Event
Window::get_events(int* mouse_x, int* mouse_y)
{
    SDL_Event e;
    if (SDL_PollEvent(&e))
    {
        if (e.type == SDL_QUIT)
        {
            return GUI::Event::Quit;
        }
        if (e.type == SDL_KEYDOWN)
        {
            switch (e.key.keysym.sym)
            {
            case SDLK_ESCAPE:
                return GUI::Event::Quit;
            case SDLK_UP:
                return GUI::Event::Forward;
            case SDLK_DOWN:
                return GUI::Event::Backward;
            case SDLK_LEFT:
                return GUI::Event::Left;
            case SDLK_RIGHT:
                return GUI::Event::Right;
            case SDLK_SPACE:
                return GUI::Event::ToggleAnimate;
            }
        }
        if (e.type == SDL_MOUSEMOTION)
        {
            *mouse_x = e.motion.xrel;
            *mouse_y = e.motion.yrel;
        }
    }
    return GUI::Event::None;
}

void
Window::set_pixels(const void* pixels)
{
    deltaclock = SDL_GetTicks() - startclock;
    startclock = SDL_GetTicks();
    if ( deltaclock != 0 )
        currentFPS = 1000 / deltaclock;

    infos
        << "FPS: " << currentFPS << std::endl
        << "elapsed: " << deltaclock << std::endl;

    // Update with pixels
    SDL_UpdateTexture(texture, NULL, pixels, width * sizeof(rgba8_t));
    SDL_RenderCopy(renderer, texture, NULL, NULL);
}

void
Window::render()
{
    render_infos();
    SDL_RenderPresent(renderer);
    SDL_RenderClear(renderer);
}

void
Window::render_infos()
{
    const std::string str = infos.str();
    SDL_Surface* info_surface = NULL;
    SDL_Texture* info_texture = NULL;
    SDL_Color info_color{0, 125, 125, 255};
    SDL_Rect info_rect{0, 0, 0, 0};
    int prev_info_h = 0;

    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token))
    {
        info_surface = TTF_RenderText_Solid(info_font, token.c_str(), info_color);
        info_texture = SDL_CreateTextureFromSurface(renderer, info_surface);

        info_rect.y += prev_info_h;
        info_rect.w = info_surface->w;
        info_rect.h = info_surface->h;

        prev_info_h = info_surface->h;

        SDL_RenderCopy(renderer, info_texture, NULL, &info_rect);
    }

    SDL_FreeSurface(info_surface);
    SDL_DestroyTexture(info_texture);
    infos.str("");
    infos.clear();
}
