#pragma once
#include <memory>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <glm/vec3.hpp>
#include <glm/matrix.hpp>
#include <glm/ext/matrix_projection.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <glm/geometric.hpp>

#include "gui.h"
#include "renderer.h"

using namespace glm;

class Engine
{
public:
    Engine(int width, int height);
    ~Engine();

    void run(GUI* gui, Renderer* renderer);

private:
    void handle_event(const GUI::Event ev, int mouse_x, int mouse_y);
    void auto_move(int elasped_ms);

    vec3 position = vec3(0, 0, 3);
    vec3 speed = vec3(0.1);
    vec3 forward = vec3(0, 0, -1);
    vec3 up = vec3(0, 1, 0);
    float yaw = -90;
    float pitch = 0;
    mat4 cam = mat4();

    std::chrono::high_resolution_clock::time_point last_frame{std::chrono::high_resolution_clock::now()};
    float time = 0;

    Renderer::Pixel* pixels;
};
