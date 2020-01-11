#pragma once

#include <cstdint>
#include <glm/vec3.hpp>
#include <glm/matrix.hpp>

using namespace glm;

class Renderer
{
public:
    struct Pixel
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    virtual void render(Pixel* target, const mat4& cam) = 0;
};
