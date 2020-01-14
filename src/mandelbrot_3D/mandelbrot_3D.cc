#include "mandelbrot_3D.h"

#include <glm/gtx/string_cast.hpp>

Mandelbrot3D::Mandelbrot3D(int w, int h, bool cuda_enabled)
    : width(w), height(h), gpu(cuda_enabled)
{
    if (gpu)
        cuda_init();
}

Mandelbrot3D::~Mandelbrot3D()
{
    if (gpu)
        cuda_dinit();
}

void
Mandelbrot3D::render(Pixel* target, const mat4& cam)
{
    // cuda_naive_mandel_2d((rgba8_t*)target, width, height, pos.z);
    if (gpu)
        cuda_render(target, cam);
    else
        cpu_render(target, cam);
}

vec3 Mandelbrot3D::render_px(const vec2& px, const mat4& cam)
{
    return ::render(px, vec2(width, height), cam);
    // Antialiasing disabled below for performances
    /* vec3 col(0.0); */

    /* float aa = 2; */
    /* for (int j=0; j < aa; j++) */
    /* { */
    /*     for (int i=0; i < aa; i++) */
    /*     { */
    /*         glm::vec2 xy = glm::vec2(x, y) + (glm::vec2(i, j) / aa); */
    /*         col += ::render(xy.x, xy.y, width, height, pos, rot); */
    /*     } */
    /*     col /= aa * aa; */
    /* } */
}

void Mandelbrot3D::cpu_render(Pixel* target, const mat4& cam)
{
    if (cam == last_cam)
        return;
    last_cam = cam;

    /* std::cout << glm::to_string(cam) << std::endl; */
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            vec3 col = render_px(vec2(j, i), cam);
            target[i * width + j] = {uint8_t(col.x * 255), uint8_t(col.y * 255), uint8_t(col.z * 255), 255};
        }
    }
}
