#pragma once
#include <algorithm>
#include <iostream>

//#define GLM_FORCE_CUDA 1
//#define GLM_COMPILER 0
#define CUDA_VERSION 8000

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/matrix.hpp>
#include <glm/ext/matrix_projection.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../engine/renderer.h"
#include "ray.h"

class Mandelbrot3D : public Renderer
{
public:
    Mandelbrot3D(int w, int h, bool cuda_enabled);
    ~Mandelbrot3D();
    void render(Pixel* target, const glm::mat4& cam) override;

private:
    bool gpu = false;

    void cpu_render(Pixel* target, const glm::mat4& cam);
    void cuda_render(Pixel* target, const glm::mat4& cam);
    void cuda_init();
    void cuda_dinit();

    glm::vec3 render_px(const glm::vec2& px, const glm::mat4& cam);

    const int width;
    const int height;

    glm::mat4 last_cam = glm::mat4(0);
    char* colors;
    size_t pitch;
};

float compute_iter(const glm::vec3& pos, int max_iter, float max_val, int exponent);

glm::vec3 render(const glm::vec2& p, const glm::vec2& resolution, const glm::mat4 cam);

template<typename T>
void check(T result, char const *const func, const char *const file, int const line);

#define checkCudaErrors(val) check ( (val), #val, __FILE__, __LINE__ )
