#pragma once
#include <algorithm>
#include <cmath>
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

using namespace glm;

class Mandelbrot3D : public Renderer
{
public:
    Mandelbrot3D(int w, int h, int i=100);
    void render(Pixel* target, const mat4& cam) override;

private:
    void cpu_render(Pixel* target, const mat4& cam);
    void cuda_render(Pixel* target, const mat4& cam);
 
    vec3 render_px(const vec2& px, const mat4& cam);

    const int width;
    const int height;
    const int max_iter;

    mat4 last_cam = mat4(0);
};

float compute_iter(const vec3& pos, int max_iter, float max_val, int exponent);

vec3 render(const vec2& p, const vec2& resolution, const mat4& cam);
//void cuda_render_px(Renderer::Pixel* colors, int width, int height, const mat4& cam);

template<typename T>
void check(T result, char const *const func, const char *const file, int const line);

#define checkCudaErrors(val) check ( (val), #val, __FILE__, __LINE__ )
