#pragma once
#include <memory>
#include <iostream>
#include "../color.h"
#include "../engine/renderer.h"

class Mandelbrot2D : public Renderer
{
public:
    Mandelbrot2D(int w, int h, bool gpu);
    ~Mandelbrot2D();
    void render(Pixel* target, const mat4& cam) override;

private:
    bool gpu = false;

    int width;
    int height;
    int max_iter;

    void cpu_naive_mandel_2d(Pixel* pix, float size);
    void cuda_naive_mandel_2d(Pixel* pix, float size);
};

template<typename T>
void check(T result, char const *const func, const char *const file, int const line);

#define checkCudaErrors(val) check ( (val), #val, __FILE__, __LINE__ )

