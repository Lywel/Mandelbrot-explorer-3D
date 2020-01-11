#pragma once
#include <memory>
#include <iostream>
#include "../color.h"

const float mandel_center_x = -1.401155; // -0.1011;
const float mandel_center_y = 0; // 0.9563;

void cpu_naive_mandel_2d(rgba8_t* pix, int width, int height, float size);
void cuda_naive_mandel_2d(rgba8_t* pix, int width, int height, float size);

template<typename T>
void check(T result, char const *const func, const char *const file, int const line);

#define checkCudaErrors(val) check ( (val), #val, __FILE__, __LINE__ )
