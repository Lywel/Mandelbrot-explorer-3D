#include "mandel.h"

template<typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
      std::cerr << file << ":" << line << ": CUDA error in function ‘" << func << "’:"
        << " code=" << static_cast<unsigned int>(result)
        << "(" << cudaGetErrorName(result) << ") "
        << cudaGetErrorString(result) << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

__host__ __device__ rgba8_t
heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return rgba8_t{0, g, 255, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return rgba8_t{0, 255, b, 255};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return rgba8_t{r, 255, 0, 255};
  }
  else if (x < 1.f)
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return rgba8_t{255, b, 0, 255};
  }
  else
  {
    return rgba8_t{0, 0, 0, 255};
  }
}

__host__ __device__ std::uint8_t
compute_pixel(const float x0, const float y0, const int max_iter) {
    float xx = 0.0f, yy = 0.0f, xtemp;
    unsigned char i = 0;
    for (; i < max_iter && xx*xx + yy*yy < 4; ++i) {
        xtemp = xx*xx - yy*yy + x0;
        yy = 2*xx*yy + y0;
        xx = xtemp;
    }
    return i;
}


__host__ void
cpu_naive_mandel_iter(std::uint8_t* iters, int width, int height, int max_iter, float size)
{
    const float delta = size / width;

    float x0, y0;
    for (int py = 0; py < height; py++)
    {
        for (int px = 0; px < width; px++)
        {
            x0 = mandel_center_x + delta * (px - width / 2.f);
            y0 = mandel_center_y + delta * (py - height / 2.f);
            //x0 = (float)px / width * 3.5f - 2.5f;
            //y0 = (float)py / height * 2.f - 1.f;

            iters[py * width + px] = compute_pixel(x0, y0, max_iter);
        }
    }
}


__host__ void
cpu_naive_mandel_2d(rgba8_t* pix, int width, int height, float size)
{
    std::uint8_t iters[width * height];
    cpu_naive_mandel_iter(iters, width, height, 255, size);
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            pix[i * width + j] = heat_lut(iters[i * width + j] / 255.f);
}


__global__ void
cuda_compute_iter(std::uint8_t* iters, int width, int height, int max_iter, float size)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const float delta = size / width;

    /* float x0 = (float)x / width * 3.5f - 2.5f; */
    /* float y0 = (float)y / height * 2.f - 1.f; */
    float x0 = mandel_center_x + delta * (x - width / 2.f);
    float y0 = mandel_center_y + delta * (y - height / 2.f);

    iters[y * width + x] = compute_pixel(x0, y0, max_iter);
}


__global__ void
cuda_compute_hist(const std::uint8_t* iters, int width, int height, int max_iter, uint32_t* hist)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    atomicAdd(&hist[iters[y * width + x]], 1);
}


__global__ void
cuda_compute_LUT(const std::uint32_t* hist, int max_iter, rgba8_t* LUT)
{
    std::uint32_t partial_sum[256];

    for (int i = 0; i <= max_iter; ++i)
    {
        partial_sum[i] = hist[i];
        if (i > 0)
            partial_sum[i] += partial_sum[i-1];
    }

    for (int i = 0; i <= max_iter; ++i)
        LUT[i] = heat_lut((float)partial_sum[i] / partial_sum[255]);
}



__global__ void
cuda_apply_LUT(rgba8_t* colors, int width, int height, int max_iter,
    std::uint8_t* iters, const rgba8_t* LUT)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    colors[y * width + x] = LUT[iters[y * width + x]];
}

void
cuda_naive_mandel_2d(rgba8_t* hostBuffer, int width, int height, float size)
{
    int max_iter = 255;

    std::uint8_t* iters;
    checkCudaErrors(cudaMalloc(&iters, width * sizeof(std::uint8_t) * height));

    rgba8_t* colors;
    checkCudaErrors(cudaMalloc(&colors, width * sizeof(rgba8_t) * height));

    std::uint32_t* hist;
    checkCudaErrors(cudaMalloc(&hist, sizeof(std::uint32_t) * (max_iter + 1)));

    rgba8_t* LUT;
    checkCudaErrors(cudaMalloc(&LUT, sizeof(rgba8_t) * (max_iter + 1)));

    {
        int bsize = 32;
        int w     = std::ceil((float)width / bsize);
        int h     = std::ceil((float)height / bsize);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);
        // Compute iterations
        cuda_compute_iter<<<dimGrid, dimBlock>>>(iters, width, height, max_iter, size);

        // Copute iterations histogram
        cuda_compute_hist<<<dimGrid, dimBlock>>>(iters, width, height, max_iter, hist);

        // Copmute LUT
        cuda_compute_LUT<<<max_iter + 1, 1>>>(hist, max_iter, LUT);

        // Apply LUT
        cuda_apply_LUT<<<dimGrid, dimBlock>>>(colors, width, height, 255, iters, LUT);
    }

    // Copy back to main memory
    checkCudaErrors(cudaMemcpy(hostBuffer, colors, width * sizeof(rgba8_t) * height, cudaMemcpyDeviceToHost));

    // Free
    checkCudaErrors(cudaFree(LUT));
    checkCudaErrors(cudaFree(hist));
    checkCudaErrors(cudaFree(iters));
    checkCudaErrors(cudaFree(colors));
}
