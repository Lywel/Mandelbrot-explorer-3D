#include "mandelbrot.h"

const float mandel_center_x = -1.401155; // -0.1011;
const float mandel_center_y = 0; // 0.9563;

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

__host__ __device__ Renderer::Pixel
heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return Renderer::Pixel{0, g, 255, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return Renderer::Pixel{0, 255, b, 255};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return Renderer::Pixel{r, 255, 0, 255};
  }
  else if (x < 1.f)
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return Renderer::Pixel{255, b, 0, 255};
  }
  else
  {
    return Renderer::Pixel{0, 0, 0, 255};
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
Mandelbrot2D::cpu_naive_mandel_2d(Pixel* pix, float size)
{
    std::uint8_t* iters = new std::uint8_t[width * height];
    cpu_naive_mandel_iter(iters, width, height, 255, size);
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            pix[i * width + j] = heat_lut(iters[i * width + j] / 255.f);
    delete iters;
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
cuda_compute_LUT(const std::uint32_t* hist, int max_iter, Renderer::Pixel* LUT)
{
    std::uint32_t partial_sum[256];

    for (int i = 0; i <= max_iter; ++i)
    {
        partial_sum[i] = hist[i];
        if (i > 0)
            partial_sum[i] += partial_sum[i-1];
    }

    for (int i = 0; i <= max_iter; ++i)
        LUT[i] = heat_lut((float)partial_sum[i] / partial_sum[max_iter]);
}



__global__ void
cuda_apply_LUT(Renderer::Pixel* colors, int width, int height, int max_iter,
    std::uint8_t* iters, const Renderer::Pixel* LUT)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    colors[y * width + x] = LUT[iters[y * width + x]];
}

void
Mandelbrot2D::cuda_naive_mandel_2d(Pixel* hostBuffer, float size)
{
    std::uint8_t* iters;
    Pixel* colors;
    std::uint32_t* hist;
    Pixel* LUT;

    checkCudaErrors(cudaMalloc(&iters, width * sizeof(std::uint8_t) * height));
    checkCudaErrors(cudaMalloc(&colors, width * sizeof(Pixel) * height));
    checkCudaErrors(cudaMalloc(&hist, sizeof(std::uint32_t) * (max_iter + 1)));
    checkCudaErrors(cudaMalloc(&LUT, sizeof(Pixel) * (max_iter + 1)));

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
        cuda_apply_LUT<<<dimGrid, dimBlock>>>(colors, width, height, max_iter, iters, LUT);
    }

    // Copy back to main memory
    checkCudaErrors(cudaMemcpy(hostBuffer, colors, width * sizeof(Pixel) * height, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(LUT));
    checkCudaErrors(cudaFree(hist));
    checkCudaErrors(cudaFree(iters));
    checkCudaErrors(cudaFree(colors));
}


Mandelbrot2D::Mandelbrot2D(int w, int h, bool gpu_enabled)
    : width(w), height(h), gpu(gpu_enabled)
{
    max_iter = (1 << sizeof(std::uint8_t) * CHAR_BIT) - 1;
    std::cout << "gpu enabled: " << (gpu ? "yes" : "no") << std::endl;
}

Mandelbrot2D::~Mandelbrot2D()
{
}

void Mandelbrot2D::render(Pixel* target, const mat4& cam)
{
    if (gpu)
    {
        cuda_naive_mandel_2d(target, cam[0].w * -2);
    }
    else
    {
        cpu_naive_mandel_2d(target, cam[0].w * -2);
    }
}
