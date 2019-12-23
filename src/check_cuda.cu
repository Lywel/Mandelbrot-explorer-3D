#include "check_cuda.h"

bool
gpu_available(bool debug)
{
    int dev_nb;

    cudaError_t err = cudaGetDeviceCount(&dev_nb);

    if (dev_nb < 1 || err)
    {
        if (debug)
        {
            std::cout << cudaGetErrorName(err)
                << ": " << cudaGetErrorString(err) << std::endl;
        }
        return false;
    }

    if (debug)
    {
        for (int i = 0; i < dev_nb; ++i)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            std::cout << "Running on CUDA capable device " << i
                << ": " << prop.name << std::endl;
            std::cout << "\tMemory Clock Rate (KHz): "
                << prop.memoryClockRate << std::endl;
            std::cout << "\tMemory Bus Width (bits): "
                << prop.memoryBusWidth << std::endl;
            std::cout << "\tPeak Memory Bandwidth (GB/s): "
                << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6
                << std::endl << std::endl;
        }
    }
    return true;
}
