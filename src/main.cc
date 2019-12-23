#include "window.h"
#include "mandel_2d.h"
#include "color.h"
#include "check_cuda.h"

#include <iostream>
#include <SDL.h>

int
main(int argc, char** argv)
{
    bool cuda_enabled = gpu_available(true);

    /* if (!cuda_enabled) */
    /*     exit(1); */

    int w = 1920;
    int h = 1080;

    Window win(w, h, "SDL window");

    rgba8_t pix[w * h];
    float size = 10.0f;

    while (!win.input_pool())
    {
        /* if (cuda_enabled) */
        /*     cuda_naive_mandel_2d(pix, w, h, size); */
        /* else */
        /*     cpu_naive_mandel_2d(pix, w, h, size); */

        size *= 0.9975;

        win.display_stat("zoom", 1/size*10);
        win.render(pix);
    }

    return 0;
}
