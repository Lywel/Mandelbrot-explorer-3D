#include "check_cuda.h"
#include "window/window.h"
#include "mandelbrot_3D/mandelbrot_3D.h"
#include "engine/engine.h"

#include <iostream>
#include <memory>
#include <SDL.h>

int
main(int argc, char** argv)
{
    bool cuda_enabled = gpu_available(true);

    /* if (!cuda_enabled) */
    /*     exit(1); */

    int w = 180;
    int h = 180;

    /* Window win(w, h, "SDL window"); */

    /* rgba8_t pix[w * h]; */
    /* float size = 8.0f; */

    /* while (!win.input_pool()) */
    /* { */
    /*     if (cuda_enabled) */
    /*         cuda_naive_mandel_2d(pix, w, h, size); */
    /*     else */
    /*         cpu_naive_mandel_2d(pix, w, h, size); */

    /*     size *= 0.999; */

    /*     win.display_stat("zoom", 1/size*8.f); */
    /*     win.render(pix); */
    /* } */

    Window gui (w, h, "SDL window");
    Mandelbrot3D renderer(w, h);
    Engine engine(w , h);

    engine.run(&gui, &renderer);

    return 0;
}
