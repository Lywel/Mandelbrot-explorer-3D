#include "check_cuda.h"
#include "window/window.h"
#include "mandelbrot_3D/mandelbrot_3D.h"
#include "mandelbrot_2D/mandelbrot.h"
#include "engine/engine.h"

#include <iostream>
#include <memory>
#include <SDL.h>
#include <CLI/CLI.hpp>

int
main(int argc, char** argv)
{
    // Config
    std::string mode = "3D";
    int width = 640;
    int height = 480;
    int niter = 100;
    bool gpu = false;

    CLI::App app{argv[0]};
    app.add_flag("--gpu,!--no-gpu", gpu, "Enable CUDA (if available)");
    app.add_set("-d", mode, {"3D", "2D"}, "Fractal type. Either '3D' or '2D'");
    app.add_option("niter", niter, "Number of iteration");
    app.add_option("width", width, "Width of the window");
    app.add_option("height", height, "Height of the window");

    CLI11_PARSE(app, argc, argv);

    bool cuda_enabled = gpu_available(true);

    Window gui (width, height, "SDL window");
    Engine engine(width, height);

    if (mode == "3D")
    {
        Mandelbrot3D renderer(width, height, cuda_enabled && gpu);
        engine.run(&gui, &renderer);
    }
    else
    {
        Mandelbrot2D renderer(width, height, cuda_enabled && gpu);
        engine.run(&gui, &renderer);
    }

    // Main loop

    return 0;
}
