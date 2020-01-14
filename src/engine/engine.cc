#include "engine.h"

Engine::Engine(int width, int height, bool autoplay)
    : running(autoplay)
{
    pixels = new Renderer::Pixel[width * height];
    auto_move(0);
    /* glm::vec3 direction; */

    /* direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch)); */
    /* direction.y = sin(glm::radians(pitch)); */
    /* direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch)); */
    /* forward = normalize(direction); */
    /* cam = glm::lookAt(position, position + forward, up); */
}

Engine::~Engine()
{
    delete pixels;
}

float sensivity = -0.005f;

void
Engine::handle_event(const GUI::Event ev, int mouse_x, int mouse_y) {
    switch (ev)
    {
    case GUI::Event::Forward:
        position += speed * forward;
        break;
    case GUI::Event::Backward:
        position -= speed * forward;
        break;
    case GUI::Event::Left:
        position -= normalize(cross(forward, up)) * speed;
        break;
    case GUI::Event::Right:
        position += normalize(cross(forward, up)) * speed;
        break;
    case GUI::Event::ToggleAnimate:
        running = !running;
        if (running)

        break;
    default:
        break;
    }

    /* if (mouse_x > 1 || mouse_y > 1) */
    /* { */
    /*     yaw += mouse_x * sensivity; */
    /*     pitch += mouse_y * sensivity; */

    /*     if (pitch > 89.0f) */
    /*         pitch =  89.0f; */
    /*     if (pitch < -89.0f) */
    /*         pitch = -89.0f; */

    /*     glm::vec3 direction; */

    /*     direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch)); */
    /*     direction.y = sin(glm::radians(pitch)); */
    /*     direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch)); */
    /*     forward = normalize(direction); */
    /* } */
}

#include <glm/gtx/string_cast.hpp>
void
Engine::auto_move(int elapsed_ms)
{
    time += elapsed_ms / 1000.f;

// camera
    float di = 1.4 + 0.1 * cos(.29 * time) + 1.5;
    glm::vec3  ro = di * glm::vec3( cos(.33*time), 0.8*sin(.37*time), sin(.31*time) );
    glm::vec3  ta = glm::vec3(0.0,0.1,0.0);
    float cr = 0.5*cos(0.1*time);

    // camera glm::matrix
    glm::vec3 cp = glm::vec3(sin(cr), cos(cr),0.0);
    glm::vec3 cw = normalize(ta-ro);
    glm::vec3 cu = normalize(cross(cw,cp));
    glm::vec3 cv =          (cross(cu,cw));

    cam[0] = glm::vec4(cu, ro.x);
    cam[1] = glm::vec4(cv, ro.y);
    cam[2] = glm::vec4(cw, ro.z);
    cam[3] = glm::vec4(0, 0, 0, 1);
}

void
Engine::run(GUI* gui, Renderer* renderer)
{
    std::chrono::high_resolution_clock::time_point current_frame{
        std::chrono::high_resolution_clock::now()};

    GUI::Event ev = GUI::Event::None;
    while (ev != GUI::Event::Quit)
    {
        current_frame = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_frame - last_frame);

        renderer->render(pixels, cam);
        gui->set_pixels(pixels);

        gui->infos << std::setprecision(2) << "(" <<  cam[0].w << ", " << cam[1].w << ", " << cam[2].w << ")" << std::endl;

        gui->render();

        int mx = 0, my = 0;
        ev = gui->get_events(&mx, &my);
        handle_event(ev, mx, my);

        if (running)
            auto_move(elapsed.count());

        last_frame = current_frame;
    }
}
