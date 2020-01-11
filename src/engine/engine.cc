#include "engine.h"

Engine::Engine(int width, int height)
{
    pixels = new Renderer::Pixel[width * height];
        glm::vec3 direction;

        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        forward = normalize(direction);
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
    default:
        break;
    }

    if (mouse_x > 1 || mouse_y > 1)
    {
        yaw += mouse_x * sensivity;
        pitch += mouse_y * sensivity;

        if (pitch > 89.0f)
            pitch =  89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        glm::vec3 direction;

        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        forward = normalize(direction);
    }
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

        auto cam = glm::lookAt(position, position + forward, up);
        renderer->render(pixels, cam);
        gui->set_pixels(pixels);
        gui->infos << "(" <<  cam[3].x << ", " << cam[3].y << ", " << cam[3].z << ")" << std::endl;
        gui->render();

        int mx = 0, my = 0;
        ev = gui->get_events(&mx, &my);
        handle_event(ev, mx, my);

        last_frame = current_frame;
    }
}
