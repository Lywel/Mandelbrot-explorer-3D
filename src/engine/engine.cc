#include "engine.h"

Engine::Engine(int width, int height)
{
    pixels = new Renderer::Pixel[width * height];
        glm::vec3 direction;

        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        forward = normalize(direction);
        cam = glm::lookAt(position, position + forward, up);
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

#include <glm/gtx/string_cast.hpp>
void
Engine::auto_move(int elapsed_ms)
{
    time += elapsed_ms / 1000.f / 5;

// camera
	float di = 1.4 + 0.1 * cos(.29 * time);
	vec3  ro = di * vec3( cos(.33*time), 0.8*sin(.37*time), sin(.31*time) );
	vec3  ta = vec3(0.0,0.1,0.0);
	float cr = 0.5*cos(0.1*time);

    // camera matrix
	vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cw = normalize(ta-ro);
	vec3 cu = normalize(cross(cw,cp));
	vec3 cv =          (cross(cu,cw));

    /* std::cout << "di (distance): " << std::to_string(di) << std::endl; */
    /* std::cout << "ro (pos?)       : " << to_string(ro) << std::endl; */
    /* std::cout << "ta (target?)       : " << to_string(ta) << std::endl; */
    /* std::cout << "cr (?)       : " << std::to_string(cr) << std::endl << std::endl; */
    /* std::cout << "cp (?)       : " << to_string(cp) << std::endl; */
    /* std::cout << "cw (dir?)       : " << to_string(cw) << std::endl; */
    /* std::cout << "cu (?)       : " << to_string(cu) << std::endl; */
    /* std::cout << "cv (?)       : " << to_string(cv) << std::endl << std::endl << std::endl; */
    //cam = lookAt(ro, cw, up);
    //
    // cam = transpose(mat4(cu, ro.x, cv, ro.y, cw, ro.z, 0.0, 0.0, 0.0, 1.0 ));
    cam[0] = vec4(cu, ro.x);
    cam[1] = vec4(cv, ro.y);
    cam[2] = vec4(cw, ro.z);
    cam[3] = vec4(0, 0, 0, 1);
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
        gui->infos << "(" <<  cam[3].x << ", " << cam[3].y << ", " << cam[3].z << ")" << std::endl;
        gui->render();

        int mx = 0, my = 0;
        ev = gui->get_events(&mx, &my);
        /* handle_event(ev, mx, my); */
        auto_move(elapsed.count());

        last_frame = current_frame;
    }
}
