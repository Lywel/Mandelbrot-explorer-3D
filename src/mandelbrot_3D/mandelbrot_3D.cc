#include "mandelbrot_3D.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>


Mandelbrot3D::Mandelbrot3D(int w, int h, int i)
    : width(w), height(h), max_iter(i)
{
}

void
Mandelbrot3D::render(Pixel* target, const mat4& cam)
{
    // cuda_naive_mandel_2d((rgba8_t*)target, width, height, pos.z);
    cpu_render(target, cam);
}


const vec3 light1(0.577, 0.577, -0.577);
const vec3 light2(-0.707, 0.000, 0.707);


vec3
render(const vec2& p, const vec2& resolution, const mat4& cam)
{
    // ray setup
    const float fle = 1.2;

    // Convert pixel to ndc space (-1, 1)
    vec2 sp = (2.f * p - resolution) / resolution.y;
    float px = 2.f / (resolution.y * fle);

    mat4 proj = perspective(radians(50.f), 1.f, .1f, 100.f);
    vec4 viewport(0, 0, resolution);


    vec3 wp = unProject(vec3(p, 0), cam, proj, viewport);

    mat4 invVP = inverse(proj * cam);
    vec4 screenPos = glm::vec4(sp.x, sp.y, 1.0f, 1.0f);
    vec4 worldPos = invVP * screenPos;

    vec3 ro = vec3(-cam[3].x, -cam[3].y, -cam[3].z);
    vec3 rd = glm::normalize(wp - ro);

    vec4 rd4 = cam * vec4(sp, fle, 0.0);
    //vec3 rd = normalize(vec3(rd4.x, rd4.y, rd4.z));



    //vec3 rd = compute_ray_dir(50.0f, resolution.x, resolution.y, p.x, p.y);

    vec4 tra;
    float dist = intersect(ro, rd, px, tra);

    // Coloring
    vec3 col(0);

    // color sky
    if (dist < 0.0 || dist >= 10.0)
    {
        col = vec3(0.8, .9, 1.1) * (0.6f + 0.4f * rd.y);
        col += 5.0f * vec3(0.8, 0.7, 0.5) * powf(std::clamp(glm::dot(rd, light1), 0.0f, 1.0f), 32.0);
    }
    // color fractal
    else
    {
        // color
        col = vec3(0.01);

        col = glm::mix( col, glm::vec3(0.10,0.20,0.30), glm::clamp(tra.y, 0.f, 1.f) );
        col = glm::mix( col, glm::vec3(0.02,0.10,0.30), glm::clamp(tra.z*tra.z, 0.f, 1.f) );
        col = glm::mix( col, glm::vec3(0.30,0.10,0.02), glm::clamp(powf(tra.w,6.0), 0.f, 1.f) );
        col *= 0.5;

        // lighting terms
        vec3 hit = ro + dist * rd;
        vec3 nor = compute_normal(hit, px);
        vec3 hal = glm::normalize(light1 - rd);
        //vec3 ref = reflect(rd, nor);
        float occ = std::clamp(0.05 * log(tra.x), 0.0, 1.0);
        float fac = std::clamp(1.0 + glm::dot(rd, nor), 0.0, 1.0);

        // sun
        float sha1 = softshadow(ro + 0.001f * nor, light1, 32.0 );
        float dif1 = std::clamp(glm::dot(light1, nor), 0.0f, 1.0f) * sha1;
        float spe1 = powf(std::clamp(glm::dot(nor, hal), 0.0f, 1.0f), 32.0) * dif1 * (0.04 + 0.96 * pow(std::clamp(1.0 - glm::dot(hal, light1), 0.0, 1.0), 5.0));
        // bounce
        float dif2 = std::clamp( 0.5 + 0.5* glm::dot( light2, nor ), 0.0, 1.0 )*occ;
        // sky
        float dif3 = (0.7+0.3*nor.y)*(0.2+0.8*occ);

        vec3 lin(0, 0, 0);
        lin += 7.0f * vec3(1.50,1.10,0.70) * dif1;
        lin += 4.0f * glm::vec3(0.25,0.20,0.15)*dif2;
        lin += 1.5f * glm::vec3(0.10,0.20,0.30)*dif3;
        lin += 2.5f * vec3(0.35, 0.30, 0.25) * (0.05f + 0.95f * occ); // ambient
        lin += 4.0 * fac * occ;                          // fake SSS
        col *= lin;
        col += spe1 * 15.0f;
    }

    col = sqrt(col);

    return col;
}

vec3 Mandelbrot3D::render_px(const vec2& px, const mat4& cam)
{
    return ::render(px, vec2(width, height), cam);
    // Antialiasing disabled below for performances
    /* vec3 col(0.0); */

    /* float aa = 2; */
    /* for (int j=0; j < aa; j++) */
    /* { */
    /*     for (int i=0; i < aa; i++) */
    /*     { */
    /*         glm::vec2 xy = glm::vec2(x, y) + (glm::vec2(i, j) / aa); */
    /*         col += ::render(xy.x, xy.y, width, height, pos, rot); */
    /*     } */
    /*     col /= aa * aa; */
    /* } */
}

void Mandelbrot3D::cpu_render(Pixel* target, const mat4& cam)
{
    if (cam == last_cam)
        return;
    last_cam = cam;

    std::cout << glm::to_string(cam) << std::endl;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            vec3 col = render_px(vec2(j, i), cam);
            target[i * width + j] = {col.x * 255, col.y * 255, col.z * 255, 255};
        }
    }
}
