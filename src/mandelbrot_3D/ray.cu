#include "ray.h"

__host__ __device__
glm::vec2
isphere(glm::vec4 sph, glm::vec3 ro, glm::vec3 rd)
{
    vec3 oc = ro - vec3(sph.x, sph.y, sph.z);

    float b = dot(oc,rd);
    float c = dot(oc,oc) - sph.w*sph.w;
    float h = b*b - c;

    if( h<0.0 ) return vec2(-1.0);

    h = sqrt( h );

    return -b + vec2(-h,h);
}

__host__ __device__
inline float radians(float x)
{
    return x / 180 * M_PI;
}


__host__ __device__
float mandel_SDF(const vec3& sample, glm::vec4& color)
{
    vec3 w = sample;
    float m = glm::dot(w, w);

    glm::vec4 trap = glm::vec4(glm::abs(w), m);

    float dz = 1.0f;

    float q = 8;

    for (int i = 0; i < 4; ++i)
    {
        dz = q*pow(sqrt(m), q-1)*dz + 1.0;

        float r = glm::length(w);
        float b = q * acos(w.y / r);
        float a = q * atan2(w.x, w.z);
        w = sample + powf(r, q) * vec3(sin(b) * sin(a), cos(b), sin(b) * cos(a));

        trap = glm::min(trap, glm::vec4(glm::abs(w), m));

        m = glm::dot(w, w);
        if( m > 256.0 )
            break;
    }

    color = glm::vec4(m, trap.y, trap.z, trap.w);

    return 0.25*log(m)*sqrt(m)/dz;
}

__host__ __device__
float sphere_SDF(const vec3& sample, float radius)
{
    return glm::length(sample) - radius;
}

__host__ __device__
float scene_SDF(const vec3& sample, glm::vec4& color)
{
    //return sphere_SDF(sample, 1);
    return mandel_SDF(sample, color);
    //return compute_iter(sample, 4, max_dist, 2);
}

__host__ __device__
float scene_SDF(const vec3& sample)
{
    glm::vec4 o;
    return scene_SDF(sample, o);
}

__host__ __device__
vec3 compute_normal(const vec3& pos, float px)
{
    float e = 0.5773 * 0.25 * px;

    vec3 a(e, -e, -e);
    vec3 b(-e, -e, e);
    vec3 c(-e, e, -e);
    vec3 d(e, e, e);

    return glm::normalize(a * scene_SDF(pos + a) +
            b * scene_SDF(pos + b) +
            c * scene_SDF(pos + c) +
            d * scene_SDF(pos + d));
}

/* float dist_to_surface(const vec3& eye, const vec3& dir) */
/* { */
/*     float depth = MIN_DIST; */

/*     for (int i = 0; i < max_steps; ++i) */
/*     { */
/*         float dist = scene_SDF(eye + depth * dir); */
/*         if (dist < epsilon) */
/*             return dist; */

/*         depth += dist; */
/*         if (depth >= max_dist) */
/*             return max_dist; */
/*     } */

/*     return max_dist; */
/* } */

__host__ __device__
float intersect(glm::vec3 ro, glm::vec3 rd, float px, glm::vec4& color)
{
    float res = -1.0;

    // bounding sphere
    vec2 dis = isphere( vec4(0.0,0.0,0.0,10), ro, rd );
    if( dis.y<0.0 )
        return -1.0;
    dis.x = max( dis.x, 0.0f );
    dis.y = min( dis.y, 10.0f );

    // raymarch fractal distance field
    glm::vec4 trap(0);

    float t = dis.x;
    for (int i=0; i < 64; i++)
    {
        vec3 pos = ro + rd * t;
        float th = 0.25 * px * t;
        float h = scene_SDF(pos, trap);
        if (t > dis.y || h < th)
            break;
        t += h;
    }


    if (t < dis.y )
    {
        color = trap;
        res = t;
    }

    return res;
}

__host__ __device__
vec3 compute_ray_dir(float fov, int width, int height, float px, float py)
{
    float x = px - width / 2.0f;
    float y = py - height / 2.0f;
    float z = height / tan(radians(fov) / 2.0f);

    return glm::normalize(vec3(x, y, -z));
}

__host__ __device__
float softshadow(vec3 ro, vec3 rd, float k)
{
    float res = 1.0;
    float t = 0.0;
    for (int i=0; i<64; i++)
    {
        float h = scene_SDF(ro + rd * t);
        res = min(res, k * h / t);
        if (res < 0.001)
            break;
        t += clamp(h, 0.01f, 0.2f);
    }
    return clamp(res, 0.0f, 1.0f);
}
