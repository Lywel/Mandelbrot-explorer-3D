#include "mandelbrot_3D.h"

float
compute_iter(const vec3& pos, int max_iter, float max_val, int exponent)
{
    vec3 z = pos;
    float dr = 1.0;
    float r = 0.0;
    for (int i = 0; i < max_iter; i++)
    {
        r = z.length();
        if (r > max_val) break;

        // convert to polar coordinates
        float theta = acos(z.z / r);
        float phi = atan2(z.y, z.x);
        dr =  pow(r, exponent - 1.0) * exponent * dr + 1.0;

        // scale and rotate the point
        float zr = pow(r, exponent);
        theta = theta * exponent;
        phi = phi * exponent;

        // convert back to cartesian coordinates
        z = zr * vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
        z = z + pos;
    }
    return 0.5 * log(r) * r / dr;
}

