#include "mandelbrot_3D.cu.h"
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


__device__ __host__
vec3
render(const vec2& p, const vec2& resolution, const mat4& cam)
{
    const vec3 light1 = vec3(0.577, 0.577, -0.577);
    const vec3 light2 = vec3(-0.707, 0.000, 0.707);

    // ray setup
    const float fle = 1.2;

    // Convert pixel to ndc space (-1, 1)
    vec2 sp = (2.f * p - resolution) / resolution.y;
    float px = 2.f / (resolution.y * fle);

    // <WORKING
    /* mat4 proj = perspective(radians(50.f), 1.f, .1f, 100.f); */
    /* vec4 viewport = vec4(0, 0, resolution); */

    /* vec3 wp = unProject(vec3(p, 0), cam, proj, viewport); */
    /* vec3 ro = vec3(-cam[3].x, -cam[3].y, -cam[3].z); */
    /* vec3 rd = glm::normalize(wp - ro); */
    // WORKING>

    vec3  ro = vec3( cam[0].w, cam[1].w, cam[2].w );
    // doing this:  rd = normalize((cam * vec4(sp, fle, 0.0)).xyz);
    vec4 rd4 = cam * vec4(sp, fle, 0.0);
    vec3 rd = normalize(vec3(rd4.x, rd4.y, rd4.z));

    //vec3 rd = compute_ray_dir(50.0f, resolution.x, resolution.y, p.x, p.y);

    vec4 tra;
    float dist = intersect(ro, rd, px, tra);

    // Coloring
    vec3 col = vec3(0);

    // color sky
    if (dist < 0.0 || dist >= 10.0)
    {
        col = vec3(0.8, .9, 1.1) * (0.6f + 0.4f * rd.y);
        col += 5.0f * vec3(0.8, 0.7, 0.5) * powf(clamp(dot(rd, light1), 0.0f, 1.0f), 32.0);
    }
    // color fractal
    else
    {
        // color
        col = vec3(0.01);

        col = mix(col, vec3(0.10,0.20,0.30), clamp(tra.y, 0.f, 1.f) );
        col = mix(col, vec3(0.02,0.10,0.30), clamp(tra.z*tra.z, 0.f, 1.f) );
        col = mix(col, vec3(0.30,0.10,0.02), clamp(powf(tra.w,6.0), 0.f, 1.f) );
        col *= 0.5;

        // lighting terms
        vec3 hit = ro + dist * rd;
        vec3 nor = compute_normal(hit, px);
        vec3 hal = normalize(light1 - rd);
        //vec3 ref = reflect(rd, nor);
        float occ = clamp(0.05 * log(tra.x), 0.0, 1.0);
        float fac = clamp(1.0 + dot(rd, nor), 0.0, 1.0);

        // sun
        float sha1 = softshadow(ro + 0.001f * nor, light1, 32.0 );
        float dif1 = clamp(dot(light1, nor), 0.0f, 1.0f) * sha1;
        float spe1 = powf(clamp(dot(nor, hal), 0.0f, 1.0f), 32.0) * dif1 * (0.04 + 0.96 * pow(clamp(1.0 - dot(hal, light1), 0.0, 1.0), 5.0));
        // bounce
        float dif2 = clamp( 0.5 + 0.5* dot( light2, nor ), 0.0, 1.0 )*occ;
        // sky
        float dif3 = (0.7+0.3*nor.y)*(0.2+0.8*occ);

        vec3 lin = vec3(0);
        lin += 7.0f * vec3(1.50,1.10,0.70) * dif1;
        lin += 4.0f * vec3(0.25,0.20,0.15)*dif2;
        lin += 1.5f * vec3(0.10,0.20,0.30)*dif3;
        lin += 2.5f * vec3(0.35, 0.30, 0.25) * (0.05f + 0.95f * occ); // ambient
        lin += 4.0 * fac * occ;                          // fake SSS
        col *= lin;
        col += spe1 * 15.0f;
    }

    col = sqrt(col);

    return col;
}

__global__ void
cuda_render_px(Renderer::Pixel* colors, int width, int height, const mat4& cam)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    vec3 col = ::render(vec2(x, y), vec2(width, height), cam);
    colors[y * width + x] = Renderer::Pixel{col.x * 255, col.y * 255, col.z * 255, 255};
}

void
Mandelbrot3D::cuda_render(Pixel* target, const mat4& cam)
{
    if (cam == last_cam)
        return;
    last_cam = cam;

    Pixel* colors;
    checkCudaErrors(cudaMalloc(&colors, width * sizeof(Pixel) * height));

    {
        int bsize = 4;
        int w     = std::ceil((float)width / bsize);
        int h     = std::ceil((float)height / bsize);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);

        cuda_render_px<<<dimGrid, dimBlock>>>(colors, width, height, cam);

        // Copy back to main memory
        checkCudaErrors(cudaMemcpy(target, colors, width * sizeof(Pixel) * height, cudaMemcpyDeviceToHost));
    }

    // Free
    checkCudaErrors(cudaFree(colors));
}
