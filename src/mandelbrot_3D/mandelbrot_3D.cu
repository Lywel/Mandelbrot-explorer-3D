#include "mandelbrot_3D.h"

float
compute_iter(const glm::vec3& pos, int max_iter, float max_val, int exponent)
{
    glm::vec3 z = pos;
    float dr = 1.f;
    float r = 0.f;
    for (int i = 0; i < max_iter; i++)
    {
        r = z.length();
        if (r > max_val) break;

        // convert to polar coordinates
        float theta = acosf(z.z / r);
        float phi = atan2f(z.y, z.x);
        dr =  powf(r, exponent - 1.f) * exponent * dr + 1.f;

        // scale and rotate the point
        float zr = powf(r, exponent);
        theta = theta * exponent;
        phi = phi * exponent;

        // convert back to cartesian coordinates
        z = zr * glm::vec3(sinf(theta) * cosf(phi), sinf(phi) * sinf(theta), cosf(theta));
        z = z + pos;
    }
    return 0.5 * log(r) * r / dr;
}

__host__ __device__
glm::vec2
isphere(glm::vec4 sph, glm::vec3 ro, glm::vec3 rd)
{
    glm::vec3 oc = ro - glm::vec3(sph.x, sph.y, sph.z);

    float b = glm::dot(oc, rd);
    float c = glm::dot(oc,oc) - sph.w * sph.w;
    float h = b * b - c;

    if (h < 0.f )
      return glm::vec2(-1.f);

    h = sqrtf(h);

    return -b + glm::vec2(-h, h);
}

__host__ __device__
float mandel_SDF(const glm::vec3& sample, glm::vec4& color)
{
    glm::vec3 w = sample;
    float m = glm::dot(w, w);

    glm::vec4 trap = glm::vec4(glm::abs(w), m);

    float dz = 1.0f;

    for (int i = 0; i < 3; ++i)
    {
        float m2 = m*m;
        float m4 = m2*m2;
        dz = 8.f * sqrtf(m4*m2*m) * dz + 1.f;

        float x = w.x; float x2 = x*x; float x4 = x2*x2;
        float y = w.y; float y2 = y*y; float y4 = y2*y2;
        float z = w.z; float z2 = z*z; float z4 = z2*z2;

        float k3 = x2 + z2;
        float k2 = rsqrtf( k3*k3*k3*k3*k3*k3*k3 );
        float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
        float k4 = x2 - y2 + z2;

        w.x = sample.x +  64.0*x*y*z*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2;
        w.y = sample.y + -16.0*y2*k3*k4*k4 + k1*k1;
        w.z = sample.z +  -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2;

        trap = glm::min(trap, glm::vec4(glm::abs(w), m));

        m = glm::dot(w, w);
        if( m > 256.f )
            break;
    }

    color = glm::vec4(m, trap.y, trap.z, trap.w);

    return 0.25f * logf(m) * sqrtf(m) / dz;
}

__host__ __device__
float sphere_SDF(const glm::vec3& sample, float radius)
{
    return glm::length(sample) - radius;
}

__host__ __device__
float scene_SDF(const glm::vec3& sample, glm::vec4& color)
{
    //return sphere_SDF(sample, 1);
    return mandel_SDF(sample, color);
    //return compute_iter(sample, 4, max_dist, 2);
}

__host__ __device__
float scene_SDF(const glm::vec3& sample)
{
    glm::vec4 o;
    return scene_SDF(sample, o);
}

__host__ __device__
glm::vec3 compute_normal(const glm::vec3& pos, float px)
{
    float e = 0.5773f * 0.25f * px;

    glm::vec3 a(e, -e, -e);
    glm::vec3 b(-e, -e, e);
    glm::vec3 c(-e, e, -e);
    glm::vec3 d(e, e, e);

    return glm::normalize(a * scene_SDF(pos + a) +
            b * scene_SDF(pos + b) +
            c * scene_SDF(pos + c) +
            d * scene_SDF(pos + d));
}

/* float dist_to_surface(const glm::vec3& eye, const glm::vec3& dir) */
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
float intersect(glm::vec3 ro, glm::vec3 rd, float px, glm::vec4* color)
{
    float res = -1.f;

    // bounding sphere
    glm::vec2 dis = isphere( glm::vec4(0.f, 0.f, 0.f, 2.f), ro, rd );
    if (dis.y < 0.0f)
        return -1.f;

    dis.x = fmaxf(dis.x, 0.0f );
    dis.y = fminf(dis.y, 10.0f );

    // raymarch fractal distance field
    glm::vec4 trap(0.f);

    float t = dis.x;
    for (int i=0; i < 128; i++)
    {
        glm::vec3 pos = ro + rd * t;
        float th = 0.25f * px * t;
        float h = scene_SDF(pos, trap);
        if (t > dis.y || h < th)
            break;
        t += h;
    }


    if (t < dis.y)
    {
        *color = trap;
        res = t;
    }

    return res;
}

__host__ __device__
glm::vec3 compute_ray_dir(float fov, int width, int height, float px, float py)
{
    float x = px - width / 2.0f;
    float y = py - height / 2.0f;
    float z = height / tanf(radians(fov) / 2.0f);

    return glm::normalize(glm::vec3(x, y, -z));
}

__host__ __device__
float softshadow(glm::vec3 ro, glm::vec3 rd, float k)
{
    float res = 1.f;
    float t = 0.f;
    for (int i=0; i < 64; i++)
    {
        float h = scene_SDF(ro + rd * t);
        res = min(res, k * h / t);
        if (res < 0.001f)
            break;
        t += clamp(h, 0.01f, 0.2f);
    }
    return clamp(res, 0.0f, 1.0f);
}

__device__ __host__
glm::vec3
render(const glm::vec2& p, const glm::vec2& resolution, const glm::mat4 cam)
{
    const glm::vec3 light1 = glm::vec3(0.577f, 0.577f, -0.577f);
    const glm::vec3 light2 = glm::vec3(-0.707f, 0.000f, 0.707f);

    // ray setup
    const float fle = 1.5f;

    // Convert pixel to ndc space (-1, 1)
    glm::vec2 sp = (2.f * p - resolution) / resolution.y;
    float px = 2.f / (resolution.y * fle);

    // <WORKING
    /* glm::mat4 proj = perspective(radians(50.f), 1.f, .1f, 100.f); */
    /* glm::vec4 viewport = glm::vec4(0, 0, resolution); */

    /* glm::vec3 wp = unProject(glm::vec3(p, 0), cam, proj, viewport); */
    /* glm::vec3 ro = glm::vec3(-cam[3].x, -cam[3].y, -cam[3].z); */
    /* glm::vec3 rd = glm::normalize(wp - ro); */
    // WORKING>

    glm::vec3  ro = glm::vec3( cam[0].w, cam[1].w, cam[2].w );
    // doing this:  rd = normalize((cam * glm::vec4(sp, fle, 0.0)).xyz);
    glm::vec4 rd4 = cam * glm::vec4(sp, fle, 0.f);
    glm::vec3 rd = normalize(glm::vec3(rd4.x, rd4.y, rd4.z));

    //glm::vec3 rd = compute_ray_dir(50.0f, resolution.x, resolution.y, p.x, p.y);

    glm::vec4 tra = glm::vec4(0.f);
    float dist = intersect(ro, rd, px, &tra);


    // Coloring
    glm::vec3 col = glm::vec3(0.0f);

    // color sky
    if (dist < 0.f || dist >= 10.f)
    {
        col = glm::vec3(0.8f, .9f, 1.1f) * (0.6f + 0.4f * rd.y);
        col += 5.0f * glm::vec3(0.8f, 0.7f, 0.5f) * powf(clamp(dot(rd, light1), 0.0f, 1.0f), 32.f);
    }
    // color fractal
    else
    {
        // color
        col = glm::vec3(0.01);

        //col = mix(col, glm::vec3(0.10,0.20,0.30), clamp(tra.y, 0.f, 1.f) );
        //col = mix(col, glm::vec3(0.02,0.10,0.30), clamp(tra.z*tra.z, 0.f, 1.f) );
        //col = mix(col, glm::vec3(0.30,0.10,0.02), clamp(powf(tra.w,6.0), 0.f, 1.f) );
        col = mix(col, glm::vec3(0.2f, 0.01f, 0.01f), clamp(tra.y, 0.f, 1.f) );
        col = mix(col, glm::vec3(0.01f, 0.2f, 0.01f), clamp(tra.z*tra.z, 0.f, 1.f) );
        col = mix(col, glm::vec3(0.01f, 0.01f, 0.2f), clamp(powf(tra.w, 6.f), 0.f, 1.f) );
        col *= 0.5;

        // lighting terms
        glm::vec3 hit = ro + dist * rd;
        glm::vec3 nor = compute_normal(hit, px);
        glm::vec3 hal = glm::normalize(light1 - rd);
        float occ = clamp(0.05f * log(tra.x), 0.0f, 1.0f);
        float fac = clamp(1.f + dot(rd, nor), 0.0f, 1.0f);

        // sun
        float sha1 = softshadow(ro + 0.001f * nor, light1, 32.f);
        float dif1 = clamp(glm::dot(light1, nor), 0.0f, 1.0f) * sha1;
        float spe1 = powf(clamp(glm::dot(nor, hal), 0.0f, 1.0f), 32.f) * dif1
          * (0.04f + 0.96f * powf(clamp(1.f - glm::dot(hal, light1), 0.f, 1.f), 5.f));
        // bounce
        float dif2 = clamp(0.5f + 0.5f * glm::dot(light2, nor), 0.f, 1.f) * occ;
        // sky
        float dif3 = (0.7f + 0.3f * nor.y) * (0.2f + 0.8f * occ);

        glm::vec3 lin = glm::vec3(0.f, 0.f, 0.f);
             lin += 6.0f * glm::vec3(1.50f,1.10f,0.70f) * dif1;
             lin += 4.0f * glm::vec3(0.25f,0.20f,0.15f) * dif2;
             lin += 1.5f * glm::vec3(0.10f,0.20f,0.30f) * dif3;
             lin += 3.5f * glm::vec3(0.35f, 0.30f, 0.25f) * (0.05f + 0.95f * occ); // ambient
             lin += 4.0f * fac * occ;                          // fake SSS
        col *= lin;

        col = glm::pow(col, glm::vec3(0.7f, 0.9f, 1.0));
        col += spe1 * 5.0f;
    }

    col = glm::sqrt(col);

    col *= 1.f - 0.05f * glm::length(sp);

    return col;
}

__global__ void
cuda_render_px(char* colors, int width, int height, size_t pitch, const glm::mat4 cam)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    glm::vec3 col = ::render(glm::vec2(x, y), glm::vec2(width, height), cam);
    Renderer::Pixel* lineptr = (Renderer::Pixel*)(colors + y * pitch);


    lineptr[x] = Renderer::Pixel{col.x * 255.f, col.y * 255.f, col.z * 255.f, 255};
}

void
Mandelbrot3D::cuda_dinit()
{
    checkCudaErrors(cudaFree(colors));
    cudaDeviceReset();
}

void
Mandelbrot3D::cuda_init()
{
    checkCudaErrors(cudaMallocPitch(&colors, &pitch, width * sizeof(Pixel), height));
}

void
Mandelbrot3D::cuda_render(Pixel* target, const glm::mat4& cam)
{
    if (cam == last_cam)
        return;
    last_cam = cam;


    {
        int bsize = 16;
        int w     = std::ceil((float)width / bsize);
        int h     = std::ceil((float)height / bsize);

        dim3 dimBlock(bsize, bsize);
        dim3 dimGrid(w, h);

        cuda_render_px<<<dimGrid, dimBlock>>>(colors, width, height, pitch, cam);

        checkCudaErrors(cudaMemcpy2D(
              target, width * sizeof(Pixel),
              colors, pitch, width * sizeof(Pixel),
              height, cudaMemcpyDeviceToHost));
    }
}
