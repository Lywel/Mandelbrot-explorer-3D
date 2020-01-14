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
float mandel_SDF(const vec3& sample, glm::vec4& color)
{
    vec3 w = sample;
    float m = glm::dot(w, w);

    glm::vec4 trap = glm::vec4(glm::abs(w), m);

    float dz = 1.0f;

    float q = 8;

    for (int i = 0; i < 3; ++i)
    {
# if 1
        float m2 = m*m;
        float m4 = m2*m2;
        dz = 8.0*sqrt(m4*m2*m)*dz + 1.0;

        float x = w.x; float x2 = x*x; float x4 = x2*x2;
        float y = w.y; float y2 = y*y; float y4 = y2*y2;
        float z = w.z; float z2 = z*z; float z4 = z2*z2;

        float k3 = x2 + z2;
        float k2 = inversesqrt( k3*k3*k3*k3*k3*k3*k3 );
        float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
        float k4 = x2 - y2 + z2;

        w.x = sample.x +  64.0*x*y*z*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2;
        w.y = sample.y + -16.0*y2*k3*k4*k4 + k1*k1;
        w.z = sample.z +  -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2;
#else
        dz = q*pow(sqrt(m), q-1)*dz + 1.0;

        float r = glm::length(w);
        float b = q * acos(w.y / r);
        float a = q * atan2(w.x, w.z);
        w = sample + powf(r, q) * vec3(sin(b) * sin(a), cos(b), sin(b) * cos(a));
#endif

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
float intersect(glm::vec3 ro, glm::vec3 rd, float px, glm::vec4* color)
{
    float res = -1.0;

    // bounding sphere
    vec2 dis = isphere( vec4(0.0,0.0,0.0, 2), ro, rd );
    if( dis.y<0.0 )
        return -1.0;
    dis.x = max( dis.x, 0.0f );
    dis.y = min( dis.y, 10.0f );

    // raymarch fractal distance field
    glm::vec4 trap(0);

    float t = dis.x;
    for (int i=0; i < 128; i++)
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
        *color = trap;
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
    for (int i=0; i < 64; i++)
    {
        float h = scene_SDF(ro + rd * t);
        res = min(res, k * h / t);
        if (res < 0.001)
            break;
        t += clamp(h, 0.01f, 0.2f);
    }
    return clamp(res, 0.0f, 1.0f);
}

__device__ __host__
vec3
render(const vec2& p, const vec2& resolution, const mat4 cam)
{
    const vec3 light1 = vec3(0.577, 0.577, -0.577);
    const vec3 light2 = vec3(-0.707, 0.000, 0.707);

    // ray setup
    const float fle = 1.5;

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

    vec4 tra = vec4(0);
    float dist = intersect(ro, rd, px, &tra);

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

        //col = mix(col, vec3(0.10,0.20,0.30), clamp(tra.y, 0.f, 1.f) );
        //col = mix(col, vec3(0.02,0.10,0.30), clamp(tra.z*tra.z, 0.f, 1.f) );
        //col = mix(col, vec3(0.30,0.10,0.02), clamp(powf(tra.w,6.0), 0.f, 1.f) );
        col = mix(col, vec3(0.2,0.01,0.01), clamp(tra.y, 0.f, 1.f) );
        col = mix(col, vec3(0.01,0.2,0.01), clamp(tra.z*tra.z, 0.f, 1.f) );
        col = mix(col, vec3(0.01,0.01,0.2), clamp(powf(tra.w, 6.0), 0.f, 1.f) );
        col *= 0.5;

        // lighting terms
        vec3 hit = ro + dist * rd;
        vec3 nor = compute_normal(hit, px);
        vec3 hal = normalize(light1 - rd);
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

        vec3 lin = vec3(0, 0, 0);
             lin += 6.0f * vec3(1.50,1.10,0.70) * dif1;
             lin += 4.0f * vec3(0.25,0.20,0.15) * dif2;
             lin += 1.5f * vec3(0.10,0.20,0.30) * dif3;
             lin += 3.5f * vec3(0.35, 0.30, 0.25) * (0.05f + 0.95f * occ); // ambient
             lin += 4.0 * fac * occ;                          // fake SSS
        col *= lin;

        col = pow( col, vec3(0.7,0.9,1.0) );
        col += spe1 * 5.0f;
    }

    col = sqrt(col);

    col *= 1.0 - 0.05*length(sp);

    return col;
}

__global__ void
cuda_render_px(Renderer::Pixel* colors, int width, int height, const mat4 cam)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    vec3 col = ::render(vec2(x, y), vec2(width, height), cam);
    colors[y * width + x] = Renderer::Pixel{col.x * 255, col.y * 255, col.z * 255, 255};
}

__device__ vec3
compute_ray_dir(const mat4 cam, float fle, float px)
{
    /* vec4 rd4 = cam * vec4(sp, fle, 0.0); */
    /* return normalize(vec3(rd4.x, rd4.y, rd4.z)); */
}

__global__ void
compute_distance(Renderer::Pixel* out, vec2 dim, const mat4 cam)
{

}

void
Mandelbrot3D::cuda_dinit()
{
    checkCudaErrors(cudaFree(colors));
}

void
Mandelbrot3D::cuda_init()
{
    checkCudaErrors(cudaMalloc(&colors, width * sizeof(Pixel) * height));
}

void
Mandelbrot3D::cuda_render(Pixel* target, const mat4& cam)
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

        cuda_render_px<<<dimGrid, dimBlock>>>(colors, width, height, cam);

        // Copy back to main memory
        checkCudaErrors(cudaMemcpy(target, colors, width * sizeof(Pixel) * height, cudaMemcpyDeviceToHost));
    }
}
