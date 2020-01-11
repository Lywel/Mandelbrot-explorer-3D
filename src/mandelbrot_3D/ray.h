#pragma once

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/geometric.hpp>
#include <glm/matrix.hpp>

using namespace glm;

vec2 isphere(vec4 sph, vec3 ro, vec3 rd);

float mandel_SDF(const vec3& sample, vec4& color);
float sphere_SDF(const vec3& sample, float radius);
float scene_SDF(const vec3& sample, vec4& color);
float scene_SDF(const vec3& sample);

float softshadow(vec3 ro, vec3 rd, float k);

vec3 compute_normal(const vec3& pos, float px);
vec3 compute_ray_dir(float fov, int width, int height, float px, float py);
float intersect(vec3 ro, vec3 rd, float px, vec4& color);

/* float dist_to_surface(const vec3& eye, const vec3& dir); */
