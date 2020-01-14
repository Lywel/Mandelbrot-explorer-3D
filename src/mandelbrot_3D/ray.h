#pragma once

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/geometric.hpp>
#include <glm/matrix.hpp>

glm::vec2 isphere(glm::vec4 sph, glm::vec3 ro, glm::vec3 rd);

float mandel_SDF(const glm::vec3& sample, glm::vec4& color);
float sphere_SDF(const glm::vec3& sample, float radius);
float scene_SDF(const glm::vec3& sample, glm::vec4& color);
float scene_SDF(const glm::vec3& sample);

float softshadow(glm::vec3 ro, glm::vec3 rd, float k);

glm::vec3 compute_normal(const glm::vec3& pos, float px);
glm::vec3 compute_ray_dir(float fov, int width, int height, float px, float py);
float intersect(glm::vec3 ro, glm::vec3 rd, float px, glm::vec4* color);
float intersec(glm::vec3 ro, glm::vec3 rd, float px, glm::vec4* color);

/* float dist_to_surface(const glm::vec3& eye, const glm::vec3& dir); */
