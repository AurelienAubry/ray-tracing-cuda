#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"

class material {
    public:
        __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, curandState *local_rand_state) const = 0;

};

class lambertian : public material {

    public:
        __device__ lambertian(const color &a) : albedo(a) {}

        __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, curandState *local_rand_state) const override {
            vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);

            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = ray(rec.hit_point, scatter_direction);
            attenuation = albedo;
            return true;
        }

    public: 
        color albedo;
};

__device__ vec3 reflect(const vec3 &v, const vec3 &n) {
            return v - 2.0*dot(v, n) * n;
}

__device__ vec3 refract(const vec3 &uv, const vec3 &n, float etai_over_etat) {
    float cos_theta = min(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

class metal : public material {

    public:
        __device__ metal(const color &a, float f) : albedo(a), fuzz(f < 1 ? f : 1 ) {}

        __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, curandState *local_rand_state) const override {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            
            scattered = ray(rec.hit_point, reflected + fuzz * random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

    public: 
        color albedo;
        float fuzz;
};

class dielectric : public material {

    public:
        __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

        __device__ __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, curandState *local_rand_state) const override {
            attenuation = color(1.0, 1.0, 1.0);
            float refraction_ratio = rec.front_face ? (1.0/ir) : ir;

            vec3 unit_direction = unit_vector(r_in.direction());
            float cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
            float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            vec3 direction;
            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, refraction_ratio);

            scattered = ray(rec.hit_point, direction);
            return true;
        }

    public:
        float ir; // Index of Refraction

    private:
        __device__ static double reflectance(float cosine, float ref_idx) {
            // Use Schlick's approximation for reflectance.
            float r0 = (1-ref_idx) / (1+ref_idx);
            r0 = r0*r0;
            return r0 + (1-r0)*pow((1 - cosine),5);
        }
};

#endif