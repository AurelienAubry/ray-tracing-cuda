#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {

    public:
        __device__ sphere() {}
        __device__ sphere(point3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m) {};
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

        point3 center;
        float radius;
        material *mat_ptr;

};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 origin_center = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(origin_center, r.direction());
    float c = origin_center.length_squared() - radius * radius;
    float discriminant = half_b*half_b - a*c;
    
    if(discriminant < 0) {
        return false;
    }

    float sqrt_discriminant = sqrt(discriminant);

    // Find nearest root in [t_min, t_max]
    float root = (-half_b - sqrt_discriminant) / a;
    if(root < t_min || root > t_max) {
        root = (-half_b + sqrt_discriminant) / a;
        if(root < t_min || root > t_max) {
            return false;
        }
    }

    rec.t = root;
    rec.hit_point = r.at(rec.t);
    vec3 outward_normal = (rec.hit_point - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    
    return true;
}


#endif