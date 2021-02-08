#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

const double pi = 3.1415926535897932385;
__device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

class camera {
    public:
        __device__ camera(point3 lookfrom, point3 lookat, vec3 vup, float vfov, float aspect_ratio) {
            float theta = degrees_to_radians(vfov);
            float h = tan(theta / 2);
            float viewport_height = 2.0 * h;
            float viewport_width = aspect_ratio * viewport_height;

            vec3 w = unit_vector(lookfrom - lookat); // "z"
            vec3 u = unit_vector(cross(vup, w)); // "x"
            vec3 v = cross(w, u); // "y"

            origin = lookfrom;
            horizontal = viewport_width * u;
            vertical = viewport_height * v;
            lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
        }

        __device__ ray get_ray(float s, float t) const {
            return ray(origin,  lower_left_corner + s * horizontal + t * vertical - origin);
        }

    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;      

};

#endif