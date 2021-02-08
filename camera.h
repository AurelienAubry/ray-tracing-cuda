#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

const double pi = 3.1415926535897932385;
__device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

class camera {
    public:
        __device__ camera(
            point3 lookfrom, 
            point3 lookat, 
            vec3 vup, 
            float vfov, 
            float aspect_ratio,
            float aperture,
            float focus_dist
        ) {
            float theta = degrees_to_radians(vfov);
            float h = tan(theta / 2);
            float viewport_height = 2.0 * h;
            float viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat); // "z"
            u = unit_vector(cross(vup, w)); // "x"
            v = cross(w, u); // "y"

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist*w;

            lens_radius = aperture / 2;
        }

        __device__ ray get_ray(float s, float t, curandState *local_rand_state) const {
            vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
            vec3 offset = u * rd.x() + v * rd.y();

            return ray(origin + offset,  lower_left_corner + s * horizontal + t * vertical - origin - offset);
        }

    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;      
        vec3 u, v, w;
        float lens_radius;

};

#endif