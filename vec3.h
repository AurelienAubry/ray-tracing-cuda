#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class vec3 {

    public: 
        __host__ __device__ vec3() : e {0, 0, 0} {}
        __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

        __host__ __device__ inline float x() const { return e[0]; }
        __host__ __device__ inline float y() const { return e[1]; }
        __host__ __device__ inline float z() const { return e[2]; }

        __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

        __host__ __device__ vec3& operator+=(const vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        __host__ __device__ vec3& operator*=(const float t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        __host__ __device__ inline vec3& operator*=(const vec3 &v){
            e[0]  *= v.e[0];
            e[1]  *= v.e[1];
            e[2]  *= v.e[2];
            return *this;
        }

        __host__ __device__ vec3& operator/=(const float t) {
            return *this *= 1/t;
        }

        __host__ __device__ inline float length() const {
            return sqrt(length_squared());
        }

        __host__ __device__ inline float length_squared() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }

        __device__ inline static vec3 random(curandState *local_rand_state) {
            return vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
        }

        __device__ bool near_zero() const {
            const float eps = 1e-8;
            return (fabsf(e[0] < eps) && fabsf(e[1] < eps) && fabsf(e[2] < eps));
        }
        
    public:
        float e[3];
};

using point3 = vec3;   // 3D point
using color = vec3;    // RGB color

inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__  inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__  inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__  inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__  inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

 __device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    while(true) {
        p = 2.0f * vec3::random(local_rand_state) - vec3(1.0f,1.0f,1.0f);
        if (p.length_squared() >= 1.0f) continue;
        return p;
    }
}

__device__ vec3 random_unit_vector(curandState *local_rand_state) {
    return unit_vector(random_in_unit_sphere(local_rand_state));
}

#endif