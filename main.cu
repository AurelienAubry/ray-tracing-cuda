#include <iostream>
#include <float.h>
#include <curand_kernel.h>

#include "vec3.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"

#define COL 1200
#define ROW 600


__device__ float hit_sphere(const point3& center, float radius, const ray& r) {

    vec3 origin_center = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(origin_center, r.direction());
    float c = origin_center.length_squared() - radius * radius;
    float discriminant = half_b*half_b - a*c;
    if(discriminant < 0) {
        return -1.0;
    } else {
        return (-half_b -sqrt(discriminant)) / a;
    }
}

__device__ color ray_color(const ray& r, hittable **world) {
    color white = color(1.0, 1.0, 1.0);
    color blue = color(0.5, 0.7, 1.0);
    color red = color(1.0, 0.0, 0.0);

    hit_record rec;
    if((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }
    
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*white + t*blue;
}

__global__ void render_init(int max_row, int max_col, curandState *rand_state) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if ((col >= max_col) || (row >= max_row)) {
        // Pixel outside image
        return;
    }

    int pixel_index = row * max_col + col;

    // Initialize random number generator for the current thread
    curand_init(2021, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(color *frame_buffer, int max_col, int max_row, camera **cam, hittable **world, curandState *rand_state, int samples_per_pixel) {

    point3 origin = point3(0, 0, 0);
    point3 lower_left_corner = point3(-2.0, -1.0, -1.0);
    vec3 horizontal = vec3(4.0, 0.0, 0);
    vec3 vertical = vec3(0.0, 2.0, 0);

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if ((col >= max_col) || (row >= max_row)) {
        // Pixel outside image
        return;
    }

    // Pixel index in the frame buffer (each pixel = 3 floats)
    int pixel_index = row * max_col + col;

    curandState local_rand_state = rand_state[pixel_index];
    color pixel_color(0, 0, 0);
    for(int s = 0; s < samples_per_pixel; s++) {
        float u = float(col + curand_uniform(&local_rand_state)) / float(max_col);
        float v = float(row + curand_uniform(&local_rand_state)) / float(max_row);
        ray r(origin, lower_left_corner + u*horizontal + v*vertical);
        pixel_color += ray_color(r, world);
    }
    
    frame_buffer[pixel_index] = pixel_color;
}

__global__ void create_world(hittable **d_objects_list, hittable **d_world, camera **d_camera) {
    // Make sure this is only executed once
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_objects_list) = new sphere(vec3(0, 0, -1), 0.5);
        *(d_objects_list+1) = new sphere(vec3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_objects_list, 2);
        *d_camera   = new camera();
    }
}

__global__ void free_world(hittable **d_objects_list, hittable **d_world, camera **d_camera) {
   delete *(d_objects_list);
   delete *(d_objects_list+1);
   delete *d_world;
   delete *d_camera;
}

__host__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__host__ void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) {
            float r = pixel_color.x();
            float g = pixel_color.y();
            float b = pixel_color.z();

            float scale = 1.0 / samples_per_pixel;
            r *= scale;
            g *= scale;
            b *= scale;

            int ir = int(255.99*clamp(r, 0.0, 0.999));
            int ig = int(255.99*clamp(g, 0.0, 0.999));
            int ib = int(255.99*clamp(b, 0.0, 0.999));
            out << ir << " " << ig << " " << ib << "\n";
} 

int main() {
    const int num_pixels = COL*ROW;
    const int samples_per_pixel = 100;
    size_t frame_buffer_size = num_pixels * sizeof(color);

    // Allocate Frame Buffer
    color *frame_buffer;
    cudaMallocManaged((void **)&frame_buffer, frame_buffer_size);

    // Allocate world
    hittable **d_objects_list;
    cudaMalloc((void **)&d_objects_list, 2*sizeof(hittable *));
    hittable **d_world;
    cudaMalloc((void **)&d_world, sizeof(hittable *));
    camera **d_camera;
    cudaMalloc((void **)&d_camera, sizeof(camera *));
    create_world<<<1,1>>>(d_objects_list, d_world, d_camera);
    cudaDeviceSynchronize();


    // Render Frame Buffer
    int t_col = 8;
    int t_row = 8;

    // Nb of blocks in the grid
    dim3 blocks(COL/t_col + 1, ROW/t_row + 1);
    // Nb of threads in each block (one per pixel)
    dim3 threads(t_col, t_row);

    // List of pixels random number generator states
    curandState *d_rand_state;
    cudaMalloc((void**)&d_rand_state, num_pixels*sizeof(curandState));

    render_init<<<blocks, threads>>>(COL, ROW, d_rand_state);
    cudaDeviceSynchronize();

    render<<<blocks, threads>>>(frame_buffer, COL, ROW, d_camera, d_world, d_rand_state, samples_per_pixel);
    cudaDeviceSynchronize();

    // Output frame buffer as PPM image
    std::cout << "P3\n" << COL << " " << ROW << "\n255\n";
    for(int row = ROW - 1; row >= 0; row--) {
        for(int col = 0; col < COL; col++) {
            size_t pixel_index = row*COL + col;
            write_color(std::cout, frame_buffer[pixel_index], samples_per_pixel);
        }
    }

    // Clean up
    cudaDeviceSynchronize();
    free_world<<<1,1>>>(d_objects_list, d_world, d_camera);
    cudaFree(d_objects_list);
    cudaFree(d_world);
    cudaFree(frame_buffer);

    cudaDeviceReset();
}