#include <iostream>
#include <float.h>
#include <curand_kernel.h>

#include "vec3.h"
#include "ray.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#define COL 600
#define ROW 300

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ color ray_color(const ray& r, hittable **world, curandState *local_rand_state, int max_depth) {
    color white = color(1.0, 1.0, 1.0);
    color blue = color(0.5, 0.7, 1.0);
    color red = color(1.0, 0.0, 0.0);

    ray cur_ray = r;
    color cur_attenuation = color(1.0, 1.0, 1.0);
    for(int i = 0; i < max_depth; i++) {
    hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            color attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return color(0.0,0.0,0.0);
            }
            
        } else {
            vec3 unit_direction = unit_vector(r.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*white + t*blue;
            return cur_attenuation * c;
        }
    }
    //return cur_attenuation * color(0,0,0);

    return color(0.0,0.0,0.0); // exceeded recursion
   
}

__global__ void render_init(int max_col, int max_row, curandState *rand_state) {
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

__global__ void render(color *frame_buffer, int max_col, int max_row, camera **cam, hittable **world, curandState *rand_state, int samples_per_pixel, int max_depth) {

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
        ray r = (*cam)->get_ray(u,v);
        pixel_color += ray_color(r, world, &local_rand_state, max_depth);
    }
    rand_state[pixel_index] = local_rand_state;
    frame_buffer[pixel_index] = pixel_color;
}

__global__ void create_world(hittable **d_objects_list, hittable **d_world, camera **d_camera) {
    // Make sure this is only executed once
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_objects_list) = new sphere(vec3(0, 0, -1), 0.5, new lambertian(color(0.1, 0.2, 0.5)));
        *(d_objects_list+1) = new sphere(vec3(1, 0, -1), 0.5, new metal(color(0.8, 0.8, 0.5), 0.5));
        *(d_objects_list+2) = new sphere(vec3(-1, 0, -1), 0.5, new dielectric(1.5));
        *(d_objects_list+3) = new sphere(vec3(-1, 0, -1), -0.4, new dielectric(1.5));
        *(d_objects_list+4) = new sphere(vec3(0, -100.5, -1), 100, new lambertian(color(0.8, 0.8, 0.0)));
        *d_world = new hittable_list(d_objects_list, 5);
        *d_camera   = new camera(point3(-2, 2, 1), point3(0, 0, -1), vec3(0, 1, 0), 20, 16.0 / 9.0);
    }
}

__global__ void free_world(hittable **d_objects_list, hittable **d_world, camera **d_camera) {
   delete *(d_objects_list);
   delete *(d_objects_list+1);
   delete *(d_objects_list+2);
   delete *(d_objects_list+3);
   delete *(d_objects_list+4);
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
            r = sqrt(r * scale);
            g = sqrt(g * scale);
            b = sqrt(b * scale);

            int ir = int(255.99*clamp(r, 0.0, 0.999));
            int ig = int(255.99*clamp(g, 0.0, 0.999));
            int ib = int(255.99*clamp(b, 0.0, 0.999));
            out << ir << " " << ig << " " << ib << "\n";
} 

int main() {
    cudaDeviceReset();
    const int num_pixels = COL*ROW;
    const int samples_per_pixel = 100;
    const int max_depth = 50;
    size_t frame_buffer_size = num_pixels * sizeof(color);

    // Allocate Frame Buffer
    color *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, frame_buffer_size));

    // Allocate world
    hittable **d_objects_list;
    checkCudaErrors(cudaMalloc((void **)&d_objects_list, 5*sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_objects_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Render Frame Buffer
    int t_col = 8;
    int t_row = 8;

    // Nb of blocks in the grid
    dim3 blocks(COL/t_col + 1, ROW/t_row + 1);
    // Nb of threads in each block (one per pixel)
    dim3 threads(t_col, t_row);

    // List of pixels random number generator states
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels*sizeof(curandState)));

    render_init<<<blocks, threads>>>(COL, ROW, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(frame_buffer, COL, ROW, d_camera, d_world, d_rand_state, samples_per_pixel, max_depth);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output frame buffer as PPM image
    std::cout << "P3\n" << COL << " " << ROW << "\n255\n";
    for(int row = ROW - 1; row >= 0; row--) {
        for(int col = 0; col < COL; col++) {
            size_t pixel_index = row*COL + col;
            write_color(std::cout, frame_buffer[pixel_index], samples_per_pixel);
        }
    }

    // Clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_objects_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(frame_buffer));
    cudaDeviceReset();
}