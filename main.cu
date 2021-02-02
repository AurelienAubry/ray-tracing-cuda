#include <iostream>
#include "vec3.h"
#include "ray.h"

#define COL 1200
#define ROW 600


__device__ color ray_color(const ray& r) {
    color white = color(1.0, 1.0, 1.0);
    color blue = color(0.5, 0.7, 1.0);
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*white + t*blue;
}


__global__ void render(color *frame_buffer, int max_col, int max_row) {

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
    float u = float(col) / float(max_col);
    float v = float(row) / float(max_row);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    frame_buffer[pixel_index] = ray_color(r);
}

__host__ void write_color(std::ostream &out, color pixel_color) {
            int ir = int(255.99*pixel_color.x());
            int ig = int(255.99*pixel_color.y());
            int ib = int(255.99*pixel_color.z());
            out << ir << " " << ig << " " << ib << "\n";
} 

int main() {
    int num_pixels = COL*ROW;
    size_t frame_buffer_size = num_pixels * sizeof(color);

    // Allocate Frame Buffer
    color *frame_buffer;
    cudaMallocManaged((void **)&frame_buffer, frame_buffer_size);

    // Render Frame Buffer
    int t_col = 8;
    int t_row = 8;

    // Nb of blocks in the grid
    dim3 blocks(COL/t_col + 1, ROW/t_row + 1);
    // Nb of threads in each block (one per pixel)
    dim3 threads(t_col, t_row);

    render<<<blocks, threads>>>(frame_buffer, COL, ROW);
    cudaDeviceSynchronize();

    // Output frame buffer as PPM image
    std::cout << "P3\n" << COL << " " << ROW << "\n255\n";
    for(int row = ROW - 1; row >= 0; row--) {
        for(int col = 0; col < COL; col++) {
            size_t pixel_index = row*COL + col;
            write_color(std::cout, frame_buffer[pixel_index]);
        }
    }

    cudaFree(frame_buffer);

}