#include <iostream>
#include "vec3.h"

#define COL 1200
#define ROW 600

__global__ void render(float *frame_buffer, int max_col, int max_row) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if ((col >= max_col) || (row >= max_row)) {
        // Pixel outside image
        return;
    }

    // Pixel index in the frame buffer (each pixel = 3 floats)
    int pixel_index = row * max_col * 3 + col * 3;
    frame_buffer[pixel_index + 0] = float(col) / max_col;
    frame_buffer[pixel_index + 1] = float(row) / max_row;
    frame_buffer[pixel_index + 2] = 0.2;
}

__host__ void write_color(std::ostream &out, color pixel_color) {
            int ir = int(255.99*pixel_color.x());
            int ig = int(255.99*pixel_color.y());
            int ib = int(255.99*pixel_color.z());
            out << ir << " " << ig << " " << ib << "\n";
} 

int main() {
    int num_pixels = COL*ROW;
    size_t frame_buffer_size = 3 * num_pixels * sizeof(float);

    // Allocate Frame Buffer
    float *frame_buffer;
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
            size_t pixel_index = row*3*COL + col*3;
            color pixel_color(frame_buffer[pixel_index + 0], frame_buffer[pixel_index + 1], frame_buffer[pixel_index + 2]);
            write_color(std::cout, pixel_color);
        }
    }

    cudaFree(frame_buffer);

}