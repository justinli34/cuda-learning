#pragma once
#include <cuda_runtime.h>

// Row-major
struct Matrix {
    float* data;
    int rows;
    int cols;

    __host__ __device__ float& operator()(int row, int col) {
        return data[row * cols + col];
    }
};

void matmul(Matrix A, Matrix B, Matrix out, cudaStream_t stream = nullptr);