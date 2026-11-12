#include "matmul.cuh"
#include "utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void matmulKernel(Matrix A, Matrix B, Matrix out) {
    // map col to x: threads in a warp have consecutive threadIdx.x and matrix is row-major, so accessing adjacent
    // memory addresses can be coalesced (combined into fewer transactions)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int rows = out.rows;
    int cols = out.cols;

    if (row >= rows || col >= cols) return;

    float result = 0.0f;
    for (int k = 0; k < A.cols; k++) {
        result += A(row, k) * B(k, col);
    }

    out(row, col) = result;
}

void matmul(Matrix A, Matrix B, Matrix out, cudaStream_t stream) {
    if (A.cols != B.rows || out.rows != A.rows || out.cols != B.cols) {
        std::fprintf(stderr, "matmul: shape mismatch (%dx%d) * (%dx%d) -> (%dx%d)\n", //
                     A.rows, A.cols, B.rows, B.cols, out.rows, out.cols);
        std::exit(1);
    }

    dim3 block(16, 16);                                                                // threads
    dim3 grid((out.cols + block.x - 1) / block.x, (out.rows + block.y - 1) / block.y); // blocks
    matmulKernel<<<grid, block, 0, stream>>>(A, B, out);
    cudaCheck(cudaGetLastError());
}