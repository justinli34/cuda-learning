// Parallel sum by tree reduction.
// Build: nvcc sum.cu -o sum
// Error checking of CUDA calls is omitted for clarity.

#include <cstdio>
#include <cuda_runtime.h>
#include <utility>
#include <vector>

#define TPB 256 // threads per block (must be a power of two)

// Each block sums its own slice of `in` (in place) and writes the
// result to out[blockIdx.x]. Note: this overwrites the contents of `in`.
__global__ void sumBlocks(float *in, float *out, int n) {
    int t = threadIdx.x;
    int base = blockIdx.x * blockDim.x; // start of this block's slice

    // Round with stride s: thread t (a multiple of 2s) adds in its
    // partner s positions to the right. The second condition keeps the
    // last, partially filled block from reading past the end.
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (t % (2 * s) == 0 && base + t + s < n) in[base + t] += in[base + t + s];
        __syncthreads(); // everyone finishes this round before the next
    }

    if (t == 0) out[blockIdx.x] = in[base];
}

// Sums n >= 1 floats from host memory.
float sum(const float *h_a, int n) {
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, ((n + TPB - 1) / TPB) * sizeof(float));
    cudaMemcpy(d_in, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    // One launch per level: n values -> one partial sum per block.
    // Launches run in order, so each level sees the previous one's output.
    while (n > 1) {
        int blocks = (n + TPB - 1) / TPB;
        sumBlocks<<<blocks, TPB>>>(d_in, d_out, n);
        std::swap(d_in, d_out); // this level's output is the next input
        n = blocks;
    }

    // cudaMemcpy waits for the pending kernels before copying.
    float result;
    cudaMemcpy(&result, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return result;
}

int main() {
    const int n = 10000000;
    std::vector<float> a(n, 1.0f);
    printf("sum = %.0f (expected %d)\n", sum(a.data(), n), n);
    return 0;
}
