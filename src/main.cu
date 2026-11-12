#include "matmul.cuh"
#include "utils.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

int main() {
    constexpr int M = 1024;
    constexpr int K = 1024;
    constexpr int N = 1024;

    std::vector<float> hA(static_cast<size_t>(M) * K, 1.0f);
    std::vector<float> hB(static_cast<size_t>(K) * N, 2.0f);

    Matrix A = {nullptr, M, K};
    Matrix B = {nullptr, K, N};
    Matrix out = {nullptr, M, N};

    cudaCheck(cudaMalloc(&A.data, hA.size() * sizeof(float)));
    cudaCheck(cudaMalloc(&B.data, hB.size() * sizeof(float)));
    cudaCheck(cudaMalloc(&out.data, static_cast<size_t>(M) * N * sizeof(float)));

    cudaCheck(cudaMemcpy(A.data, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(B.data, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    float avg_time = benchmark([&] { matmul(A, B, out); }, 20);
    printf("%.3f ms per run\n", avg_time);

    cudaCheck(cudaFree(A.data));
    cudaCheck(cudaFree(B.data));
    cudaCheck(cudaFree(out.data));
    return 0;
}
