#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <source_location>

inline void cudaCheck(cudaError_t err, std::source_location loc = std::source_location::current()) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s at %s:%u\n", cudaGetErrorString(err), loc.file_name(),
                     static_cast<unsigned>(loc.line()));
        std::exit(1);
    }
}

// Returns average time in ms
template <typename F> float benchmark(F&& func, int runs) {
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    // warm-up
    func();
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        func();
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms = 0;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    return ms / runs;
}