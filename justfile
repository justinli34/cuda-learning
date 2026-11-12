set windows-shell := ["pwsh.exe", "-NoProfile", "-Command"]

config := "Release"

configure:
    cmake -B build -G "Visual Studio 17 2022"

build: configure
    cmake --build build --config {{config}} --parallel

run name="main" *args: build
    ./build/{{config}}/{{name}}.exe {{args}}

clean:
    Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

cuda_flags := "-xcuda -std=c++20 '--cuda-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4' --cuda-gpu-arch=sm_120 -Wno-unknown-cuda-version"

format:
    clang-format -i (Get-ChildItem src -Include *.cu,*.cuh -Recurse).FullName

lint *args:
    clang-tidy {{args}} (Get-ChildItem src -Filter *.cu -Recurse).FullName -- {{cuda_flags}}
