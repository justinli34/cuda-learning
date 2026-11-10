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
