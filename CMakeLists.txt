cmake_minimum_required (VERSION 3.14)
project (mandelbrot_3D CUDA CXX)

option (GPU "Build against cuda" NO)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sdl2)

include(FetchContent)
FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.8.0.tar.gz
  )
FetchContent_MakeAvailable(CLI11)

# Compilation flags
set(CXXFLAGS -Wall -Wextra -pedantic -O5 -std=c++17)
set(CUDAFLAGS "--compiler-options=-Wall -Wextra -O5")
#set(CXXFLAGS -Wall -Wextra -pedantic -ggdb -g -std=c++17)
#set(CUDAFLAGS "--compiler-options=-Wall -Wextra -ggdb -lineinfo" -G -g)
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:${CXXFLAGS}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:${CUDAFLAGS}>"
)

# System requirements
find_package (CUDA REQUIRED)
find_package (SDL2 REQUIRED)
find_package (SDL2_ttf REQUIRED)


# Copy static files
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})

file (GLOB_RECURSE SOURCES "src/*.cc")
file (GLOB_RECURSE CU_SOURCES "src/*.cu")
add_executable (${PROJECT_NAME} ${SOURCES} ${CU_SOURCES})
#target_compile_features (${PROJECT_NAME} PUBLIC cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} SDL2::Main SDL2::TTF CLI11::CLI11)

