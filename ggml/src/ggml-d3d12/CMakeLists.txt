cmake_minimum_required(VERSION 3.13)

find_package(Python3 REQUIRED)

# HLSL sources are embedded as strings and compiled at runtime with dxcompiler.dll
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hlsl")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(SHADER_HEADER "${SHADER_OUTPUT_DIR}/ggml-d3d12-shaders.hpp")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

file(GLOB HLSL_SHADER_FILES
    "${SHADER_DIR}/*.hlsl"
    "${SHADER_DIR}/*.hlsli"
)

add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND} -E echo "Embedding HLSL shaders to ggml-d3d12-shaders.hpp"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONIOENCODING=utf-8
        ${Python3_EXECUTABLE} ${SHADER_DIR}/embed_hlsl.py
            --input_dir "${SHADER_DIR}"
            --output_file "${SHADER_HEADER}"
    DEPENDS ${HLSL_SHADER_FILES} ${SHADER_DIR}/embed_hlsl.py
    VERBATIM
)

add_custom_target(generate_d3d12_shaders DEPENDS ${SHADER_HEADER})

ggml_add_backend_library(ggml-d3d12
    ggml-d3d12.cpp
    ${SHADER_HEADER}
    ../../include/ggml-d3d12.h
)

add_dependencies(ggml-d3d12 generate_d3d12_shaders)

target_compile_features(ggml-d3d12 PRIVATE cxx_std_17)
target_include_directories(ggml-d3d12 PRIVATE ${SHADER_OUTPUT_DIR})
target_link_libraries(ggml-d3d12 PRIVATE d3d12 dxgi dxguid)

if (GGML_D3D12_DEBUG)
    target_compile_definitions(ggml-d3d12 PRIVATE GGML_D3D12_DEBUG=1)
endif()
