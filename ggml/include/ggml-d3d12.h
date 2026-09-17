#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_D3D12_NAME "D3D12"

GGML_BACKEND_API ggml_backend_t ggml_backend_d3d12_init(int device);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_d3d12_reg(void);

#ifdef  __cplusplus
}
#endif
