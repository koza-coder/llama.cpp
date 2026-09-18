#include "common.hlsli"

// SILU_BACK (f32): dst = dy * s * (1 + x*(1 - s)) with s = sigmoid(x), following
// ggml_silu_backward_f32. src[0] is the incoming gradient dy, src[1] is the forward input x.
// Elementwise: one thread each.

RWByteAddressBuffer dy_buf : register(u0);
RWByteAddressBuffer x_buf  : register(u1);
RWByteAddressBuffer dst    : register(u2);

cbuffer Params : register(b0) {
    uint offset_dy;
    uint offset_x;
    uint offset_dst;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const float x  = LOAD_F32(x_buf, offset_x + i);
    const float dy = LOAD_F32(dy_buf, offset_dy + i);
    const float s  = 1.0f / (1.0f + exp(-x));
    STORE_F32(dst, offset_dst + i, dy * s * (1.0f + x * (1.0f - s)));
}
