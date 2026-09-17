#include "common.hlsli"

// GET_ROWS for quantized src: dst[i1, i2, i3] = dequantize(src[idx[i1, i2, i3], i2, i3]), idx is i32, dst f32.
// TPR threads share one row and dequantize every TPR-th sub-block, reusing the matrix-vector dequant paths.
// src offsets and strides are in blocks. defines: SRC0_<TYPE>, TPR

#define MAX_COLS 1

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer idx : register(u1);
RWByteAddressBuffer dst : register(u2);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_idx;
    uint offset_dst;

    uint stride_src1;
    uint stride_src2;
    uint stride_src3;

    uint stride_idx0;
    uint stride_idx1;
    uint stride_idx2;

    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint k;        // dst row length
    uint ne1;      // idx ne0
    uint ne2;      // idx ne1
    uint n_rows;   // rows of dst

    uint nwg_x;
};

static uint g_dst_base;

#define ACC(a, kidx) { STORE_F32(dst, g_dst_base + (kidx), a); }
#include "dequant_row.hlsli"

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint t = flat_index(gid, nwg_x);
    if (t >= n_rows * TPR) {
        return;
    }
    const uint lane = t % TPR;
    uint       r    = t / TPR;

    const uint i3 = r / (ne2 * ne1);
    r = r % (ne2 * ne1);
    const uint i2 = r / ne1;
    const uint i1 = r % ne1;

    const uint row = (uint) LOAD_I32(idx, offset_idx + i1 * stride_idx0 + i2 * stride_idx1 + i3 * stride_idx2);
    g_dst_base     = offset_dst + i1 * stride_dst1 + i2 * stride_dst2 + i3 * stride_dst3;

    uint  src1_base[MAX_COLS];
    float acc[MAX_COLS];
    src1_base[0] = 0;
    acc[0]       = 0.0f;
    dot_row(src, offset_src + row * stride_src1 + i2 * stride_src2 + i3 * stride_src3, lane, 0, src1_base, acc);
}
