#include "common.hlsli"

// SSM_CONV (f32): dst[i1, i2, i3] = sum_i0 src0[i2 + i0, i1, i3] * src1[i0, i1]
// src0 {d_conv - 1 + n_t, d_inner, n_s}, src1 {d_conv, d_inner}, dst {d_inner, n_t, n_s}; one thread per dst element

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;

    uint stride_src01;   // = src0->ne[0]
    uint stride_src02;
    uint stride_src11;
    uint stride_dst0;
    uint stride_dst1;
    uint stride_dst2;

    uint nc;             // d_conv
    uint nr;             // d_inner
    uint n_t;
    uint ne;             // nr * n_t * n_s

    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint i3 = i / (nr * n_t);
    i = i % (nr * n_t);
    const uint i2 = i / nr;
    const uint i1 = i % nr;

    const uint s_base = offset_src0 + i3 * stride_src02 + i1 * stride_src01 + i2;
    const uint c_base = offset_src1 + i1 * stride_src11;
    float sum = 0.0f;
    for (uint i0 = 0; i0 < nc; i0++) {
        sum += LOAD_F32(src0, s_base + i0) * LOAD_F32(src1, c_base + i0);
    }
    STORE_F32(dst, offset_dst + i3 * stride_dst2 + i2 * stride_dst1 + i1 * stride_dst0, sum);
}
