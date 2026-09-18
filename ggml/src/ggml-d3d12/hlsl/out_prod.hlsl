#include "common.hlsli"

// OUT_PROD (f32): dst[i0, i1, i2, i3] = sum over i01 of
//     src0[i0, i01, i2 / dps2, i3 / dps3] * src1[i1, i01, i2, i3]
// The CPU zeroes dst and accumulates with scattered mads; expressed as a gather, one thread owns
// one destination element and no zeroing pass is needed. dps2/dps3 are the grouped-query
// broadcast factors ne2 / ne02 and ne3 / ne03.

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;
    uint stride_01;
    uint stride_02;
    uint stride_03;
    uint stride_10;
    uint stride_11;
    uint stride_12;
    uint stride_13;
    uint stride_d1;
    uint stride_d2;
    uint stride_d3;
    uint ne0;
    uint ne1;
    uint ne2;
    uint ne01;
    uint dps2;
    uint dps3;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    const uint i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    const uint i1 = i / ne0;
    const uint i0 = i % ne0;

    const uint a_base = offset_src0 + i0 + (i2 / dps2) * stride_02 + (i3 / dps3) * stride_03;
    const uint b_base = offset_src1 + i1 * stride_10 + i2 * stride_12 + i3 * stride_13;

    float sum = 0.0f;
    for (uint i01 = 0; i01 < ne01; i01++) {
        sum += LOAD_F32(src0, a_base + i01 * stride_01) * LOAD_F32(src1, b_base + i01 * stride_11);
    }
    STORE_F32(dst, offset_dst + i1 * stride_d1 + i2 * stride_d2 + i3 * stride_d3 + i0, sum);
}
