#include "common.hlsli"

// ADD1 (f32): dst = src0 + src1[0]. The addend is a scalar living in device memory, so it is read
// from the buffer rather than passed in the constant buffer.

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
    uint stride_d1;
    uint stride_d2;
    uint stride_d3;
    uint ne0;
    uint ne1;
    uint ne2;
    uint n_rows;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint r = gid.y * nwg_x + gid.x;
    if (r >= n_rows) {
        return;
    }
    const uint i3 = r / (ne2 * ne1);
    r = r % (ne2 * ne1);
    const uint i2 = r / ne1;
    const uint i1 = r % ne1;

    const uint src_row = offset_src0 + i3 * stride_03 + i2 * stride_02 + i1 * stride_01;
    const uint dst_row = offset_dst + i3 * stride_d3 + i2 * stride_d2 + i1 * stride_d1;
    const float addend = LOAD_F32(src1, offset_src1);

    for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
        STORE_F32(dst, dst_row + c, LOAD_F32(src0, src_row + c) + addend);
    }
}
