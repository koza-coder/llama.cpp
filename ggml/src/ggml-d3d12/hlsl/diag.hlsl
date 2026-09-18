#include "common.hlsli"

// DIAG (f32): turn each vector into a diagonal matrix. src is [ne0, 1, ne2, ne3] and dst is
// [ne0, ne0, ne2, ne3], with dst[i0, i1, ...] = i0 == i1 ? src[i1, 0, ...] : 0.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src2;
    uint stride_src3;
    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;
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

    const uint dst_row = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1;
    const uint src_row = offset_src + i3 * stride_src3 + i2 * stride_src2;

    for (uint i0 = gtid.x; i0 < ne0; i0 += WG_SIZE) {
        STORE_F32(dst, dst_row + i0, i0 == i1 ? LOAD_F32(src, src_row + i1) : 0.0f);
    }
}
