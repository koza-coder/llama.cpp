#include "common.hlsli"

// TRI (f32): keep the triangular part of each matrix and zero the rest. tri_type selects which
// half and whether the diagonal is included. The values are the ggml_tri_type enum in ggml.h,
// which is NOT the order the switch in ops.cpp happens to list:
//   0 UPPER_DIAG (i0 >= i1), 1 UPPER (i0 > i1), 2 LOWER_DIAG (i0 <= i1), 3 LOWER (i0 < i1)

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src1;
    uint stride_src2;
    uint stride_src3;
    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;
    uint ne0;
    uint ne1;
    uint ne2;
    uint n_rows;
    uint tri_type;
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

    const uint src_row = offset_src + i3 * stride_src3 + i2 * stride_src2 + i1 * stride_src1;
    const uint dst_row = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1;

    for (uint i0 = gtid.x; i0 < ne0; i0 += WG_SIZE) {
        bool keep;
        if (tri_type == 0) {
            keep = i0 >= i1;
        } else if (tri_type == 1) {
            keep = i0 > i1;
        } else if (tri_type == 2) {
            keep = i0 <= i1;
        } else {
            keep = i0 < i1;
        }
        STORE_F32(dst, dst_row + i0, keep ? LOAD_F32(src, src_row + i0) : 0.0f);
    }
}
