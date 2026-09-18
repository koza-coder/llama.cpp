#include "common.hlsli"

// PAD_REFLECT_1D (f32): dst row = src row placed at offset p0, with both margins filled by
// reflecting about the edge WITHOUT repeating it. The CPU builds this by copying the row and then
// mirroring in place; expressed as a gather, dst[j] reads src[k] with
//   j <  p0            -> k = p0 - j
//   j >= p0 + src_ne0  -> k = 2 * (src_ne0 - 1) - (j - p0)
//   otherwise          -> k = j - p0

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
    uint src_ne0;
    uint p0;
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

    const uint src_row = offset_src + i3 * stride_src3 + i2 * stride_src2 + i1 * stride_src1;
    const uint dst_row = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1;

    for (uint j = gtid.x; j < ne0; j += WG_SIZE) {
        uint k;
        if (j < p0) {
            k = p0 - j;
        } else if (j >= p0 + src_ne0) {
            k = 2u * (src_ne0 - 1u) - (j - p0);
        } else {
            k = j - p0;
        }
        STORE_F32(dst, dst_row + j, LOAD_F32(src, src_row + k));
    }
}
