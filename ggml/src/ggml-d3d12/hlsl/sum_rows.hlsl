#include "common.hlsli"

// SUM_ROWS (f32): dst[0, i1, i2, i3] = sum of the row, one workgroup per row
// With MEAN defined the same kernel divides by the row length, which is ggml's MEAN op.

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
    uint nwg_x;
};

groupshared float scratch[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint i = gid.y * nwg_x + gid.x;
    if (i >= n_rows) {
        return;
    }
    const uint i3 = i / (ne2 * ne1);
    i = i % (ne2 * ne1);
    const uint i2 = i / ne1;
    const uint i1 = i % ne1;
    const uint src_row = offset_src + i3 * stride_src3 + i2 * stride_src2 + i1 * stride_src1;

    float sum = 0;
    for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
        sum += LOAD_F32(src, src_row + c);
    }
    scratch[gtid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = WG_SIZE / 2; off > 0; off /= 2) {
        if (gtid.x < off) {
            scratch[gtid.x] += scratch[gtid.x + off];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gtid.x == 0) {
#ifdef MEAN
        const float res = scratch[0] / (float) ne0;
#else
        const float res = scratch[0];
#endif
        STORE_F32(dst, offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1, res);
    }
}
