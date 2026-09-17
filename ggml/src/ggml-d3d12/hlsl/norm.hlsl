#include "common.hlsli"

// NORM (layer norm without weights), one workgroup per row, f32 only:
// dst = (src - mean) / sqrt(mean((src - mean)^2) + eps)

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

    float eps;

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
    const uint dst_row = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1;

    // pass 1: sum
    float sum = 0;
    for (uint col = gtid.x; col < ne0; col += WG_SIZE) {
        sum += LOAD_F32(src, src_row + col);
    }
    scratch[gtid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = WG_SIZE / 2; off > 0; off /= 2) {
        if (gtid.x < off) {
            scratch[gtid.x] += scratch[gtid.x + off];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float mean = scratch[0] / (float) ne0;

    // pass 2: variance around the mean
    float var = 0;
    for (uint c2 = gtid.x; c2 < ne0; c2 += WG_SIZE) {
        const float v = LOAD_F32(src, src_row + c2) - mean;
        var += v * v;
    }
    scratch[gtid.x] = var;
    GroupMemoryBarrierWithGroupSync();
    for (uint off2 = WG_SIZE / 2; off2 > 0; off2 /= 2) {
        if (gtid.x < off2) {
            scratch[gtid.x] += scratch[gtid.x + off2];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float scale = 1.0f / sqrt(scratch[0] / (float) ne0 + eps);

    for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
        STORE_F32(dst, dst_row + c, (LOAD_F32(src, src_row + c) - mean) * scale);
    }
}
