#include "common.hlsli"

// CUMSUM (f32): inclusive prefix sum along each row, one workgroup per row. Rows longer than the
// workgroup are scanned tile by tile with a running carry. The CPU sums serially, so the two
// differ in rounding, not in value.

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

groupshared float sdata[WG_SIZE];

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

    float carry = 0.0f;
    for (uint base = 0; base < ne0; base += WG_SIZE) {
        const uint idx = base + gtid.x;
        sdata[gtid.x] = idx < ne0 ? LOAD_F32(src, src_row + idx) : 0.0f;
        GroupMemoryBarrierWithGroupSync();

        // Hillis-Steele: read the partner before writing, so the two need separate barriers
        for (uint off = 1; off < WG_SIZE; off <<= 1) {
            const float add = gtid.x >= off ? sdata[gtid.x - off] : 0.0f;
            GroupMemoryBarrierWithGroupSync();
            sdata[gtid.x] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        if (idx < ne0) {
            STORE_F32(dst, dst_row + idx, sdata[gtid.x] + carry);
        }
        GroupMemoryBarrierWithGroupSync();
        carry += sdata[WG_SIZE - 1];
        GroupMemoryBarrierWithGroupSync();
    }
}
