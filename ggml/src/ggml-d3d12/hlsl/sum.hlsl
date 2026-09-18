#include "common.hlsli"

// SUM (f32): dst is a scalar holding the sum of every element of src, one workgroup total.
// The CPU reference accumulates in double; this accumulates in float, but the group tree keeps
// the error close to pairwise summation rather than a serial walk.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src1;
    uint stride_src2;
    uint stride_src3;
    uint ne0;
    uint ne1;
    uint ne2;
    uint n_rows;
    uint nwg_x;
};

groupshared float scratch[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID) {
    float sum = 0;
    for (uint r = 0; r < n_rows; r++) {
        const uint i3 = r / (ne2 * ne1);
        const uint rm = r % (ne2 * ne1);
        const uint i2 = rm / ne1;
        const uint i1 = rm % ne1;
        const uint src_row = offset_src + i3 * stride_src3 + i2 * stride_src2 + i1 * stride_src1;
        for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
            sum += LOAD_F32(src, src_row + c);
        }
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
        STORE_F32(dst, offset_dst, scratch[0]);
    }
}
