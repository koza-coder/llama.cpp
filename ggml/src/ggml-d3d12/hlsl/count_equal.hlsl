#include "common.hlsli"

// COUNT_EQUAL (i32, i32 -> i64): how many elements of the two tensors are equal, as one scalar.
// A single workgroup walks everything, as in sum.hlsl. The destination is 64-bit, so the count
// goes in the low word and the high word is cleared; the count cannot exceed the element count,
// which the backend caps at UINT32_MAX.

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst_bytes;   // dst is i64, so the C++ side converts its element offset to bytes
    uint stride_01;
    uint stride_02;
    uint stride_03;
    uint stride_11;
    uint stride_12;
    uint stride_13;
    uint ne0;
    uint ne1;
    uint ne2;
    uint n_rows;
    uint nwg_x;
};

groupshared uint scratch[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID) {
    uint count = 0;
    for (uint r = 0; r < n_rows; r++) {
        const uint i3 = r / (ne2 * ne1);
        const uint rm = r % (ne2 * ne1);
        const uint i2 = rm / ne1;
        const uint i1 = rm % ne1;
        const uint row0 = offset_src0 + i3 * stride_03 + i2 * stride_02 + i1 * stride_01;
        const uint row1 = offset_src1 + i3 * stride_13 + i2 * stride_12 + i1 * stride_11;
        for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
            if (LOAD_I32(src0, row0 + c) == LOAD_I32(src1, row1 + c)) {
                count++;
            }
        }
    }
    scratch[gtid.x] = count;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = WG_SIZE / 2; off > 0; off /= 2) {
        if (gtid.x < off) {
            scratch[gtid.x] += scratch[gtid.x + off];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gtid.x == 0) {
        dst.Store(offset_dst_bytes, scratch[0]);
        dst.Store(offset_dst_bytes + 4, 0u);
    }
}
