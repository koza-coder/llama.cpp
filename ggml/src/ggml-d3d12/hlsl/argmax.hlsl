#include "common.hlsli"

// ARGMAX (f32 -> i32): dst[i1] = index of the largest element of row i1, one workgroup per row.
// Ties resolve to the LAST index, matching ggml_vec_argmax_f32, whose running-max compare fires
// again on every element equal to the max.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src1;
    uint ne0;
    uint n_rows;
    uint nwg_x;
};

groupshared float best_val[WG_SIZE];
groupshared uint  best_idx[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint row = gid.y * nwg_x + gid.x;
    if (row >= n_rows) {
        return;
    }
    const uint src_row = offset_src + row * stride_src1;

    float v = asfloat(0xFF800000u);   // -inf; the literal form differs between compilers
    uint  k = 0;
    for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
        const float x = LOAD_F32(src, src_row + c);
        // >= keeps the later index on a tie, and the stride means c only grows
        if (x >= v) {
            v = x;
            k = c;
        }
    }
    best_val[gtid.x] = v;
    best_idx[gtid.x] = k;
    GroupMemoryBarrierWithGroupSync();

    for (uint off = WG_SIZE / 2; off > 0; off /= 2) {
        if (gtid.x < off) {
            const float ov = best_val[gtid.x + off];
            const uint  oi = best_idx[gtid.x + off];
            // equal values: the larger column index wins, as on the CPU
            if (ov > best_val[gtid.x] || (ov == best_val[gtid.x] && oi > best_idx[gtid.x])) {
                best_val[gtid.x] = ov;
                best_idx[gtid.x] = oi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gtid.x == 0) {
        STORE_I32(dst, offset_dst + row, (int) best_idx[0]);
    }
}
