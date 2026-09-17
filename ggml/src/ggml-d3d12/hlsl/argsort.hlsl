#include "common.hlsli"

// ARGSORT and TOP_K (f32 -> i32), one workgroup per row: every element computes its own rank, so no sorting
// network, and writes its index to dst[rank] when rank < k_out (k_out = ne0 for ARGSORT).
// Ties keep the smaller source index first, which matches the CPU reference for distinct values.
// defines: SORT_DESC (TOP_K always sorts descending)

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src1;
    uint ne0;
    uint k_out;
    uint row0;     // first row of this dispatch
    uint n_rows;   // rows in this dispatch
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint wg = gid.y * nwg_x + gid.x;
    if (wg >= n_rows) {
        return;
    }
    const uint r = row0 + wg;
    const uint src_row = offset_src + r * stride_src1;
    const uint dst_row = offset_dst + r * k_out;
    for (uint i = gtid.x; i < ne0; i += WG_SIZE) {
        const float v = LOAD_F32(src, src_row + i);
        uint rank = 0;
        for (uint j = 0; j < ne0; j++) {
            const float w = LOAD_F32(src, src_row + j);
#if defined(SORT_DESC)
            const bool before = w > v || (w == v && j < i);
#else
            const bool before = w < v || (w == v && j < i);
#endif
            if (before) {
                rank++;
            }
        }
        if (rank < k_out) {
            STORE_I32(dst, dst_row + rank, (int) i);
        }
    }
}
