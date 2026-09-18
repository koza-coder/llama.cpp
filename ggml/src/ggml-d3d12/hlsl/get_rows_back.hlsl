#include "common.hlsli"

// GET_ROWS_BACK (f32), one workgroup per destination row. The CPU zeroes dst and scatters
// src0 row i onto dst row idx[i], accumulating when an index repeats. Read as a gather this
// is one scan of the index list per destination row, which needs no atomics: the index list
// is short (it has one entry per row that the forward GET_ROWS picked).
// The index list is read flat, exactly as ggml_compute_forward_get_rows_back_f32 reads it.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer idx : register(u1);
RWByteAddressBuffer dst : register(u2);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_idx;
    uint offset_dst;
    uint nc;           // columns per row
    uint n_rows;       // rows of dst
    uint nr;           // entries in the index list
    uint stride_src;   // src0 nb[1] / 4
    uint stride_dst;   // dst  nb[1] / 4
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint row = gid.y * nwg_x + gid.x;
    if (row >= n_rows) {
        return;
    }
    for (uint col = gtid.x; col < nc; col += WG_SIZE) {
        float sum = 0.0f;
        for (uint i = 0; i < nr; i++) {
            if ((uint) LOAD_I32(idx, offset_idx + i) == row) {
                sum += LOAD_F32(src, offset_src + i * stride_src + col);
            }
        }
        STORE_F32(dst, offset_dst + row * stride_dst + col, sum);
    }
}
