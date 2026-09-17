#include "common.hlsli"

// SET_ROWS f32 -> q8_0: dst[idx[row]] = quantize(src[row]), one thread per source row. q8_0 blocks are 34 bytes
// (f16 scale, 32 int8), so rows do not start on word boundaries; a row writes its bytes word by word and keeps
// the neighbour bytes of its first and last word. Rows whose global dst row numbers have the other parity run in
// a second dispatch, so two threads never touch the same word. dst is contiguous; offsets and strides are in
// blocks. defines: I64_IDX

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer idx : register(u1);
RWByteAddressBuffer dst : register(u2);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_idx;
    uint offset_dst;

    uint stride_src1;
    uint stride_src2;
    uint stride_src3;

    uint stride_idx0;
    uint stride_idx1;
    uint stride_idx2;

    uint stride_dst1;   // blocks per dst row
    uint dst_ne1;
    uint dst_ne2;

    uint ne0;
    uint n_rows;
    uint ne2;
    uint ne3;

    uint idx1;
    uint idx2;
    uint parity;

    uint nwg_x;
};

static uint g_word;

// write one byte at absolute byte address pos of a row that ends at byte end
void put_byte(uint pos, uint end, uint b) {
    if ((pos & 3u) == 0u) {
        // a word that the row does not fill to the end keeps the neighbour's high bytes
        g_word = pos + 4u > end ? dst.Load(pos) : 0u;
    }
    const uint sh = (pos & 3u) * 8u;
    g_word = (g_word & ~(0xFFu << sh)) | ((b & 0xFFu) << sh);
    if ((pos & 3u) == 3u || pos + 1u == end) {
        dst.Store(pos & ~3u, g_word);
    }
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint row = flat_index(gid, nwg_x);
    if (row >= ne3 * ne2 * n_rows) {
        return;
    }
    const uint i_src3 = row / (ne2 * n_rows);
    row = row % (ne2 * n_rows);
    const uint i_src2 = row / n_rows;
    const uint i_src1 = row % n_rows;

    const uint idx_elem = offset_idx + i_src1 * stride_idx0 + (i_src2 % idx1) * stride_idx1 + (i_src3 % idx2) * stride_idx2;
#ifdef I64_IDX
    const uint idx_val = idx.Load(idx_elem * 8);
#else
    const uint idx_val = idx.Load(idx_elem * 4);
#endif
    const uint grow = (i_src3 * dst_ne2 + i_src2) * dst_ne1 + idx_val;   // dst row number in the contiguous tensor
    if ((grow & 1u) != parity) {
        return;
    }

    const uint i_src_row = offset_src + i_src1 * stride_src1 + i_src2 * stride_src2 + i_src3 * stride_src3;
    const uint base      = (offset_dst + grow * stride_dst1) * 34u;
    const uint n_blocks  = ne0 / 32u;
    const uint end       = base + n_blocks * 34u;

    // the first word may start before the row: keep the neighbour's low bytes
    g_word = dst.Load(base & ~3u);
    uint pos = base;
    for (uint b = 0; b < n_blocks; b++) {
        float amax = 0.0f;
        for (uint j = 0; j < 32u; j++) {
            amax = max(amax, abs(LOAD_F32(src, i_src_row + b * 32u + j)));
        }
        const float d  = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint  dh = f32_to_f16_rne(d);
        put_byte(pos, end, dh);
        pos++;
        put_byte(pos, end, dh >> 8);
        pos++;
        for (uint j2 = 0; j2 < 32u; j2++) {
            const float x = LOAD_F32(src, i_src_row + b * 32u + j2) * id;
            const int   q = (int) (sign(x) * floor(abs(x) + 0.5f));   // roundf: halves away from zero
            put_byte(pos, end, (uint) q);
            pos++;
        }
    }
}
