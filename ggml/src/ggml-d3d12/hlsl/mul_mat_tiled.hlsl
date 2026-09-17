#include "common.hlsli"

// Tiled matrix product for the prompt path: dst[row, col] = sum_k src0[row, k] * src1[col, k].
//
// The matrix-vector kernel re-reads src0 once per group of 4 columns, which is the right shape for
// single token decoding and the wrong one for a long prompt. Here a workgroup owns a TILE_M x TILE_N
// tile of dst and walks k in steps of TILE_K: each step dequantizes TILE_M x TILE_K weights into
// groupshared once and every one of them is then used TILE_N times from there. Accumulators live in
// registers, so there is no reduction tree and no groupshared traffic per output.
//
// TILE_K is 32: one Q4_0/Q8_0 block, and one sub-block of a K-quant super-block, so a tile never
// straddles a scale boundary.
//
// The dequantization mirrors dequant_row.hlsli, which the matrix-vector kernel uses and which the op
// suite covers; only the loop shape differs (a contiguous run of 32 instead of a lane-strided walk).
//
// defines: SRC0_{F32,F16,Q4_0,Q8_0,Q4_K,Q6_K}, SRC1_F16 (f16 columns instead of f32)
// src0 offsets/strides are in elements for float types and in blocks for quant types.

#define TILE_M 64
#define TILE_N 32
#define TILE_K 32

// 256 threads as a 16 x 16 grid; each thread owns 4 rows and 2 columns of the output tile
#define TH_X 16
#define TH_Y 16
#define REG_M (TILE_M / TH_Y)
#define REG_N (TILE_N / TH_X)
#define NTHREADS (TH_X * TH_Y)

RWByteAddressBuffer src1 : register(u0);
RWByteAddressBuffer src0 : register(u1);
RWByteAddressBuffer dst  : register(u2);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_dst;
    uint m;           // rows of src0 / dst
    uint n;           // columns of src1 / dst
    uint k;           // shared dimension
    uint stride_01;   // src0 row stride, elements or blocks
    uint stride_02;
    uint stride_03;
    uint stride_11;   // src1 column stride, elements
    uint stride_12;
    uint stride_13;
    uint ne2;         // dst dim 2
    uint broadcast2;  // src1 dim 2 / src0 dim 2
    uint broadcast3;
    uint n_batches;   // dst ne2 * ne3; the dispatch is rounded up to a 2D grid, so it can overshoot
    uint nwg_x;
};

groupshared float Atile[TILE_K][TILE_M];
groupshared float Btile[TILE_K][TILE_N];

#if defined(SRC1_F16)
float load_src1(uint i) {
    uint bits;
    LOAD_U16_UNALIGNED(src1, i * 2, bits);
    return f16tof32(bits);
}
#else
float load_src1(uint i) {
    return LOAD_F32(src1, i);
}
#endif

// Dequantize the TILE_K elements of one src0 row starting at element k0 into vals[0..TILE_K-1].
// row_base is the row's offset in elements (float types) or blocks (quant types). k0 is a multiple of
// TILE_K, so for the block types it lands exactly on a block or K-quant sub-block boundary.
void load_a_row(uint row_base, uint k0, out float vals[TILE_K]) {
#if defined(SRC0_F32)
    [unroll] for (uint i = 0; i < TILE_K; i += 4) {
        const uint4 w = src0.Load4((row_base + k0 + i) * 4);
        vals[i + 0] = asfloat(w.x);
        vals[i + 1] = asfloat(w.y);
        vals[i + 2] = asfloat(w.z);
        vals[i + 3] = asfloat(w.w);
    }
#elif defined(SRC0_F16)
    [unroll] for (uint i = 0; i < TILE_K; i += 2) {
        uint w;
        LOAD_U32_UNALIGNED(src0, (row_base + k0 + i) * 2, w);
        vals[i + 0] = f16tof32(w & 0xFFFFu);
        vals[i + 1] = f16tof32(w >> 16);
    }
#elif defined(SRC0_Q8_0)
    // block: f16 d, 32 int8
    {
        const uint base = (row_base + k0 / 32) * 34;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        const float d = f16tof32(dbits);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                vals[j * 4 + b] = (float) sbyte_of(q, b) * d;
            }
        }
    }
#elif defined(SRC0_Q4_0)
    // block: f16 d, 16 bytes of nibbles; low nibbles are elements 0..15, high nibbles 16..31
    {
        const uint base = (row_base + k0 / 32) * 18;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base, dbits);
        const float d = f16tof32(dbits);
        [unroll] for (uint j = 0; j < 4; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, base + 2 + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                const uint byte = byte_of(q, b);
                vals[j * 4 + b]      = ((float) (byte & 0xFu) - 8.0f) * d;
                vals[16 + j * 4 + b] = ((float) (byte >> 4) - 8.0f) * d;
            }
        }
    }
#elif defined(SRC0_Q4_K)
    // super-block of 256: f16 d, f16 dmin, 12 bytes of 6-bit scales/mins, 128 bytes of nibbles.
    // sub-block s (32 values): pair s/2 uses bytes 16 + 32*(s/2), low nibbles for even s, high for odd s.
    {
        const uint blk  = k0 / 256;
        const uint s    = (k0 % 256) / 32;
        const uint base = (row_base + blk) * 144;
        uint w;
        LOAD_U32_UNALIGNED(src0, base, w);
        const float d    = f16tof32(w & 0xFFFFu);
        const float dmin = f16tof32(w >> 16);
        uint sc0, sc1, sc2;
        LOAD_U32_UNALIGNED(src0, base + 4, sc0);
        LOAD_U32_UNALIGNED(src0, base + 8, sc1);
        LOAD_U32_UNALIGNED(src0, base + 12, sc2);
        uint sc, mn;
        if (s < 4) {
            sc = byte_of(sc0, s) & 63u;
            mn = byte_of(sc1, s) & 63u;
        } else {
            sc = (byte_of(sc2, s - 4) & 0xFu) | ((byte_of(sc0, s - 4) >> 6) << 4);
            mn = (byte_of(sc2, s - 4) >> 4) | ((byte_of(sc1, s - 4) >> 6) << 4);
        }
        const float dl    = d * (float) sc;
        const float ml    = dmin * (float) mn;
        const uint  shift = (s & 1u) * 4u;
        const uint  qbase = base + 16 + 32 * (s / 2);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint q;
            LOAD_U32_UNALIGNED(src0, qbase + 4 * j, q);
            [unroll] for (uint b = 0; b < 4; b++) {
                vals[j * 4 + b] = dl * (float) ((byte_of(q, b) >> shift) & 0xFu) - ml;
            }
        }
    }
#elif defined(SRC0_Q6_K)
    // super-block of 256: 128 bytes ql, 64 bytes qh, 16 int8 scales, f16 d.
    // sub-block s: half h = s/4 (128 values), t = s%4 selects the quarter inside the half.
    // The 16 scales are one per 16 values, so a sub-block of 32 uses two of them.
    {
        const uint blk  = k0 / 256;
        const uint s    = (k0 % 256) / 32;
        const uint h    = s / 4;
        const uint t    = s % 4;
        const uint base = (row_base + blk) * 210;
        uint dbits;
        LOAD_U16_UNALIGNED(src0, base + 208, dbits);
        const float d = f16tof32(dbits);
        const uint  ql_base = base + 64 * h + 32 * (t & 1u);
        const uint  qh_base = base + 128 + 32 * h;
        const uint  sc_base = base + 192 + 8 * h + 2 * t;
        const uint  lshift  = (t >> 1) * 4u;
        const uint  hshift  = t * 2u;
        uint scw;
        LOAD_U32_UNALIGNED(src0, sc_base, scw);
        const float d0 = d * (float) sbyte_of(scw, 0);
        const float d1 = d * (float) sbyte_of(scw, 1);
        [unroll] for (uint j = 0; j < 8; j++) {
            uint ql, qh;
            LOAD_U32_UNALIGNED(src0, ql_base + 4 * j, ql);
            LOAD_U32_UNALIGNED(src0, qh_base + 4 * j, qh);
            const float dsc = (j < 4) ? d0 : d1;
            [unroll] for (uint b = 0; b < 4; b++) {
                const int q = (int) (((byte_of(ql, b) >> lshift) & 0xFu) | (((byte_of(qh, b) >> hshift) & 3u) << 4)) - 32;
                vals[j * 4 + b] = dsc * (float) q;
            }
        }
    }
#endif
}

[numthreads(NTHREADS, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid       = gtid.x;
    const uint tx        = tid % TH_X;          // column group
    const uint ty        = tid / TH_X;          // row group
    const uint wg_linear = gid.y * nwg_x + gid.x;

    const uint tiles_m = (m + TILE_M - 1) / TILE_M;
    const uint tiles_n = (n + TILE_N - 1) / TILE_N;
    const uint tile_id = wg_linear % (tiles_m * tiles_n);
    const uint batch   = wg_linear / (tiles_m * tiles_n);

    if (batch >= n_batches) {
        return;
    }
    const uint batches2  = ne2;
    const uint dst2_idx  = batch % batches2;
    const uint dst3_idx  = batch / batches2;
    const uint src02_idx = dst2_idx / broadcast2;
    const uint src03_idx = dst3_idx / broadcast3;

    const uint row0 = (tile_id % tiles_m) * TILE_M;
    const uint col0 = (tile_id / tiles_m) * TILE_N;

    const uint a_batch = offset_src0 + src03_idx * stride_03 + src02_idx * stride_02;
    const uint b_batch = offset_src1 + dst3_idx * stride_13 + dst2_idx * stride_12;

    float acc[REG_M][REG_N];
    [unroll] for (uint im = 0; im < REG_M; im++) {
        [unroll] for (uint jn = 0; jn < REG_N; jn++) {
            acc[im][jn] = 0.0f;
        }
    }

    // staging: threads 0..TILE_M-1 take one src0 row each, so a row is dequantized exactly once;
    // all NTHREADS threads share the src1 tile, 8 threads per column, 4 values each
    const uint b_col  = tid / 8;
    const uint b_part = tid % 8;

    for (uint kt = 0; kt < k; kt += TILE_K) {
        if (tid < TILE_M) {
            const uint r = row0 + tid;
            float vals[TILE_K];
            if (r < m) {
                load_a_row(a_batch + r * stride_01, kt, vals);
            } else {
                [unroll] for (uint z = 0; z < TILE_K; z++) {
                    vals[z] = 0.0f;
                }
            }
            [unroll] for (uint ki = 0; ki < TILE_K; ki++) {
                Atile[ki][tid] = vals[ki];
            }
        }
        {
            const uint c = col0 + b_col;
            [unroll] for (uint i = 0; i < 4; i++) {
                const uint ki = b_part * 4 + i;
                Btile[ki][b_col] = (c < n) ? load_src1(b_batch + c * stride_11 + kt + ki) : 0.0f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint kk = 0; kk < TILE_K; kk++) {
            float a[REG_M], b[REG_N];
            [unroll] for (uint im2 = 0; im2 < REG_M; im2++) {
                a[im2] = Atile[kk][ty * REG_M + im2];
            }
            [unroll] for (uint in2 = 0; in2 < REG_N; in2++) {
                b[in2] = Btile[kk][tx * REG_N + in2];
            }
            [unroll] for (uint im3 = 0; im3 < REG_M; im3++) {
                [unroll] for (uint in3 = 0; in3 < REG_N; in3++) {
                    acc[im3][in3] += a[im3] * b[in3];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const uint dst_base = offset_dst + dst3_idx * (m * n * batches2) + dst2_idx * (m * n);
    [unroll] for (uint im4 = 0; im4 < REG_M; im4++) {
        const uint r = row0 + ty * REG_M + im4;
        if (r >= m) {
            continue;
        }
        [unroll] for (uint in4 = 0; in4 < REG_N; in4++) {
            const uint c = col0 + tx * REG_N + in4;
            if (c < n) {
                STORE_F32(dst, dst_base + c * m + r, acc[im4][in4]);
            }
        }
    }
}
