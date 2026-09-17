#include "common.hlsli"

// dst[row, col] = sum_k src0[row, k] * src1[col, k] (+ addend[row, col]) for up to MAX_COLS columns per dispatch.
// One dispatch can serve N_MATS (1..3) matrices that share src1 (q/k/v projections, ffn up/gate): every
// matrix gets its own range of workgroups starting at wg_start_M. A workgroup handles ROWS = WG_SIZE / TPR
// rows: TPR threads share one row and stride over k in units (4 elements for float types, one block or
// sub-block of 32 for quant types). The reduction is a groupshared tree inside each TPR block, so it does
// not depend on the wave size.
// defines: MMID (indirect per-expert product, see ggml_mul_mat_id), SRC0_{F32,F16,BF16,MXFP4,IQ2_S,IQ3_S,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,IQ4_NL,Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,IQ4_XS}, TPR (power of two, 1..WG_SIZE), N_MATS (1..3), ONE_COL (single-column
//          products: no per-element column loop),
//          SRC1_F16 (f16 columns instead of f32),
//          FUSE_ADD (per-matrix optional addend, broadcast over dims 1..3 like ggml_add)
// src0 offsets/strides are in elements for float types and in blocks for quant types.

#define MAX_COLS 4

#define ROWS (WG_SIZE / TPR)

RWByteAddressBuffer src1   : register(u0);
RWByteAddressBuffer src0_0 : register(u1);
RWByteAddressBuffer dst_0  : register(u2);
RWByteAddressBuffer add_0  : register(u3);
#if defined(MMID)
// MUL_MAT_ID: one expert matrix per id, N_MATS is always 1 and the add slot is unused
RWByteAddressBuffer ids    : register(u4);
#endif
#if N_MATS >= 2
RWByteAddressBuffer src0_1 : register(u4);
RWByteAddressBuffer dst_1  : register(u5);
RWByteAddressBuffer add_1  : register(u6);
#endif
#if N_MATS >= 3
RWByteAddressBuffer src0_2 : register(u7);
RWByteAddressBuffer dst_2  : register(u8);
RWByteAddressBuffer add_2  : register(u9);
#endif

#define MAT_PARAMS(M) \
    uint offset_src0_##M; \
    uint offset_dst_##M; \
    uint m_##M; \
    uint stride_01_##M; \
    uint stride_02_##M; \
    uint stride_03_##M; \
    uint add_flag_##M; \
    uint offset_add_##M; \
    uint add_ne1_##M; \
    uint add_ne2_##M; \
    uint add_ne3_##M; \
    uint add_s1_##M; \
    uint add_s2_##M; \
    uint add_s3_##M;

cbuffer Params : register(b0) {
    uint offset_src1;
    uint n;
    uint k;
    uint stride_11;
    uint stride_12;
    uint stride_13;
    uint bs02;
    uint bs03;
    uint broadcast2;
    uint broadcast3;
    uint col0;        // first column of this chunk; a dispatch covers columns col0 .. col0 + MAX_COLS - 1
    uint wg_start_1;  // first workgroup of matrix 1 (unused when N_MATS < 2)
    uint wg_start_2;  // first workgroup of matrix 2 (unused when N_MATS < 3)
    MAT_PARAMS(0)
    MAT_PARAMS(1)
    MAT_PARAMS(2)
#if defined(MMID)
    uint offset_ids;
    uint ids_s1;      // ids->nb[1] in elements
    uint n_used;      // ids->ne[0]
    uint b_ne1;       // src1->ne[1], the id slot is taken modulo this
    uint dst_s1;      // dst->nb[1] in elements
    uint dst_s2;      // dst->nb[2] in elements
#endif
    uint nwg_x;
};

#if TPR > 1
groupshared float partial[MAX_COLS][WG_SIZE];
#endif

// src1 columns are f32 unless SRC1_F16 is set (the vision tower multiplies two f16 tensors)
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

// accumulate one value of src0 (already dequantized) against every column of src1
#if defined(ONE_COL)
#define ACC(a, kidx) { acc[0] += (a) * load_src1(src1_base[0] + (kidx)); }
#else
#define ACC(a, kidx) { \
    [unroll] for (uint _c = 0; _c < MAX_COLS; _c++) { \
        if (_c < ncols) { acc[_c] += (a) * load_src1(src1_base[_c] + (kidx)); } \
    } \
}
#endif

#include "dequant_row.hlsli"

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid       = gtid.x;
    const uint lane      = tid % TPR;
    const uint wg_linear = gid.y * nwg_x + gid.x;

    // which matrix does this workgroup belong to (uniform per workgroup)
    uint mat = 0, wg_local = wg_linear;
#if N_MATS >= 2
    if (wg_linear >= wg_start_1) { mat = 1; wg_local = wg_linear - wg_start_1; }
#endif
#if N_MATS >= 3
    if (wg_linear >= wg_start_2) { mat = 2; wg_local = wg_linear - wg_start_2; }
#endif
    uint mrows, o_src0, o_dst, s01, s02, s03, add_flag, o_add, add_ne1, add_ne2, add_ne3, add_s1, add_s2, add_s3;
    SELECT_MAT(0);
#if N_MATS >= 2
    if (mat == 1) SELECT_MAT(1);
#endif
#if N_MATS >= 3
    if (mat == 2) SELECT_MAT(2);
#endif

    const uint row_groups = (mrows + ROWS - 1) / ROWS;
    const uint row        = (wg_local % row_groups) * ROWS + tid / TPR;
#if defined(MMID)
    // one (id slot, token) pair per group of row groups; the id picks the expert matrix
    const uint pair      = wg_local / row_groups;
    const uint id_slot   = pair % n_used;
    const uint token     = pair / n_used;
    const uint expert    = (uint) ids.Load((offset_ids + token * ids_s1 + id_slot) * 4);
    const uint ncols     = 1;
    const uint src0_base = o_src0 + expert * s02 + row * s01;
    uint src1_base[MAX_COLS];
    [unroll] for (uint c = 0; c < MAX_COLS; c++) {
        src1_base[c] = offset_src1 + token * stride_12 + (id_slot % b_ne1) * stride_11;
    }
#else
    const uint batch      = wg_local / row_groups;
    const uint batches2   = bs02 * broadcast2;
    if (batch >= batches2 * bs03 * broadcast3) {
        return;
    }
    const uint dst2_idx  = batch % batches2;
    const uint dst3_idx  = batch / batches2;
    const uint src02_idx = dst2_idx / broadcast2;
    const uint src03_idx = dst3_idx / broadcast3;

#if defined(ONE_COL)
    const uint ncols     = 1;
#else
    const uint ncols     = min(MAX_COLS, n - col0);
#endif
    const uint src0_base = o_src0 + src03_idx * s03 + src02_idx * s02 + row * s01;
    uint src1_base[MAX_COLS];
    [unroll] for (uint c = 0; c < MAX_COLS; c++) {
        src1_base[c] = offset_src1 + dst3_idx * stride_13 + dst2_idx * stride_12 + (col0 + c) * stride_11;
    }
#endif
    float acc[MAX_COLS] = { 0, 0, 0, 0 };

    if (row < mrows) {
        if (mat == 0) {
            dot_row(src0_0, src0_base, lane, ncols, src1_base, acc);
        }
#if N_MATS >= 2
        else if (mat == 1) {
            dot_row(src0_1, src0_base, lane, ncols, src1_base, acc);
        }
#endif
#if N_MATS >= 3
        else {
            dot_row(src0_2, src0_base, lane, ncols, src1_base, acc);
        }
#endif
    }

#if TPR > 1
    // tree reduction inside each TPR block
    [unroll] for (uint c2 = 0; c2 < MAX_COLS; c2++) {
        partial[c2][tid] = acc[c2];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint off = TPR / 2; off > 0; off /= 2) {
        if (lane < off) {
            [unroll] for (uint c3 = 0; c3 < MAX_COLS; c3++) {
                partial[c3][tid] += partial[c3][tid + off];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float4 row_result = float4(partial[0][tid], partial[1][tid], partial[2][tid], partial[3][tid]);
#else
    // one thread per row: no shared memory and no barriers
    const float4 row_result = float4(acc[0], acc[1], acc[2], acc[3]);
#endif
#if defined(MMID)
    if (lane == 0 && row < mrows) {
        STORE_F32(dst_0, o_dst + token * dst_s2 + id_slot * dst_s1 + row, row_result[0]);
    }
    return;
}
#else
    if (lane == 0 && row < mrows) {
        const uint dst_base = o_dst + dst3_idx * (mrows * n * batches2) + dst2_idx * (mrows * n);
        for (uint c4 = 0; c4 < ncols; c4++) {
            float v = row_result[c4];
#if defined(FUSE_ADD)
            if (add_flag != 0) {
                const uint add_idx = o_add + row + ((col0 + c4) % add_ne1) * add_s1 + (dst2_idx % add_ne2) * add_s2 +
                                     (dst3_idx % add_ne3) * add_s3;
                if (mat == 0) {
                    v += LOAD_F32(add_0, add_idx);
                }
#if N_MATS >= 2
                else if (mat == 1) {
                    v += LOAD_F32(add_1, add_idx);
                }
#endif
#if N_MATS >= 3
                else {
                    v += LOAD_F32(add_2, add_idx);
                }
#endif
            }
#endif
            const uint dst_idx = dst_base + (col0 + c4) * mrows + row;
            if (mat == 0) {
                STORE_F32(dst_0, dst_idx, v);
            }
#if N_MATS >= 2
            else if (mat == 1) {
                STORE_F32(dst_1, dst_idx, v);
            }
#endif
#if N_MATS >= 3
            else {
                STORE_F32(dst_2, dst_idx, v);
            }
#endif
        }
    }
}
#endif
