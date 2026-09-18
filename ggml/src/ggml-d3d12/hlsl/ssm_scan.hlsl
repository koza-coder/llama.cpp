#include "common.hlsli"

// SSM_SCAN (f32): the selective state space recurrence behind Mamba-1 and Mamba-2.
//
//   s[i0] = s_prev[i0] * dA + B[i0] * (x * softplus(dt))
//   y     = dot(s, C)
//
// Mamba-2 (A is {1, n_head}) has one decay per head, Mamba-1 (A is {d_state, n_head}) one per
// state. The token loop carries state and so must stay sequential, but every (sequence, head, dim)
// triple owns a private slice of d_state values and is independent of every other, so that triple
// is the unit of parallelism: one thread each.
//
// The first token reads the incoming state from src0 at ids[seq]; later tokens read back what this
// same thread wrote into dst, which is what the CPU does when it sets s0 = s at the end of a token.

RWByteAddressBuffer src_s0  : register(u0);   // s  {d_state, dim, n_head, n_seqs}
RWByteAddressBuffer src_x   : register(u1);   // x  {dim, n_head, n_tokens, n_seqs}
RWByteAddressBuffer src_dt  : register(u2);   // dt {n_head, n_tokens, n_seqs}
RWByteAddressBuffer src_A   : register(u3);   // A  {d_state, n_head} or {1, n_head}
RWByteAddressBuffer src_B   : register(u4);   // B  {d_state, n_group, n_tokens, n_seqs}
RWByteAddressBuffer src_C   : register(u5);   // C  {d_state, n_group, n_tokens, n_seqs}
RWByteAddressBuffer src_ids : register(u6);   // ids {n_seqs}, i32
RWByteAddressBuffer dst     : register(u7);

cbuffer Params : register(b0) {
    uint offset_s0;
    uint offset_x;
    uint offset_dt;
    uint offset_A;
    uint offset_B;
    uint offset_C;
    uint offset_ids;
    uint offset_dst;
    uint stride_s3;    // src0 nb[3], the per-sequence state stride
    uint stride_x2;
    uint stride_x3;
    uint stride_dt1;
    uint stride_dt2;
    uint stride_B2;
    uint stride_B3;
    uint stride_C2;
    uint stride_C3;
    uint nc;           // d_state
    uint nr;           // dim
    uint nh;           // n_head
    uint ng;           // n_group
    uint nt;           // tokens per sequence
    uint ns;           // sequences
    uint kk;           // K, the number of state snapshots
    uint s_off;        // element offset of the state area inside dst
    uint mamba2;       // 1 when A has one element per head
    uint n_jobs;       // ns * nh * nr
    uint nwg_x;
};

float softplus(float x) {
    return x > 20.0f ? x : log(1.0f + exp(x));
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint job = flat_index(gid, nwg_x);
    if (job >= n_jobs) {
        return;
    }
    const uint i3 = job / (nh * nr);         // sequence
    const uint ii = job % (nh * nr);         // dim within head, flattened as i1 + h * nr
    const uint h  = ii / nr;
    const uint g  = h / (nh / ng);           // repeat_interleave of the group index

    const uint seq_id  = (uint) LOAD_I32(src_ids, offset_ids + i3);
    const uint s_in    = offset_s0 + seq_id * stride_s3 + ii * nc;
    const uint s_out   = offset_dst + s_off + i3 * stride_s3 + ii * nc;

    for (uint i2 = 0; i2 < nt; i2++) {
        const float dt_sp = softplus(LOAD_F32(src_dt, offset_dt + i2 * stride_dt1 + i3 * stride_dt2 + h));
        const float x_dt  = LOAD_F32(src_x, offset_x + i2 * stride_x2 + i3 * stride_x3 + ii) * dt_sp;

        const uint b_row = offset_B + i2 * stride_B2 + i3 * stride_B3 + g * nc;
        const uint c_row = offset_C + i2 * stride_C2 + i3 * stride_C3 + g * nc;

        // Mamba-2 hoists the decay out of the state loop; Mamba-1 has one per state
        const float dA_head = mamba2 != 0u ? exp(dt_sp * LOAD_F32(src_A, offset_A + h)) : 0.0f;

        float sumf = 0.0f;
        for (uint i0 = 0; i0 < nc; i0++) {
            const float prev = i2 == 0u ? LOAD_F32(src_s0, s_in + i0) : LOAD_F32(dst, s_out + i0);
            const float dA   = mamba2 != 0u ? dA_head
                                            : exp(dt_sp * LOAD_F32(src_A, offset_A + i0 + h * nc));
            const float state = prev * dA + LOAD_F32(src_B, b_row + i0) * x_dt;
            sumf += state * LOAD_F32(src_C, c_row + i0);
            STORE_F32(dst, s_out + i0, state);
        }
        STORE_F32(dst, offset_dst + i2 * (nh * nr) + i3 * (nt * nh * nr) + ii, sumf);

        // state snapshots for the last K - 1 tokens, kept for speculative decoding
        const uint slot = nt - 1u - i2;
        if (kk > 1u && slot > 0u && slot < kk) {
            const uint snap = offset_dst + s_off + (slot * ns + i3) * stride_s3 + ii * nc;
            for (uint j = 0; j < nc; j++) {
                STORE_F32(dst, snap + j, LOAD_F32(dst, s_out + j));
            }
        }
    }
}
