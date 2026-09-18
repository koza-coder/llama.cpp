#include "common.hlsli"

// GATED_LINEAR_ATTN (f32), following the CPU reference:
//   state = g * state + (k @ v)
//   dst   = (q * scale) @ state
// recursive through the tokens of each sequence.
//
// Same decomposition as rwkv_wkv6.hlsl: one thread per (sequence, head, column j) of the head's
// state matrix. All tensors are contiguous; dst holds C*T outputs followed by the states.

RWByteAddressBuffer k_buf : register(u0);   // {head_size, HEADS, T}
RWByteAddressBuffer v_buf : register(u1);
RWByteAddressBuffer q_buf : register(u2);
RWByteAddressBuffer g_buf : register(u3);
RWByteAddressBuffer s_buf : register(u4);   // incoming state {head_size*C, n_seqs}
RWByteAddressBuffer dst   : register(u5);

cbuffer Params : register(b0) {
    uint offset_k;
    uint offset_v;
    uint offset_q;
    uint offset_g;
    uint offset_s;
    uint offset_dst;
    uint cc;          // C = HEADS * head_size
    uint hs;          // head_size
    uint tps;         // tokens per sequence, T / n_seqs
    uint s_off;       // C * T
    uint scale_bits;  // the scale applied to q, as raw float bits
    uint n_jobs;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint job = flat_index(gid, nwg_x);
    if (job >= n_jobs) {
        return;
    }
    const uint seq = job / cc;
    const uint rem = job % cc;
    const uint h   = rem / hs;
    const uint j   = rem % hs;

    const float scale = asfloat(scale_bits);

    const uint st_head = seq * hs * cc + h * hs * hs + j;
    const uint st_cur  = offset_dst + s_off + st_head;
    const uint st_in   = offset_s + st_head;

    for (uint tl = 0; tl < tps; tl++) {
        const uint t_h = (seq * tps + tl) * cc + h * hs;
        const float v_val = LOAD_F32(v_buf, offset_v + t_h + j);

        float acc = 0.0f;
        for (uint i = 0; i < hs; i++) {
            const float k_val = LOAD_F32(k_buf, offset_k + t_h + i);
            const float q_val = LOAD_F32(q_buf, offset_q + t_h + i) * scale;
            const float g_val = LOAD_F32(g_buf, offset_g + t_h + i);

            const uint  slot = i * hs;
            const float prev = tl == 0u ? LOAD_F32(s_buf, st_in + slot) : LOAD_F32(dst, st_cur + slot);
            const float cur  = prev * g_val + v_val * k_val;

            acc += cur * q_val;
            STORE_F32(dst, st_cur + slot, cur);
        }
        STORE_F32(dst, offset_dst + t_h + j, acc);
    }
}
