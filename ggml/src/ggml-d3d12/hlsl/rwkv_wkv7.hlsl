#include "common.hlsli"

// RWKV_WKV7 (f32), following the CPU reference. For every token and every state row i:
//   sa       = dot(a, state_prev[i])
//   state[i] = state_prev[i] * w + k * v[i] + sa * b
//   dst[i]   = dot(state[i], r)
//
// Unlike wkv6 the row, not the column, is the unit that stays together - sa is a reduction over a
// whole row - so one thread owns one (sequence, head, row i) and the token loop stays sequential
// inside it. All tensors are contiguous; dst holds C*T outputs followed by the per-sequence states.

RWByteAddressBuffer r_buf : register(u0);   // {head_size, HEADS, T}
RWByteAddressBuffer w_buf : register(u1);
RWByteAddressBuffer k_buf : register(u2);
RWByteAddressBuffer v_buf : register(u3);
RWByteAddressBuffer a_buf : register(u4);
RWByteAddressBuffer b_buf : register(u5);
RWByteAddressBuffer s_buf : register(u6);   // incoming state {head_size*C, n_seqs}
RWByteAddressBuffer dst   : register(u7);

cbuffer Params : register(b0) {
    uint offset_r;
    uint offset_w;
    uint offset_k;
    uint offset_v;
    uint offset_a;
    uint offset_b;
    uint offset_s;
    uint offset_dst;
    uint cc;          // C = HEADS * head_size
    uint hs;          // head_size
    uint tps;         // tokens per sequence, T / n_seqs
    uint s_off;       // C * T
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
    const uint i   = rem % hs;

    // row i of this head's head_size x head_size state matrix
    const uint st_head = seq * hs * cc + h * hs * hs + i * hs;
    const uint st_cur  = offset_dst + s_off + st_head;
    const uint st_in   = offset_s + st_head;

    for (uint tl = 0; tl < tps; tl++) {
        const uint t_h = (seq * tps + tl) * cc + h * hs;
        const float v_val = LOAD_F32(v_buf, offset_v + t_h + i);

        float sa = 0.0f;
        for (uint j = 0; j < hs; j++) {
            const float prev = tl == 0u ? LOAD_F32(s_buf, st_in + j) : LOAD_F32(dst, st_cur + j);
            sa += LOAD_F32(a_buf, offset_a + t_h + j) * prev;
        }

        float result = 0.0f;
        for (uint j = 0; j < hs; j++) {
            const float prev = tl == 0u ? LOAD_F32(s_buf, st_in + j) : LOAD_F32(dst, st_cur + j);
            const float cur  = prev * LOAD_F32(w_buf, offset_w + t_h + j)
                             + v_val * LOAD_F32(k_buf, offset_k + t_h + j)
                             + sa * LOAD_F32(b_buf, offset_b + t_h + j);
            STORE_F32(dst, st_cur + j, cur);
            result += cur * LOAD_F32(r_buf, offset_r + t_h + j);
        }
        STORE_F32(dst, offset_dst + t_h + i, result);
    }
}
