#include "common.hlsli"

// RWKV_WKV6 (f32), following the CPU reference:
//   dst   = r @ (time_faaaa * (k @ v) + state)
//   state = time_decay * state + (k @ v)
// recursive through the tokens of each sequence.
//
// Every (sequence, head, column j) owns one column of the head's state matrix and never touches
// another, so that triple is the unit of parallelism: one thread each. The token loop stays
// sequential inside the thread, and the i loop both accumulates this token's output and advances
// the column, which is why no separate zeroing pass over dst is needed.
//
// All tensors are contiguous; dst holds C*T outputs followed by the per-sequence states.

RWByteAddressBuffer k_buf : register(u0);   // {head_size, HEADS, T}
RWByteAddressBuffer v_buf : register(u1);   // {head_size, HEADS, T}
RWByteAddressBuffer r_buf : register(u2);   // {head_size, HEADS, T}
RWByteAddressBuffer tf_buf : register(u3);  // time_faaaa {head_size, HEADS}
RWByteAddressBuffer td_buf : register(u4);  // time_decay {head_size, HEADS, T}
RWByteAddressBuffer s_buf : register(u5);   // incoming state {head_size*C, n_seqs}
RWByteAddressBuffer dst   : register(u6);

cbuffer Params : register(b0) {
    uint offset_k;
    uint offset_v;
    uint offset_r;
    uint offset_tf;
    uint offset_td;
    uint offset_s;
    uint offset_dst;
    uint cc;          // C = HEADS * head_size
    uint hs;          // head_size
    uint tps;         // tokens per sequence, T / n_seqs
    uint s_off;       // C * T, the element offset of the state area inside dst
    uint n_jobs;      // n_seqs * HEADS * head_size
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint job = flat_index(gid, nwg_x);
    if (job >= n_jobs) {
        return;
    }
    const uint per_seq = cc;                 // HEADS * head_size
    const uint seq = job / per_seq;
    const uint rem = job % per_seq;
    const uint h   = rem / hs;
    const uint j   = rem % hs;

    // the state matrix of one head is head_size x head_size; this thread owns column j of it
    const uint st_head = seq * hs * cc + h * hs * hs + j;
    const uint st_cur  = offset_dst + s_off + st_head;
    const uint st_in   = offset_s + st_head;

    for (uint tl = 0; tl < tps; tl++) {
        const uint t_h = (seq * tps + tl) * cc + h * hs;
        const float v_val = LOAD_F32(v_buf, offset_v + t_h + j);

        float acc = 0.0f;
        for (uint i = 0; i < hs; i++) {
            const float k_val = LOAD_F32(k_buf, offset_k + t_h + i);
            const float r_val = LOAD_F32(r_buf, offset_r + t_h + i);
            const float tf    = LOAD_F32(tf_buf, offset_tf + h * hs + i);
            const float td    = LOAD_F32(td_buf, offset_td + t_h + i);   // RWKV v6: per token

            const uint  slot = i * hs;
            const float prev = tl == 0u ? LOAD_F32(s_buf, st_in + slot) : LOAD_F32(dst, st_cur + slot);
            const float kv   = v_val * k_val;

            acc += (kv * tf + prev) * r_val;
            STORE_F32(dst, st_cur + slot, prev * td + kv);
        }
        STORE_F32(dst, offset_dst + t_h + j, acc);
    }
}
