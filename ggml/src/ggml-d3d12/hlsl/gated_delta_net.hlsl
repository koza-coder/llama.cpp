#include "common.hlsli"

// GATED_DELTA_NET (f32), following the CPU reference. Every state row j of every (sequence, head) evolves on its
// own, so there is one thread per row: for each token t in t0..t1-1
//   row *= exp(g)  (scalar g, or per element for KDA)
//   delta = (v[j] - dot(row, k)) * beta;  row += delta * k;  out[t, j] = dot(row, q) * scale
// dst holds the attention output [S_v * H * n_tokens * n_seqs] followed by K state snapshots
// [S_v * S_v * H * n_seqs each]. With K == 1 a dispatch may cover a token range: the state continues from the
// dst state area when t0 > 0. defines: S_V (multiple of 4), KDA

RWByteAddressBuffer q_buf : register(u0);
RWByteAddressBuffer k_buf : register(u1);
RWByteAddressBuffer v_buf : register(u2);
RWByteAddressBuffer g_buf : register(u3);
RWByteAddressBuffer b_buf : register(u4);
RWByteAddressBuffer s_buf : register(u5);
RWByteAddressBuffer dst   : register(u6);

cbuffer Params : register(b0) {
    uint offset_q;
    uint offset_k;
    uint offset_v;
    uint offset_g;
    uint offset_b;
    uint offset_s;
    uint offset_dst;

    uint stride_q1;
    uint stride_q2;
    uint stride_q3;
    uint stride_k1;
    uint stride_k2;
    uint stride_k3;
    uint stride_v1;
    uint stride_v2;
    uint stride_v3;
    uint stride_g1;
    uint stride_g2;
    uint stride_g3;
    uint stride_b1;
    uint stride_b2;
    uint stride_b3;

    uint n_head;
    uint n_tokens;
    uint neq1;
    uint nek1;
    uint rq3;
    uint rk3;
    uint state_seq_stride;   // src_state->nb[3] in elements
    uint n_snap;             // K
    uint t0;
    uint t1;
    uint n_rows;             // n_seqs * n_head * S_V
    float scale;

    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint r = flat_index(id, nwg_x);
    if (r >= n_rows) {
        return;
    }
    const uint iv3 = r / (n_head * S_V);
    const uint iv1 = (r / S_V) % n_head;
    const uint j   = r % S_V;
    const uint iq1 = iv1 % neq1;
    const uint ik1 = iv1 % nek1;
    const uint iq3 = iv3 / rq3;
    const uint ik3 = iv3 / rk3;

    const uint attn_elems = S_V * n_head * n_tokens * (n_rows / (n_head * S_V));
    const uint snap_elems = S_V * S_V * n_head * (n_rows / (n_head * S_V));
    const uint state_row  = (iv3 * n_head + iv1) * S_V * S_V + j * S_V;

    float4 row[S_V / 4];
    if (t0 == 0) {
        const uint s_in = offset_s + iv3 * state_seq_stride + iv1 * S_V * S_V + j * S_V;
        for (uint a = 0; a < S_V / 4; a++) {
            row[a] = asfloat(s_buf.Load4((s_in + 4 * a) * 4));
        }
    } else {
        // K == 1: continue from the state written by the previous token range
        for (uint a2 = 0; a2 < S_V / 4; a2++) {
            row[a2] = asfloat(dst.Load4((offset_dst + attn_elems + state_row + 4 * a2) * 4));
        }
    }

    for (uint t = t0; t < t1; t++) {
        const uint q_base = offset_q + iq3 * stride_q3 + t * stride_q2 + iq1 * stride_q1;
        const uint k_base = offset_k + ik3 * stride_k3 + t * stride_k2 + ik1 * stride_k1;
        const uint g_base = offset_g + iv3 * stride_g3 + t * stride_g2 + iv1 * stride_g1;
        const float v    = LOAD_F32(v_buf, offset_v + iv3 * stride_v3 + t * stride_v2 + iv1 * stride_v1 + j);
        const float beta = LOAD_F32(b_buf, offset_b + iv3 * stride_b3 + t * stride_b2 + iv1 * stride_b1);

#if defined(KDA)
        for (uint a3 = 0; a3 < S_V / 4; a3++) {
            row[a3] *= exp(asfloat(g_buf.Load4((g_base + 4 * a3) * 4)));
        }
#else
        const float decay = exp(LOAD_F32(g_buf, g_base));
        for (uint a3 = 0; a3 < S_V / 4; a3++) {
            row[a3] *= decay;
        }
#endif
        float sum = 0.0f;
        for (uint a4 = 0; a4 < S_V / 4; a4++) {
            sum += dot(row[a4], asfloat(k_buf.Load4((k_base + 4 * a4) * 4)));
        }
        const float delta = (v - sum) * beta;
        float out_v = 0.0f;
        for (uint a5 = 0; a5 < S_V / 4; a5++) {
            row[a5] += delta * asfloat(k_buf.Load4((k_base + 4 * a5) * 4));
            out_v += dot(row[a5], asfloat(q_buf.Load4((q_base + 4 * a5) * 4)));
        }
        STORE_F32(dst, offset_dst + (iv3 * n_tokens * n_head + t * n_head + iv1) * S_V + j, out_v * scale);

        if (n_snap > 1) {
            const int slot = (int) n_tokens - 1 - (int) t;
            if (slot >= 0 && slot < (int) n_snap) {
                const uint o = offset_dst + attn_elems + (uint) slot * snap_elems + state_row;
                for (uint a6 = 0; a6 < S_V; a6++) {
                    STORE_F32(dst, o + a6, row[a6 / 4][a6 % 4]);
                }
            }
        }
    }

    if (n_snap == 1) {
        const uint o1 = offset_dst + attn_elems + state_row;
        for (uint a7 = 0; a7 < S_V; a7++) {
            STORE_F32(dst, o1 + a7, row[a7 / 4][a7 % 4]);
        }
    }
}
