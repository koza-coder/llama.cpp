#include "common.hlsli"

// LIGHTNING_INDEXER (f32 q/w, f32 or f16 k, f16 mask, f32 dst):
//   dst[s, t, ik] = sum over heads of max(dot(q[s, t, h], k[s, ik]), 0) * w[s, t, h] + mask[s, t, ik]
//
// The weights are prescaled by the caller, so there is nothing to normalise afterwards and every
// destination element is independent: one thread each. The mask is always f16, so this kernel
// always needs USE_16BIT. defines: K_F16

RWByteAddressBuffer q_buf : register(u0);   // {n_embd, n_head, n_tokens, n_stream}
RWByteAddressBuffer k_buf : register(u1);   // {n_embd, *, n_kv, n_stream}
RWByteAddressBuffer w_buf : register(u2);   // {n_head, n_tokens, *, n_stream}
RWByteAddressBuffer m_buf : register(u3);   // {n_kv, n_tokens, *, n_stream or 1}, f16
RWByteAddressBuffer dst   : register(u4);   // {n_kv, n_tokens, *, n_stream}

cbuffer Params : register(b0) {
    uint offset_q;
    uint offset_k;
    uint offset_w;
    uint offset_m;
    uint offset_dst;
    uint stride_q1;   // per head
    uint stride_q2;   // per token
    uint stride_q3;   // per stream
    uint stride_k2;   // per kv position, in k elements
    uint stride_k3;
    uint stride_w1;
    uint stride_w3;
    uint stride_m1;   // in f16 elements
    uint stride_m3;
    uint stride_d1;
    uint stride_d3;
    uint n_embd;
    uint n_head;
    uint n_tokens;
    uint n_kv;
    uint m_streams;   // mask ne[3], the stream axis the mask is broadcast over
    uint ne;
    uint nwg_x;
};

#ifdef K_F16
#define LOAD_K(i) LOAD_F16(k_buf, (i))
#else
#define LOAD_K(i) LOAD_F32(k_buf, (i))
#endif

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint s  = i / (n_tokens * n_kv);
    const uint r  = i % (n_tokens * n_kv);
    const uint t  = r / n_kv;
    const uint ik = r % n_kv;

    const uint k_row = offset_k + ik * stride_k2 + s * stride_k3;
    const uint w_row = offset_w + t * stride_w1 + s * stride_w3;
    const uint q_pos = offset_q + t * stride_q2 + s * stride_q3;

    float score = 0.0f;
    for (uint h = 0; h < n_head; h++) {
        const uint q_row = q_pos + h * stride_q1;
        float qk = 0.0f;
        for (uint e = 0; e < n_embd; e++) {
            qk += LOAD_F32(q_buf, q_row + e) * LOAD_K(k_row + e);
        }
        score += max(qk, 0.0f) * LOAD_F32(w_buf, w_row + h);
    }
    const uint m_row = offset_m + t * stride_m1 + (s % m_streams) * stride_m3;
    STORE_F32(dst, offset_dst + t * stride_d1 + s * stride_d3 + ik, score + LOAD_F16(m_buf, m_row + ik));
}
