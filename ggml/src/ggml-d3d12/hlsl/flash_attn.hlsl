#include "common.hlsli"

// FLASH_ATTN_EXT with the online softmax of the CPU reference, in two passes so no per-thread loop runs over the
// whole KV length and no n_q x n_kv score matrix is stored:
//   pass 1 (default): one thread per (output row, KV block of BLK entries) writes the block's log-sum-exp of the
//     scores and its softmax-weighted value sum to tmp[(t * n_blocks + b) * (DV + 1)]
//   pass 2 (COMBINE): one thread per output row merges its blocks (and the sink) into dst
// Output row = i3 * n_q * n_head + i1 * n_head + i2; a dispatch covers rows row0 .. row0 + n_rows - 1 and t is
// the row index inside that range.
// defines: DK, DV (head sizes, multiples of 4), K_F16, K_F32 or K_Q8_0, V_F16, V_F32 or V_Q8_0, K_ALIGNED, V_ALIGNED,
//          HAS_MASK (f16 mask), HAS_SINKS, SOFTCAP, COMBINE

RWByteAddressBuffer q_buf : register(u0);
RWByteAddressBuffer k_buf : register(u1);
RWByteAddressBuffer v_buf : register(u2);
RWByteAddressBuffer mask  : register(u3);
RWByteAddressBuffer sinks : register(u4);
RWByteAddressBuffer dst   : register(u5);
RWByteAddressBuffer tmp   : register(u6);

cbuffer Params : register(b0) {
    uint offset_q;
    uint offset_k;
    uint offset_v;
    uint offset_mask;
    uint offset_sinks;
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
    uint stride_m1;
    uint stride_m2;
    uint stride_m3;

    uint mask_ne2;
    uint mask_ne3;
    uint n_q;       // query rows (q->ne[1])
    uint n_head;    // q->ne[2]
    uint n_kv;      // k->ne[1]
    uint rk2;       // q heads per k head
    uint rk3;
    uint rv2;
    uint rv3;

    float scale;    // already divided by logit_softcap when SOFTCAP is set
    float max_bias;
    float logit_softcap;
    float n_head_log2;
    float m0;
    float m1;

    uint blk_size;
    uint n_blocks;
    uint row0;
    uint n_rows;
    uint nwg_x;
};

// K and V rows are read 4 elements at a time (DK and DV are multiples of 4). f16 rows whose byte address is
// a multiple of 4 (K_ALIGNED / V_ALIGNED) take one Load per two elements.
#define F16_LOAD4(buf, i, out) { \
    uint _b0, _b1, _b2, _b3; \
    LOAD_U16_UNALIGNED(buf, (i) * 2, _b0); \
    LOAD_U16_UNALIGNED(buf, (i) * 2 + 2, _b1); \
    LOAD_U16_UNALIGNED(buf, (i) * 2 + 4, _b2); \
    LOAD_U16_UNALIGNED(buf, (i) * 2 + 6, _b3); \
    out = f16tof32(uint4(_b0, _b1, _b2, _b3)); \
}
#define F16_LOAD4_ALIGNED(buf, i, out) { \
    const uint _w0 = (buf).Load((i) * 2); \
    const uint _w1 = (buf).Load((i) * 2 + 4); \
    out = f16tof32(uint4(_w0 & 0xFFFFu, _w0 >> 16, _w1 & 0xFFFFu, _w1 >> 16)); \
}

// q8_0 K/V: offsets and strides are in blocks, element i = block * 32 + index in block; the 4 elements always
// share one block (head sizes are multiples of 32)
float4 load_q8_0_4(RWByteAddressBuffer buf, uint i) {
    const uint byte = (i / 32u) * 34u;
    uint dbits, w;
    LOAD_U16_UNALIGNED(buf, byte, dbits);
    LOAD_U32_UNALIGNED(buf, byte + 2u + i % 32u, w);
    const int4 q = (int4) (uint4(w << 24, w << 16, w << 8, w)) >> 24;
    return f16tof32(dbits) * (float4) q;
}

float4 load_k4(uint i) {
    float4 r;
#if defined(K_Q8_0)
    r = load_q8_0_4(k_buf, i);
#elif defined(K_F16) && defined(K_ALIGNED)
    F16_LOAD4_ALIGNED(k_buf, i, r);
#elif defined(K_F16)
    F16_LOAD4(k_buf, i, r);
#else
    r = asfloat(k_buf.Load4(i * 4));
#endif
    return r;
}

float4 load_v4(uint i) {
    float4 r;
#if defined(V_Q8_0)
    r = load_q8_0_4(v_buf, i);
#elif defined(V_F16) && defined(V_ALIGNED)
    F16_LOAD4_ALIGNED(v_buf, i, r);
#elif defined(V_F16)
    F16_LOAD4(v_buf, i, r);
#else
    r = asfloat(v_buf.Load4(i * 4));
#endif
    return r;
}

#define NEG_INF_SCORE -3.4028235e38f

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint gi = flat_index(id, nwg_x);
#if defined(COMBINE)
    const uint t = gi;
    if (t >= n_rows) {
        return;
    }
    const uint row = row0 + t;
    const uint i2  = row % n_head;

    // same online softmax as pass 1, over block log-sum-exps whose values are already normalized
    float4 acc[DV / 4];
    for (uint d0 = 0; d0 < DV / 4; d0++) {
        acc[d0] = 0.0f;
    }
    float M = NEG_INF_SCORE;
    float S = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        const uint base = (t * n_blocks + b) * (DV + 1);
        const float s = LOAD_F32(tmp, base);
        if (s <= NEG_INF_SCORE) {   // every entry of the block masked
            continue;
        }
        if (s > M) {
            const float ms = exp(M - s);
            M = s;
            for (uint d1 = 0; d1 < DV / 4; d1++) {
                acc[d1] = acc[d1] * ms + asfloat(tmp.Load4((base + 1 + 4 * d1) * 4));
            }
            S = S * ms + 1.0f;
        } else {
            const float vs = exp(s - M);
            for (uint d2 = 0; d2 < DV / 4; d2++) {
                acc[d2] += vs * asfloat(tmp.Load4((base + 1 + 4 * d2) * 4));
            }
            S += vs;
        }
    }

#if defined(HAS_SINKS)
    // the sink is one more logit in the denominator with no value vector
    const float sink = LOAD_F32(sinks, offset_sinks + i2);
    if (sink > M) {
        const float ms = exp(M - sink);
        for (uint d3 = 0; d3 < DV / 4; d3++) {
            acc[d3] *= ms;
        }
        S = S * ms + 1.0f;
    } else {
        S += exp(sink - M);
    }
#endif

    const float inv = S == 0.0f ? 0.0f : 1.0f / S;
    const uint dst_base = offset_dst + row * DV;
    for (uint d4 = 0; d4 < DV; d4++) {
        STORE_F32(dst, dst_base + d4, acc[d4 / 4][d4 % 4] * inv);
    }
#else
    if (gi >= n_rows * n_blocks) {
        return;
    }
    const uint t   = gi / n_blocks;
    const uint b   = gi % n_blocks;
    const uint row = row0 + t;
    const uint i3  = row / (n_q * n_head);
    const uint i1  = (row % (n_q * n_head)) / n_head;
    const uint i2  = row % n_head;

    float4 q[DK / 4];
    const uint q_base = offset_q + i3 * stride_q3 + i2 * stride_q2 + i1 * stride_q1;
    for (uint a = 0; a < DK / 4; a++) {
        q[a] = asfloat(q_buf.Load4((q_base + 4 * a) * 4));
    }
    const uint k_base = offset_k + (i3 / rk3) * stride_k3 + (i2 / rk2) * stride_k2;
    const uint v_base = offset_v + (i3 / rv3) * stride_v3 + (i2 / rv2) * stride_v2;

#if defined(HAS_MASK)
    const uint m_base = offset_mask + (i3 % mask_ne3) * stride_m3 + (i2 % mask_ne2) * stride_m2 + i1 * stride_m1;
    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const float h = (float) i2;
        slope = h < n_head_log2 ? pow(m0, h + 1.0f) : pow(m1, 2.0f * (h - n_head_log2) + 1.0f);
    }
#endif

    float4 acc[DV / 4];
    for (uint d0 = 0; d0 < DV / 4; d0++) {
        acc[d0] = 0.0f;
    }
    float M = NEG_INF_SCORE;   // running maximum; exp(M - s) is 0 for the first entry
    float S = 0.0f;            // softmax denominator scaled by exp(-M)

    const uint j1 = min((b + 1) * blk_size, n_kv);
    for (uint j = b * blk_size; j < j1; j++) {
        float mv = 0.0f;
#if defined(HAS_MASK)
        uint mbits;
        LOAD_U16_UNALIGNED(mask, (m_base + j) * 2, mbits);
        if (mbits == 0xFC00u) {   // -inf: entry not visible
            continue;
        }
        mv = slope * f16tof32(mbits);
#endif
#if defined(K_Q8_0)
        const uint kj = (k_base + j * stride_k1) * 32u;
#else
        const uint kj = k_base + j * stride_k1;
#endif
        float s = 0.0f;
        for (uint a = 0; a < DK / 4; a++) {
            s += dot(q[a], load_k4(kj + 4 * a));
        }
        s *= scale;
#if defined(SOFTCAP)
        s = logit_softcap * tanh(s);
#endif
        s += mv;

#if defined(V_Q8_0)
        const uint vj = (v_base + j * stride_v1) * 32u;
#else
        const uint vj = v_base + j * stride_v1;
#endif
        if (s > M) {
            // new maximum: rescale what was accumulated, this entry gets weight 1
            const float ms = exp(M - s);
            M = s;
            for (uint d1 = 0; d1 < DV / 4; d1++) {
                acc[d1] = acc[d1] * ms + load_v4(vj + 4 * d1);
            }
            S = S * ms + 1.0f;
        } else {
            const float vs = exp(s - M);
            for (uint d2 = 0; d2 < DV / 4; d2++) {
                acc[d2] += vs * load_v4(vj + 4 * d2);
            }
            S += vs;
        }
    }

    const uint base = (t * n_blocks + b) * (DV + 1);
    if (S == 0.0f) {
        STORE_F32(tmp, base, NEG_INF_SCORE);
        return;
    }
    const float inv = 1.0f / S;
    STORE_F32(tmp, base, M + log(S));
    for (uint d4 = 0; d4 < DV; d4++) {
        STORE_F32(tmp, base + 1 + d4, acc[d4 / 4][d4 % 4] * inv);
    }
#endif
}
