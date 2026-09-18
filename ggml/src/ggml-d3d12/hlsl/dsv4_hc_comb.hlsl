#include "common.hlsli"

// DSV4_HC_COMB (f32): build the hc x hc stream mixing matrix for each token. Per token: a softmax
// down each source column of (mix * scale + base), then eps added, then Sinkhorn normalisation -
// columns once, and rows plus columns n_iter-1 more times.
//
// hc is fixed at 4 in the CPU reference, so the whole matrix is 16 floats and lives in registers;
// one thread owns one token and the iteration stays sequential inside it.

#define HC 4

RWByteAddressBuffer m_buf : register(u0);   // mixes {(2 + hc)*hc, n_tokens}
RWByteAddressBuffer s_buf : register(u1);   // scale, element 2 is the comb scale
RWByteAddressBuffer b_buf : register(u2);   // base {(2 + hc)*hc}
RWByteAddressBuffer dst   : register(u3);   // {hc, hc, n_tokens}

cbuffer Params : register(b0) {
    uint offset_m;
    uint offset_s;
    uint offset_b;
    uint offset_dst;
    uint stride_m0;
    uint stride_m1;
    uint stride_s0;
    uint stride_b0;
    uint stride_d0;
    uint stride_d1;
    uint stride_d2;
    uint eps_bits;
    uint n_iter;
    uint n_tokens;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint it = flat_index(gid, nwg_x);
    if (it >= n_tokens) {
        return;
    }
    const float eps        = asfloat(eps_bits);
    const float scale_comb = LOAD_F32(s_buf, offset_s + 2u * stride_s0);
    const uint  comb_off   = 2u * HC;   // the comb block starts after the two leading hc vectors

    float comb[HC * HC];

    for (uint isrc = 0; isrc < HC; isrc++) {
        float mx = -3.402823466e+38f;
        for (uint idst = 0; idst < HC; idst++) {
            const uint  idx = idst + HC * isrc;
            const float xv = LOAD_F32(m_buf, offset_m + (comb_off + idx) * stride_m0 + it * stride_m1);
            const float bv = LOAD_F32(b_buf, offset_b + (comb_off + idx) * stride_b0);
            const float v  = xv * scale_comb + bv;
            comb[idx] = v;
            mx = max(mx, v);
        }
        float sum = 0.0f;
        for (uint idst = 0; idst < HC; idst++) {
            const uint  idx = idst + HC * isrc;
            const float v = exp(comb[idx] - mx);
            comb[idx] = v;
            sum += v;
        }
        const float inv_sum = 1.0f / sum;
        for (uint idst = 0; idst < HC; idst++) {
            const uint idx = idst + HC * isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    // Sinkhorn: columns once, then (rows, columns) for each further iteration
    for (uint i = 0; i < n_iter; i++) {
        if (i > 0u) {
            for (uint isrc = 0; isrc < HC; isrc++) {
                float sum = eps;
                for (uint idst = 0; idst < HC; idst++) {
                    sum += comb[idst + HC * isrc];
                }
                const float inv = 1.0f / sum;
                for (uint idst = 0; idst < HC; idst++) {
                    comb[idst + HC * isrc] *= inv;
                }
            }
        }
        for (uint idst = 0; idst < HC; idst++) {
            float sum = eps;
            for (uint isrc = 0; isrc < HC; isrc++) {
                sum += comb[idst + HC * isrc];
            }
            const float inv = 1.0f / sum;
            for (uint isrc = 0; isrc < HC; isrc++) {
                comb[idst + HC * isrc] *= inv;
            }
        }
    }

    for (uint isrc = 0; isrc < HC; isrc++) {
        for (uint idst = 0; idst < HC; idst++) {
            STORE_F32(dst, offset_dst + idst * stride_d0 + isrc * stride_d1 + it * stride_d2,
                      comb[idst + HC * isrc]);
        }
    }
}
