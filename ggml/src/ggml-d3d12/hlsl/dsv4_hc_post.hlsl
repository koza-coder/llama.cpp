#include "common.hlsli"

// DSV4_HC_POST (f32): fan one attention result back out over the hc hybrid-cache streams and mix
// in the residuals.
//   dst[i0, idst, it] = x[i0, it] * post[idst, it] + sum over isrc of residual[i0, isrc, it] * comb[idst, isrc, it]
// Without a comb matrix the mixing is the identity, so each stream just keeps its own residual.
// One thread per destination element. defines: HAS_COMB

RWByteAddressBuffer x_buf : register(u0);
RWByteAddressBuffer r_buf : register(u1);   // residual
RWByteAddressBuffer p_buf : register(u2);   // post
RWByteAddressBuffer c_buf : register(u3);   // comb, bound to the residual when unused
RWByteAddressBuffer dst   : register(u4);

cbuffer Params : register(b0) {
    uint offset_x;
    uint offset_r;
    uint offset_p;
    uint offset_c;
    uint offset_dst;
    uint stride_x0;
    uint stride_x1;
    uint stride_r0;
    uint stride_r1;
    uint stride_r2;
    uint stride_p0;
    uint stride_p1;
    uint stride_c0;
    uint stride_c1;
    uint stride_c2;
    uint stride_d0;
    uint stride_d1;
    uint stride_d2;
    uint n_embd;
    uint hc;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint i0   = i % n_embd;
    const uint idst = (i / n_embd) % hc;
    const uint it   = i / (n_embd * hc);

    const float xv = LOAD_F32(x_buf, offset_x + i0 * stride_x0 + it * stride_x1);
    const float pv = LOAD_F32(p_buf, offset_p + idst * stride_p0 + it * stride_p1);

    float sum = xv * pv;
#ifdef HAS_COMB
    for (uint isrc = 0; isrc < hc; isrc++) {
        const float rv = LOAD_F32(r_buf, offset_r + i0 * stride_r0 + isrc * stride_r1 + it * stride_r2);
        const float cv = LOAD_F32(c_buf, offset_c + idst * stride_c0 + isrc * stride_c1 + it * stride_c2);
        sum += rv * cv;
    }
#else
    sum += LOAD_F32(r_buf, offset_r + i0 * stride_r0 + idst * stride_r1 + it * stride_r2);
#endif
    STORE_F32(dst, offset_dst + i0 * stride_d0 + idst * stride_d1 + it * stride_d2, sum);
}
