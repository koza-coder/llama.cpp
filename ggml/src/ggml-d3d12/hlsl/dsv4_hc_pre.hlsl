#include "common.hlsli"

// DSV4_HC_PRE (f32): collapse the hc hybrid-cache streams of x into one, weighted per stream.
//   dst[i0, it] = scale * sum over ih of x[i0, ih, it] * w
// where w is either a per (stream, token) weight or, when gated, sigmoid of a per element gate.
// One thread per destination element. defines: GATED

RWByteAddressBuffer x_buf : register(u0);
RWByteAddressBuffer w_buf : register(u1);
RWByteAddressBuffer dst   : register(u2);

cbuffer Params : register(b0) {
    uint offset_x;
    uint offset_w;
    uint offset_dst;
    uint stride_x0;
    uint stride_x1;
    uint stride_x2;
    uint stride_w0;
    uint stride_w1;
    uint stride_w2;
    uint stride_d0;
    uint stride_d1;
    uint n_embd;
    uint hc;
    uint scale_bits;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint i0 = i % n_embd;
    const uint it = i / n_embd;

    float sum = 0.0f;
    for (uint ih = 0; ih < hc; ih++) {
        const float xv = LOAD_F32(x_buf, offset_x + i0 * stride_x0 + ih * stride_x1 + it * stride_x2);
#ifdef GATED
        const float gv = LOAD_F32(w_buf, offset_w + i0 * stride_w0 + ih * stride_w1 + it * stride_w2);
        const float wv = 1.0f / (1.0f + exp(-gv));
#else
        const float wv = LOAD_F32(w_buf, offset_w + ih * stride_w0 + it * stride_w1);
#endif
        sum += xv * wv;
    }
    STORE_F32(dst, offset_dst + i0 * stride_d0 + it * stride_d1, asfloat(scale_bits) * sum);
}
