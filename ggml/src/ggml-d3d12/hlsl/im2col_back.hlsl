#include "common.hlsli"

// IM2COL_BACK (f32): the adjoint of im2col, one thread per element of dst [IW, IH, IC, N].
// src0 holds the gradients of the im2col output, laid out contiguously as
// [IC*KH*KW, OW, OH, N]; src1 is only there for the kernel shape, so it is not bound.
// Following ggml_compute_forward_im2col_back_f32, every (ikh, ikw) whose forward mapping
// landed on this pixel contributes; a stride that skipped the pixel leaves a non-zero
// remainder and is dropped. The arithmetic is signed: tmpw can go negative.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint u_N;
    uint u_IC;
    uint u_IH;
    uint u_IW;
    uint u_KH;
    uint u_KW;
    uint u_OH;
    uint u_OW;
    int  s0;
    int  s1;
    int  p0;
    int  p1;
    int  d0;
    int  d1;
    uint offset_src;
    uint offset_dst;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint iiw = i % u_IW;
    const uint iih = (i / u_IW) % u_IH;
    const uint iic = (i / (u_IW * u_IH)) % u_IC;
    const uint in_ = i / (u_IW * u_IH * u_IC);

    const uint row_stride = u_IC * u_KH * u_KW;
    const uint kernel_off = iic * (u_KH * u_KW);

    float grad = 0.0f;
    for (uint ikh = 0; ikh < u_KH; ikh++) {
        const int tmph = (int) iih + p1 - (int) ikh * d1;
        if (tmph % s1 != 0) {
            continue;
        }
        const int ioh = tmph / s1;
        if (ioh < 0 || ioh >= (int) u_OH) {
            continue;
        }
        for (uint ikw = 0; ikw < u_KW; ikw++) {
            const int tmpw = (int) iiw + p0 - (int) ikw * d0;
            if (tmpw % s0 != 0) {
                continue;
            }
            const int iow = tmpw / s0;
            if (iow < 0 || iow >= (int) u_OW) {
                continue;
            }
            const uint base = (in_ * u_OH * u_OW + (uint) ioh * u_OW + (uint) iow) * row_stride;
            grad += LOAD_F32(src, offset_src + base + kernel_off + ikh * u_KW + ikw);
        }
    }
    STORE_F32(dst, offset_dst + i, grad);
}
