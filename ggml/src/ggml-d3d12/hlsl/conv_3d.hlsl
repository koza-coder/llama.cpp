#include "common.hlsli"

// CONV_3D (f32 src, f32 or f16 kernel, f32 dst): direct 3D convolution.
//   dst[b*OC + oc, z, y, x] = sum over (ic, kz, ky, kx) of
//       src[b*C + ic, sz, sy, sx] * knl[oc*C + ic, kz, ky, kx]
// Same shape as conv_2d.hlsl with a depth axis added: the CPU goes through im2col plus a GEMM,
// this gathers the window per output element. Batch and channel share the last tensor axis on
// both sides, which is why the source and kernel planes are indexed with a product.
//
// With an f16 kernel the CPU stores the patch as f16 before the dot, so the source is rounded the
// same way here; accumulation stays f32.
// defines: KNL_F16 (with USE_16BIT)

RWByteAddressBuffer knl : register(u0);   // src0 {KW, KH, KD, OC*C}, contiguous
RWByteAddressBuffer src : register(u1);   // src1 {W, H, D, N*C}
RWByteAddressBuffer dst : register(u2);   // dst  {OW, OH, OD, N*OC}

cbuffer Params : register(b0) {
    uint offset_knl;
    uint offset_src;
    uint offset_dst;
    uint stride_s1;   // src1 nb[1] / 4
    uint stride_s2;
    uint stride_s3;
    uint stride_d1;   // dst nb[1] / 4
    uint stride_d2;
    uint stride_d3;
    uint knl_w;
    uint knl_h;
    uint knl_d;
    uint src_w;
    uint src_h;
    uint src_d;
    uint c_in;
    uint c_out;
    uint ne0;
    uint ne1;
    uint ne2;
    int  s0;
    int  s1;
    int  s2;
    int  p0;
    int  p1;
    int  p2;
    int  d0;
    int  d1;
    int  d2;
    uint ne;
    uint nwg_x;
};

#ifdef KNL_F16
#define LOAD_KNL(i) LOAD_F16(knl, (i))
#define SRC_ROUND(v) f16tof32((uint) f32_to_f16_rne(v))
#else
#define LOAD_KNL(i) LOAD_F32(knl, (i))
#define SRC_ROUND(v) (v)
#endif

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint ocn = i / (ne2 * ne1 * ne0);   // batch * OC + oc
    uint rem       = i % (ne2 * ne1 * ne0);
    const uint dz  = rem / (ne1 * ne0);
    rem            = rem % (ne1 * ne0);
    const uint dy  = rem / ne0;
    const uint dx  = rem % ne0;

    const uint b  = ocn / c_out;
    const uint oc = ocn % c_out;

    const uint per_channel = knl_d * knl_h * knl_w;
    const uint knl_base    = offset_knl + oc * c_in * per_channel;
    const uint src_base    = offset_src + b * c_in * stride_s3;

    float sum = 0.0f;
    for (uint kz = 0; kz < knl_d; kz++) {
        const int sz = (int) dz * s2 + (int) kz * d2 - p2;
        if (sz < 0 || sz >= (int) src_d) {
            continue;
        }
        for (uint ky = 0; ky < knl_h; ky++) {
            const int sy = (int) dy * s1 + (int) ky * d1 - p1;
            if (sy < 0 || sy >= (int) src_h) {
                continue;
            }
            for (uint kx = 0; kx < knl_w; kx++) {
                const int sx = (int) dx * s0 + (int) kx * d0 - p0;
                if (sx < 0 || sx >= (int) src_w) {
                    continue;
                }
                const uint s_pos = src_base + (uint) sz * stride_s2 + (uint) sy * stride_s1 + (uint) sx;
                const uint k_pos = knl_base + kz * knl_h * knl_w + ky * knl_w + kx;
                for (uint ic = 0; ic < c_in; ic++) {
                    sum += SRC_ROUND(LOAD_F32(src, s_pos + ic * stride_s3)) * LOAD_KNL(k_pos + ic * per_channel);
                }
            }
        }
    }
    STORE_F32(dst, offset_dst + ocn * stride_d3 + dz * stride_d2 + dy * stride_d1 + dx, sum);
}
