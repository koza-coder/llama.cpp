#include "common.hlsli"

// CONV_2D (f32 src, f32 or f16 kernel, f32 dst): direct convolution.
//   dst[n, oc, y, x] = sum over (ic, ky, kx) of src[n, ic, sy, sx] * knl[oc, ic, ky, kx]
// The CPU builds im2col patches and hands them to a GEMM; here one thread owns one output element
// and reads the window straight out of src, so no patch buffer is needed. Taps outside the image
// contribute zero, as they do on the CPU.
//
// With an f16 kernel the CPU stores the patch as f16 before the dot, so the source is rounded the
// same way here; accumulation stays f32.
// defines: KNL_F16 (with USE_16BIT)

RWByteAddressBuffer knl : register(u0);   // src0 {KW, KH, IC, OC}, contiguous
RWByteAddressBuffer src : register(u1);   // src1 {W, H, C, N}
RWByteAddressBuffer dst : register(u2);   // dst  {OW, OH, OC, N}

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
    uint src_w;
    uint src_h;
    uint c_in;
    uint ne0;
    uint ne1;
    uint ne2;
    int  stride_x;
    int  stride_y;
    int  pad_x;
    int  pad_y;
    int  dilation_x;
    int  dilation_y;
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
    const uint n  = i / (ne2 * ne1 * ne0);
    uint rem      = i % (ne2 * ne1 * ne0);
    const uint oc = rem / (ne1 * ne0);
    rem           = rem % (ne1 * ne0);
    const uint oy = rem / ne0;
    const uint ox = rem % ne0;

    const uint plane    = knl_h * knl_w;
    const uint knl_base = offset_knl + oc * c_in * plane;
    const uint src_base = offset_src + n * stride_s3;

    float sum = 0.0f;
    for (uint ky = 0; ky < knl_h; ky++) {
        const int sy = (int) oy * stride_y + (int) ky * dilation_y - pad_y;
        if (sy < 0 || sy >= (int) src_h) {
            continue;
        }
        for (uint kx = 0; kx < knl_w; kx++) {
            const int sx = (int) ox * stride_x + (int) kx * dilation_x - pad_x;
            if (sx < 0 || sx >= (int) src_w) {
                continue;
            }
            const uint s_pos = src_base + (uint) sy * stride_s1 + (uint) sx;
            const uint k_pos = knl_base + ky * knl_w + kx;
            for (uint ic = 0; ic < c_in; ic++) {
                sum += SRC_ROUND(LOAD_F32(src, s_pos + ic * stride_s2)) * LOAD_KNL(k_pos + ic * plane);
            }
        }
    }
    STORE_F32(dst, offset_dst + n * stride_d3 + oc * stride_d2 + oy * stride_d1 + ox, sum);
}
