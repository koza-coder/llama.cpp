#include "common.hlsli"

// CONV_TRANSPOSE_2D (f32 src, f32 or f16 kernel, f32 dst):
//   dst[n, oc, oh, ow] = sum over (sy, ky) with sy*stride + ky == oh,
//                        over (sx, kx) with sx*stride + kx == ow,
//                        over cin of src[n, cin, sy, sx] * knl[cin, oc, ky, kx]
//
// The CPU scatters each input pixel over the kernel footprint, which needs a zeroed dst and two
// permuted staging buffers. Read backwards it is a gather over the at most ceil(Kh/stride) by
// ceil(Kw/stride) taps that can land on one output pixel, so one thread owns one output element.
//
// With an f16 kernel the CPU rounds the source to f16 as well and dots in f16, so this does the
// same; accumulation stays f32 either way.
// defines: KNL_F16 (with USE_16BIT)

RWByteAddressBuffer knl : register(u0);   // src0 {Kw, Kh, Cout, Cin}
RWByteAddressBuffer src : register(u1);   // src1 {Sw, Sh, Cin, N}
RWByteAddressBuffer dst : register(u2);   // dst  {Ow, Oh, Cout, N}

cbuffer Params : register(b0) {
    uint offset_knl;
    uint offset_src;
    uint offset_dst;
    uint stride_k1;   // src0 nb[1] in kernel elements
    uint stride_k2;
    uint stride_k3;
    uint stride_s1;   // src1 nb[1] / 4
    uint stride_s2;
    uint stride_s3;
    uint stride_d1;   // dst nb[1] / 4
    uint stride_d2;
    uint stride_d3;
    uint kw;
    uint kh;
    uint sw;
    uint sh;
    uint cin;
    uint ne0;
    uint ne1;
    uint ne2;
    uint st;          // stride
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
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint n  = i / (ne2 * ne1 * ne0);
    uint rem      = i % (ne2 * ne1 * ne0);
    const uint oc = rem / (ne1 * ne0);
    rem           = rem % (ne1 * ne0);
    const uint oh = rem / ne0;
    const uint ow = rem % ne0;

    float sum = 0.0f;
    for (uint ky = 0; ky < kh; ky++) {
        if (ky > oh) {
            break;
        }
        const uint ry = oh - ky;
        if (ry % st != 0u) {
            continue;
        }
        const uint sy = ry / st;
        if (sy >= sh) {
            continue;
        }
        for (uint kx = 0; kx < kw; kx++) {
            if (kx > ow) {
                break;
            }
            const uint rx = ow - kx;
            if (rx % st != 0u) {
                continue;
            }
            const uint sx = rx / st;
            if (sx >= sw) {
                continue;
            }
            const uint k_base = offset_knl + oc * stride_k2 + ky * stride_k1 + kx;
            const uint s_base = offset_src + n * stride_s3 + sy * stride_s1 + sx;
            for (uint c = 0; c < cin; c++) {
                sum += SRC_ROUND(LOAD_F32(src, s_base + c * stride_s2)) * LOAD_KNL(k_base + c * stride_k3);
            }
        }
    }
    STORE_F32(dst, offset_dst + n * stride_d3 + oc * stride_d2 + oh * stride_d1 + ow, sum);
}
