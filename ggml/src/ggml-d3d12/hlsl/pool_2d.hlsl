#include "common.hlsli"

// POOL_2D: one thread per dst element, following the CPU reference. Taps outside the plane are skipped, but the
// average always divides by the full kernel area. dst is f32 and contiguous. defines: SRC_F16, POOL_MAX

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src1;   // src elements per row
    uint stride_src2;   // src elements per plane
    uint src_ne0;
    uint src_ne1;
    uint px;
    uint py;
    int  k0;
    int  k1;
    int  s0;
    int  s1;
    int  p0;
    int  p1;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint plane = i / (px * py);
    const uint oy    = (i / px) % py;
    const uint ox    = i % px;

    const int ix = (int) ox * s0 - p0;
    const int iy = (int) oy * s1 - p1;
#if defined(POOL_MAX)
    float res = -3.4028235e38f;
#else
    float res = 0.0f;
#endif
    for (int ky = 0; ky < k1; ky++) {
        const int y = iy + ky;
        if (y < 0 || y >= (int) src_ne1) {
            continue;
        }
        const uint row = offset_src + plane * stride_src2 + (uint) y * stride_src1;
        for (int kx = 0; kx < k0; kx++) {
            const int x = ix + kx;
            if (x < 0 || x >= (int) src_ne0) {
                continue;
            }
#if defined(SRC_F16)
            uint bits;
            LOAD_U16_UNALIGNED(src, (row + (uint) x) * 2, bits);
            const float v = f16tof32(bits);
#else
            const float v = LOAD_F32(src, row + (uint) x);
#endif
#if defined(POOL_MAX)
            res = max(v, res);
#else
            res += v;
#endif
        }
    }
#if !defined(POOL_MAX)
    res /= (float) (k0 * k1);
#endif
    STORE_F32(dst, offset_dst + i, res);
}
