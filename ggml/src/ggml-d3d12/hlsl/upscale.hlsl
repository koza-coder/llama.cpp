#include "common.hlsli"

// UPSCALE (f32), nearest or bilinear (with or without align corners), following the CPU reference; one thread
// per dst element, any strides. defines: BILINEAR

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint  offset_src;
    uint  offset_dst;
    uint  stride_src0;
    uint  stride_src1;
    uint  stride_src2;
    uint  stride_src3;
    uint  stride_dst0;
    uint  stride_dst1;
    uint  stride_dst2;
    uint  stride_dst3;
    uint  src_ne0;
    uint  src_ne1;
    uint  ne0;
    uint  ne1;
    uint  ne2;
    uint  ne;
    float sf0;
    float sf1;
    float sf2;
    float sf3;
    float pixel_offset;
    uint  nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint i3 = i / (ne2 * ne1 * ne0);
    i = i % (ne2 * ne1 * ne0);
    const uint i2 = i / (ne1 * ne0);
    i = i % (ne1 * ne0);
    const uint i1 = i / ne0;
    const uint i0 = i % ne0;

    const uint i02  = (uint) ((float) i2 / sf2);
    const uint i03  = (uint) ((float) i3 / sf3);
    const uint base = offset_src + i02 * stride_src2 + i03 * stride_src3;
    const uint o    = offset_dst + i0 * stride_dst0 + i1 * stride_dst1 + i2 * stride_dst2 + i3 * stride_dst3;

#if defined(BILINEAR)
    const float y  = ((float) i1 + pixel_offset) / sf1 - pixel_offset;
    const int   fy = (int) floor(y);
    const int   y0 = clamp(fy, 0, (int) src_ne1 - 1);
    const int   y1 = clamp(fy + 1, 0, (int) src_ne1 - 1);
    const float dy = clamp(y - (float) y0, 0.0f, 1.0f);
    const float x  = ((float) i0 + pixel_offset) / sf0 - pixel_offset;
    const int   fx = (int) floor(x);
    const int   x0 = clamp(fx, 0, (int) src_ne0 - 1);
    const int   x1 = clamp(fx + 1, 0, (int) src_ne0 - 1);
    const float dx = clamp(x - (float) x0, 0.0f, 1.0f);

    const float a = LOAD_F32(src, base + (uint) x0 * stride_src0 + (uint) y0 * stride_src1);
    const float b = LOAD_F32(src, base + (uint) x1 * stride_src0 + (uint) y0 * stride_src1);
    const float c = LOAD_F32(src, base + (uint) x0 * stride_src0 + (uint) y1 * stride_src1);
    const float d = LOAD_F32(src, base + (uint) x1 * stride_src0 + (uint) y1 * stride_src1);
    STORE_F32(dst, o, a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy);
#else
    const uint i00 = (uint) ((float) i0 / sf0);
    const uint i01 = (uint) ((float) i1 / sf1);
    STORE_F32(dst, o, LOAD_F32(src, base + i00 * stride_src0 + i01 * stride_src1));
#endif
}
