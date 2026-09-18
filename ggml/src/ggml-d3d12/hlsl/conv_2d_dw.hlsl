#include "common.hlsli"

// CONV_2D_DW (f32 src, f32 or f16 kernel, f32 dst): depthwise 2D convolution, the WHCN layout only.
// Each output channel is convolved with its own knl_w x knl_h filter and never mixes with another,
// so one thread owns one destination element and gathers the window itself. Taps that fall outside
// the source are skipped rather than treated as zero, which is the same thing here but matches
// ggml_compute_forward_conv_2d_dw_whcn exactly.

RWByteAddressBuffer knl : register(u0);
RWByteAddressBuffer src : register(u1);
RWByteAddressBuffer dst : register(u2);

cbuffer Params : register(b0) {
    uint offset_knl;
    uint offset_src;
    uint offset_dst;
    uint src_w;
    uint src_h;
    uint dst_w;
    uint dst_h;
    uint knl_w;
    uint knl_h;
    uint channels;
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
#else
#define LOAD_KNL(i) LOAD_F32(knl, (i))
#endif

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint plane = dst_w * dst_h;
    const uint c     = i / plane;            // channel within the batch, flattened as ic + ib * channels
    const uint rem   = i % plane;
    const uint dst_y = rem / dst_w;
    const uint dst_x = rem % dst_w;

    const uint knl_base = offset_knl + (c % channels) * knl_w * knl_h;
    const uint src_base = offset_src + c * src_w * src_h;

    float sum = 0.0f;
    for (uint ky = 0; ky < knl_h; ky++) {
        const int sy = (int) dst_y * stride_y + (int) ky * dilation_y - pad_y;
        if (sy < 0 || sy >= (int) src_h) {
            continue;
        }
        for (uint kx = 0; kx < knl_w; kx++) {
            const int sx = (int) dst_x * stride_x + (int) kx * dilation_x - pad_x;
            if (sx < 0 || sx >= (int) src_w) {
                continue;
            }
            sum += LOAD_KNL(knl_base + ky * knl_w + kx) * LOAD_F32(src, src_base + (uint) sy * src_w + (uint) sx);
        }
    }
    STORE_F32(dst, offset_dst + i, sum);
}
