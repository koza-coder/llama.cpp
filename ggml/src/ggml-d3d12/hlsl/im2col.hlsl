#include "common.hlsli"

// IM2COL (f32 input): [N, IC, IH, IW] => dst [N, OH, OW, IC*KH*KW], one thread per dst element, following the
// CPU reference; out-of-image taps are 0. dst is contiguous. defines: DST_F16 (with USE_16BIT)

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_n;    // src elements per batch entry
    uint stride_c;    // src elements per channel
    uint stride_h;    // src elements per image row
    uint stride_w;    // src elements per image column
    int  s0;
    int  s1;
    int  p0;
    int  p1;
    int  d0;
    int  d1;
    uint IC;
    uint IH;
    uint IW;
    uint KH;
    uint KW;
    uint OH;
    uint OW;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint e = flat_index(gid, nwg_x);
    if (e >= ne) {
        return;
    }
    const uint K      = IC * KH * KW;
    const uint pos    = e / K;
    const uint within = e % K;
    const uint iic = within / (KH * KW);
    const uint ikh = (within / KW) % KH;
    const uint ikw = within % KW;
    const uint iow = pos % OW;
    const uint ioh = (pos / OW) % OH;
    const uint in_ = pos / (OW * OH);

    const int iiw = (int) iow * s0 + (int) ikw * d0 - p0;
    const int iih = (int) ioh * s1 + (int) ikh * d1 - p1;
    float v = 0.0f;
    if (iih >= 0 && iih < (int) IH && iiw >= 0 && iiw < (int) IW) {
        v = LOAD_F32(src, offset_src + in_ * stride_n + iic * stride_c + (uint) iih * stride_h + (uint) iiw * stride_w);
    }
#if defined(DST_F16)
    STORE_F16(dst, offset_dst + e, v);
#else
    STORE_F32(dst, offset_dst + e, v);
#endif
}
