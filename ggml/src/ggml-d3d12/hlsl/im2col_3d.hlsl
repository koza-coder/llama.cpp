#include "common.hlsli"

// IM2COL_3D (f32 input): [N*IC, ID, IH, IW] => dst [N*OD, OH, OW, IC*KD*KH*KW], one thread per dst
// element, following the CPU reference; taps outside the volume are 0. dst is contiguous.
// defines: DST_F16 (with USE_16BIT)

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_n;    // src1 nb[3] / 4, one (batch, in-channel) volume
    uint stride_d;    // src1 nb[2] / 4
    uint stride_h;    // src1 nb[1] / 4
    int  s0;
    int  s1;
    int  s2;
    int  p0;
    int  p1;
    int  p2;
    int  d0;
    int  d1;
    int  d2;
    uint IC;
    uint ID;
    uint IH;
    uint IW;
    uint KD;
    uint KH;
    uint KW;
    uint OD;
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
    const uint K      = IC * KD * KH * KW;
    const uint pos    = e / K;
    const uint within = e % K;

    const uint ikw = within % KW;
    const uint ikh = (within / KW) % KH;
    const uint ikd = (within / (KW * KH)) % KD;
    const uint iic = within / (KW * KH * KD);

    const uint iow = pos % OW;
    const uint ioh = (pos / OW) % OH;
    const uint iod = (pos / (OW * OH)) % OD;
    const uint in_ = pos / (OW * OH * OD);

    const int iiw = (int) iow * s0 + (int) ikw * d0 - p0;
    const int iih = (int) ioh * s1 + (int) ikh * d1 - p1;
    const int iid = (int) iod * s2 + (int) ikd * d2 - p2;

    float v = 0.0f;
    if (iid >= 0 && iid < (int) ID && iih >= 0 && iih < (int) IH && iiw >= 0 && iiw < (int) IW) {
        const uint base = offset_src + (in_ * IC + iic) * stride_n;
        v = LOAD_F32(src, base + (uint) iid * stride_d + (uint) iih * stride_h + (uint) iiw);
    }
#if defined(DST_F16)
    STORE_F16(dst, offset_dst + e, v);
#else
    STORE_F32(dst, offset_dst + e, v);
#endif
}
