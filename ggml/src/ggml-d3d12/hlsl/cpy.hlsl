#include "common.hlsli"

// defines: SRC_{F32,F16,I32}, DST_{F32,F16,I32}

#if defined(SRC_F32)
#define SRC_T float
#define LOAD_SRC(i) LOAD_F32(src, i)
#elif defined(SRC_F16)
#define SRC_T float
#define LOAD_SRC(i) LOAD_F16(src, i)
#elif defined(SRC_I32)
#define SRC_T int
#define LOAD_SRC(i) LOAD_I32(src, i)
#endif

#if defined(DST_F32)
#define DST_T float
#define STORE_DST(i, v) STORE_F32(dst, i, v)
#elif defined(DST_F16)
#define DST_T float
#define STORE_DST(i, v) STORE_F16(dst, i, v)
#elif defined(DST_I32)
#define DST_T int
#define STORE_DST(i, v) STORE_I32(dst, i, v)
#endif

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint ne;
    uint offset_src;
    uint offset_dst;

    uint stride_src0;
    uint stride_src1;
    uint stride_src2;
    uint stride_src3;

    uint stride_dst0;
    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint src_ne0;
    uint src_ne1;
    uint src_ne2;

    uint dst_ne0;
    uint dst_ne1;
    uint dst_ne2;

    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint idx = flat_index(gid, nwg_x);
    if (idx >= ne) {
        return;
    }

    uint i = idx;
    const uint i3 = i / (src_ne2 * src_ne1 * src_ne0);
    i = i % (src_ne2 * src_ne1 * src_ne0);
    const uint i2 = i / (src_ne1 * src_ne0);
    i = i % (src_ne1 * src_ne0);
    const uint i1 = i / src_ne0;
    const uint i0 = i % src_ne0;

    uint j = idx;
    const uint j3 = j / (dst_ne2 * dst_ne1 * dst_ne0);
    j = j % (dst_ne2 * dst_ne1 * dst_ne0);
    const uint j2 = j / (dst_ne1 * dst_ne0);
    j = j % (dst_ne1 * dst_ne0);
    const uint j1 = j / dst_ne0;
    const uint j0 = j % dst_ne0;

    const uint src_idx = i0 * stride_src0 + i1 * stride_src1 + i2 * stride_src2 + i3 * stride_src3;
    const uint dst_idx = j0 * stride_dst0 + j1 * stride_dst1 + j2 * stride_dst2 + j3 * stride_dst3;

    const SRC_T v = LOAD_SRC(offset_src + src_idx);
    STORE_DST(offset_dst + dst_idx, (DST_T) v);
}
