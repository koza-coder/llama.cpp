#include "common.hlsli"

// dst[i1, i2, i3] = src[idx[i1, i2, i3], i2, i3]; idx is i32, one thread per element.
// defines: SRC_{F32,F16,I32}; quantized sources use get_rows_q.hlsl

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer idx : register(u1);
RWByteAddressBuffer dst : register(u2);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_idx;
    uint offset_dst;

    uint stride_src1;
    uint stride_src2;
    uint stride_src3;

    uint stride_idx0;
    uint stride_idx1;
    uint stride_idx2;

    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;

    uint ne0;    // dst row length
    uint ne1;    // idx ne0
    uint ne2;    // idx ne1
    uint n_units;

    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint t = flat_index(gid, nwg_x);
    if (t >= n_units) {
        return;
    }
    const uint units_per_row = ne0;
    uint r = t / units_per_row;
    const uint u = t % units_per_row;
    const uint i3 = r / (ne2 * ne1);
    r = r % (ne2 * ne1);
    const uint i2 = r / ne1;
    const uint i1 = r % ne1;

    const uint row     = (uint) LOAD_I32(idx, offset_idx + i1 * stride_idx0 + i2 * stride_idx1 + i3 * stride_idx2);
    const uint src_row = offset_src + row * stride_src1 + i2 * stride_src2 + i3 * stride_src3;
    const uint dst_row = offset_dst + i1 * stride_dst1 + i2 * stride_dst2 + i3 * stride_dst3;

#if defined(SRC_F32)
    STORE_F32(dst, dst_row + u, LOAD_F32(src, src_row + u));
#elif defined(SRC_I32)
    STORE_I32(dst, dst_row + u, LOAD_I32(src, src_row + u));
#elif defined(SRC_F16)
    uint bits;
    LOAD_U16_UNALIGNED(src, (src_row + u) * 2, bits);
    STORE_F32(dst, dst_row + u, f16tof32(bits));
#endif
}
