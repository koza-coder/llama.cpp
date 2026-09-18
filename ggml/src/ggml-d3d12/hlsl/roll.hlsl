#include "common.hlsli"

// ROLL (f32): dst[i0, i1, i2, i3] = src[(i0 + s) % ne00, wrap(i1 - s1), wrap(i2 - s2), wrap(i3 - s3)]
// with s = wrap(-s0, ne00). wrap matches ggml_wrap_index: a single period adjustment, not a modulo,
// so an out-of-range shift behaves exactly as it does on the CPU.

RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

cbuffer Params : register(b0) {
    uint offset_src;
    uint offset_dst;
    uint stride_src1;
    uint stride_src2;
    uint stride_src3;
    uint stride_dst1;
    uint stride_dst2;
    uint stride_dst3;
    uint ne0;
    uint ne1;
    uint ne2;
    uint ne3;
    int  shift0;
    int  shift1;
    int  shift2;
    int  shift3;
    uint n_rows;
    uint nwg_x;
};

int wrap_index(int i, int ne) {
    if (i < 0) {
        return i + ne;
    }
    if (i >= ne) {
        return i - ne;
    }
    return i;
}

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint r = gid.y * nwg_x + gid.x;
    if (r >= n_rows) {
        return;
    }
    const uint i1 = r % ne1;
    const uint i2 = (r / ne1) % ne2;
    const uint i3 = r / (ne2 * ne1);

    const uint i01 = (uint) wrap_index((int) i1 - shift1, (int) ne1);
    const uint i02 = (uint) wrap_index((int) i2 - shift2, (int) ne2);
    const uint i03 = (uint) wrap_index((int) i3 - shift3, (int) ne3);

    const uint src_row = offset_src + i03 * stride_src3 + i02 * stride_src2 + i01 * stride_src1;
    const uint dst_row = offset_dst + i3 * stride_dst3 + i2 * stride_dst2 + i1 * stride_dst1;

    const uint s = (uint) wrap_index(-shift0, (int) ne0);
    for (uint i0 = gtid.x; i0 < ne0; i0 += WG_SIZE) {
        const uint from = i0 + s >= ne0 ? i0 + s - ne0 : i0 + s;
        STORE_F32(dst, dst_row + i0, LOAD_F32(src, src_row + from));
    }
}
