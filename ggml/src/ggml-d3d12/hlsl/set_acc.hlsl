#include "common.hlsli"

// SET / ACC (f32), second phase only: dst already holds a copy of src0, and this writes src1 into
// the view of dst described by (view_nb1, view_nb2, view_nb3, view_offset), all in elements.
// With ACC defined the value is added instead of assigned. One workgroup per src1 row.

RWByteAddressBuffer src1 : register(u0);
RWByteAddressBuffer dst  : register(u1);

cbuffer Params : register(b0) {
    uint offset_src1;
    uint offset_dst;
    uint stride_11;
    uint stride_12;
    uint stride_13;
    uint view_nb1;
    uint view_nb2;
    uint view_nb3;
    uint view_offset;
    uint ne10;
    uint ne11;
    uint ne12;
    uint n_rows;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint r = gid.y * nwg_x + gid.x;
    if (r >= n_rows) {
        return;
    }
    const uint i3 = r / (ne12 * ne11);
    r = r % (ne12 * ne11);
    const uint i2 = r / ne11;
    const uint i1 = r % ne11;

    const uint src_row = offset_src1 + i3 * stride_13 + i2 * stride_12 + i1 * stride_11;
    const uint dst_row = offset_dst + view_offset + i3 * view_nb3 + i2 * view_nb2 + i1 * view_nb1;

    for (uint c = gtid.x; c < ne10; c += WG_SIZE) {
        const float v = LOAD_F32(src1, src_row + c);
#ifdef ACC
        STORE_F32(dst, dst_row + c, LOAD_F32(dst, dst_row + c) + v);
#else
        STORE_F32(dst, dst_row + c, v);
#endif
    }
}
