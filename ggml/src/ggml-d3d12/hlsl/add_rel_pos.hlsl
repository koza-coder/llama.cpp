#include "common.hlsli"

// ADD_REL_POS (f32): dst = src0 with the two relative position biases added.
//
// The CPU writes this as two scatter-adds over a block of ne10 x ne10 destination elements per
// (i13, i12, i11). Inverting both scatters turns it into a gather with no overlap at all: inside a
// block, element q takes src2[jp1 + q / ne10] from the row scatter and src1[jp1 + q % ne10] from
// the column scatter, because the column scatter's index jdw + j * ne10 simplifies to
// jp1 * ne10 + i10 + j * ne10.

RWByteAddressBuffer src0 : register(u0);
RWByteAddressBuffer src1 : register(u1);
RWByteAddressBuffer src2 : register(u2);
RWByteAddressBuffer dst  : register(u3);

cbuffer Params : register(b0) {
    uint offset_src0;
    uint offset_src1;
    uint offset_src2;
    uint offset_dst;
    uint ne10;
    uint ne;
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint i = flat_index(gid, nwg_x);
    if (i >= ne) {
        return;
    }
    const uint block = i / (ne10 * ne10);
    const uint q     = i % (ne10 * ne10);
    const uint jp1   = block * ne10;

    const float base = LOAD_F32(src0, offset_src0 + i);
    const float row  = LOAD_F32(src2, offset_src2 + jp1 + q / ne10);
    const float col  = LOAD_F32(src1, offset_src1 + jp1 + q % ne10);
    STORE_F32(dst, offset_dst + i, base + row + col);
}
