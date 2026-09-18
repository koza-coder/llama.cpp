#include "common.hlsli"

// SOFT_MAX_BACK (f32), one workgroup per row. src0 is the incoming gradient dy, src1 is the
// forward output y. dx[k] = y[k] * (dy[k] - dot(y, dy)) * scale.
// All three tensors are contiguous, so a row is just row_index * ne0.

RWByteAddressBuffer dy_buf : register(u0);
RWByteAddressBuffer y_buf  : register(u1);
RWByteAddressBuffer dst    : register(u2);

cbuffer Params : register(b0) {
    uint offset_dy;
    uint offset_y;
    uint offset_dst;
    uint ne0;
    uint n_rows;
    float scale;
    uint nwg_x;
};

groupshared float scratch[WG_SIZE];

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint row = gid.y * nwg_x + gid.x;
    if (row >= n_rows) {
        return;
    }
    const uint base = row * ne0;

    float dot = 0.0f;
    for (uint col = gtid.x; col < ne0; col += WG_SIZE) {
        dot += LOAD_F32(y_buf, offset_y + base + col) * LOAD_F32(dy_buf, offset_dy + base + col);
    }
    scratch[gtid.x] = dot;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = WG_SIZE / 2; off > 0; off /= 2) {
        if (gtid.x < off) {
            scratch[gtid.x] += scratch[gtid.x + off];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float dot_y_dy = scratch[0];

    for (uint c = gtid.x; c < ne0; c += WG_SIZE) {
        const float dy = LOAD_F32(dy_buf, offset_dy + base + c);
        const float y  = LOAD_F32(y_buf, offset_y + base + c);
        STORE_F32(dst, offset_dst + base + c, (dy - dot_y_dy) * y * scale);
    }
}
