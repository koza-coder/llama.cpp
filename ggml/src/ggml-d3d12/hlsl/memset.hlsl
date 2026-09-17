#include "common.hlsli"

// Fills [offset, offset + size) bytes of the buffer with the byte pattern in `value`.
// Whole 4-byte words are written directly; the unaligned head/tail words are
// read-modify-written with a byte mask by the single thread that owns them.

#ifndef BYTES_PER_THREAD
#define BYTES_PER_THREAD 16
#endif

RWByteAddressBuffer buf : register(u0);

cbuffer Params : register(b0) {
    uint offset;   // in bytes
    uint size;     // in bytes
    uint value;    // 4 bytes, replicated or distinct
    uint nwg_x;
};

[numthreads(WG_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    const uint start = offset;
    const uint end   = offset + size;
    // each thread owns BYTES_PER_THREAD consecutive bytes, word aligned relative to `start & ~3`
    const uint base = (start & ~3u) + flat_index(gid, nwg_x) * BYTES_PER_THREAD;

    for (uint j = 0; j < BYTES_PER_THREAD; j += 4) {
        const uint word_addr = base + j;
        if (word_addr >= end) {
            return;
        }
        if (word_addr >= start && word_addr + 4 <= end) {
            buf.Store(word_addr, value);
        } else {
            uint mask = 0;
            for (uint k = 0; k < 4; k++) {
                const uint a = word_addr + k;
                if (a >= start && a < end) {
                    mask |= 0xffu << (k * 8);
                }
            }
            const uint existing = buf.Load(word_addr);
            buf.Store(word_addr, (existing & ~mask) | (value & mask));
        }
    }
}
