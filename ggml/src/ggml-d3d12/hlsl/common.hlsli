// Shared helpers for the ggml D3D12 backend kernels.
//
// All tensors are bound as raw (byte address) buffers through root UAV descriptors.
// Element offsets/strides in the constant buffer are in elements of the buffer's type.
// Every kernel is written to compile at cs_6_0; USE_16BIT adds 16-bit loads/stores
// (SM 6.2 with -enable-16bit-types, only when the device reports Native16BitShaderOps).

#ifndef WG_SIZE
#define WG_SIZE 256
#endif


// flat thread index over a 2D dispatch grid: nwg_x groups along x
uint flat_index(uint3 gid, uint nwg_x) {
    return gid.x + nwg_x * WG_SIZE * gid.y;
}

// typed load/store on raw buffers; indices are element indices
#define LOAD_F32(buf, i)     asfloat((buf).Load((i) * 4))
#define STORE_F32(buf, i, v) (buf).Store((i) * 4, asuint(v))
#define LOAD_I32(buf, i)     asint((buf).Load((i) * 4))
#define STORE_I32(buf, i, v) (buf).Store((i) * 4, asuint(v))
// f16 loads go through f16tof32 so the compiler cannot narrow float math back to half ops
// (half division is approximate on some drivers); stores convert in software with
// round-to-nearest-even, because neither f32tof16 nor the (float16_t) cast rounds the same
// way on every driver (the cast rounds toward zero on the Radeon AI PRO R9700 driver)
uint f32_to_f16_rne(float f) {
    const uint x    = asuint(f);
    const uint sign = (x >> 16) & 0x8000u;
    const uint ax   = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) {                       // inf, nan
        return sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u);
    }
    if (ax >= 0x477ff000u) {                       // >= 65520 rounds to inf
        return sign | 0x7c00u;
    }
    if (ax < 0x38800000u) {                        // below 2^-14: f16 subnormal or zero
        const uint sh = 126u - (ax >> 23);         // shift of the 24-bit mantissa, >= 14
        if (sh > 24u) {
            return sign;
        }
        const uint m    = (ax & 0x7fffffu) | 0x800000u;
        uint       mant = m >> sh;
        const uint rem  = m & ((1u << sh) - 1u);
        const uint half = 1u << (sh - 1u);
        if (rem > half || (rem == half && (mant & 1u))) {
            mant++;
        }
        return sign | mant;
    }
    const uint mant = (ax & 0x7fffffu) >> 13;
    const uint rem  = ax & 0x1fffu;
    uint h = (((ax >> 23) - 112u) << 10) | mant;   // a mantissa carry rolls into the exponent
    if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u))) {
        h++;
    }
    return sign | h;
}
#ifdef USE_16BIT
#define LOAD_F16(buf, i)     f16tof32((uint) (buf).Load<uint16_t>((i) * 2))
#define STORE_F16(buf, i, v) (buf).Store<uint16_t>((i) * 2, (uint16_t) f32_to_f16_rne(v))
#endif

// unaligned raw loads (ByteAddressBuffer.Load ignores the low 2 address bits)
#define LOAD_U32_UNALIGNED(buf, addr, out) { \
    const uint _a0 = (addr) & ~3u; \
    const uint _s  = ((addr) & 3u) * 8u; \
    const uint _lo = (buf).Load(_a0); \
    out = _s == 0 ? _lo : ((_lo >> _s) | ((buf).Load(_a0 + 4) << (32u - _s))); \
}
#define LOAD_U16_UNALIGNED(buf, addr, out) { \
    const uint _w = (buf).Load((addr) & ~3u); \
    out = ((_w >> (((addr) & 2u) * 8u)) & 0xFFFFu); \
}
uint byte_of(uint v, uint k) {
    return (v >> (k * 8u)) & 0xFFu;
}
int sbyte_of(uint v, uint k) {
    return ((int) (v << ((3u - k) * 8u))) >> 24;
}
