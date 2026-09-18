#!/usr/bin/env python3
# Regenerate iq_tables.hlsli from ggml/src/ggml-common.h.
#
#   python3 gen_iq_tables.py ../../ggml-common.h > iq_tables.hlsli
#
# The IQ codebooks live in ggml-common.h behind GGML_TABLE_BEGIN/END. HLSL has no 64-bit
# integer literals at cs_6_0, so a uint64 table becomes two uint tables, _LO and _HI, holding
# the low and high 32 bits; byte j of the entry is byte_of(LO, j) for j < 4 and
# byte_of(HI, j - 4) otherwise, which matches the little-endian byte order the CPU code reads.
# Each table is guarded by the SRC0_* defines that use it, so a pipeline only pays for its own.
import re, sys

# (table in ggml-common.h, HLSL name, bits per entry, defines that need it)
TABLES = [
    ("iq3s_grid",    "IQ3S_GRID",    32, ["SRC0_IQ3_S"]),
    ("iq2s_grid",    "IQ2S_GRID",    64, ["SRC0_IQ2_S"]),
    ("iq2xxs_grid",  "IQ2XXS_GRID",  64, ["SRC0_IQ2_XXS"]),
    ("iq2xs_grid",   "IQ2XS_GRID",   64, ["SRC0_IQ2_XS"]),
    ("iq3xxs_grid",  "IQ3XXS_GRID",  32, ["SRC0_IQ3_XXS"]),
    # iq1s_grid holds signed bytes, so read it back with sbyte_of, not byte_of
    ("iq1s_grid",    "IQ1S_GRID",    64, ["SRC0_IQ1_S", "SRC0_IQ1_M"]),
    ("ksigns_iq2xs", "KSIGNS",        8, ["SRC0_IQ2_XXS", "SRC0_IQ2_XS", "SRC0_IQ3_XXS"]),
]


def read_table(src, name):
    m = re.search(r"GGML_TABLE_BEGIN\(\s*\w+\s*,\s*" + name + r"\s*,\s*[^)]+\)(.*?)GGML_TABLE_END\(\)",
                  src, re.S)
    if not m:
        raise SystemExit(f"table {name} not found")
    return [int(v, 0) for v in re.findall(r"(0x[0-9a-fA-F]+|\d+)", m.group(1))]


def emit(name, vals, per_line=8):
    out = [f"static const uint {name}[{len(vals)}] = {{"]
    for i in range(0, len(vals), per_line):
        out.append("    " + " ".join(f"0x{v:08x}u," for v in vals[i:i + per_line]))
    out.append("};")
    return "\n".join(out)


def main():
    src = open(sys.argv[1]).read()
    print("// Codebook tables for the IQ quant types, generated from ggml/src/ggml-common.h "
          "(do not edit by hand)")
    print("// regenerate with: python3 gen_iq_tables.py ../../ggml-common.h > iq_tables.hlsli")
    for tname, hname, bits, guards in TABLES:
        vals = read_table(src, tname)
        print("#if " + " || ".join(f"defined({g})" for g in guards))
        if bits == 64:
            print(f"// {tname} from ggml-common.h: 8 grid bytes per entry, "
                  "split into the low and high 32 bits")
            print(emit(hname + "_LO", [v & 0xFFFFFFFF for v in vals]))
            print(emit(hname + "_HI", [v >> 32 for v in vals]))
        else:
            n = bits // 8
            print(f"// {tname} from ggml-common.h: {n} grid byte{'s' if n > 1 else ''} per entry")
            print(emit(hname, vals))
        print("#endif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
