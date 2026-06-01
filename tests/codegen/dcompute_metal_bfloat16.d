// Metal: BFloat16 maps to LLVM bfloat; conversions use fpext/fptrunc.

// REQUIRES: system-darwin
// RUN: %ldc -c -mdcompute-targets=metal-30 -m64 -mdcompute-file-prefix=bf16 -output-ll -output-o %s
// RUN: FileCheck %s --check-prefix=AIR < bf16_metal30_64.air.ll

@compute(CompileFor.deviceOnly) module dcompute_metal_bfloat16;
import ldc.dcompute;

align(2) struct BFloat16
{
    private align(1) ushort rep;

    @nogc pure nothrow this(ushort bits) { rep = bits; }

    pragma(inline, true)
    @nogc pure nothrow float toFloat() const
    {
        return __ldc_bfloat16_to_float(rep);
    }

    pragma(inline, true)
    @nogc pure nothrow static BFloat16 fromFloat(float f)
    {
        return BFloat16(__ldc_float_to_bfloat16(f));
    }
}

@kernel()
void bf16AddOne(GlobalPointer!BFloat16 data, size_t n) {
    if (n == 0)
        return;
    BFloat16 b = data[0];
    float v = b.toFloat();
    data[0] = BFloat16.fromFloat(v + 1.0f);
}

// AIR: define void @bf16AddOne(
// AIR: load bfloat, ptr addrspace(1)
// AIR: store bfloat, ptr addrspace(1)
// AIR-NOT: bitcast float
// AIR-NOT: bitcast i32
