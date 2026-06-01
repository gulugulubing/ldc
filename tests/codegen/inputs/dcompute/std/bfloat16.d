@compute(CompileFor.hostAndDevice) module dcompute.std.bfloat16;

import ldc.dcompute;

align(2) struct BFloat16
{
    private align(1) ushort rep;

    @nogc pure nothrow this(ushort bits) { rep = bits; }

    pragma(inline, true)
    @nogc pure nothrow float toFloat() const
    {
        if (__dcompute_reflect(ReflectTarget.Metal, 0))
            return __ldc_bfloat16_to_float(rep);
        return floatFromBits(cast(uint)rep << 16);
    }

    pragma(inline, true)
    @nogc pure nothrow static BFloat16 fromFloat(float f)
    {
        if (__dcompute_reflect(ReflectTarget.Metal, 0))
            return BFloat16(__ldc_float_to_bfloat16(f));
        return BFloat16(cast(ushort)(floatBits(f) >> 16));
    }
}

/// Non-inlinable helper to verify imported @compute modules are defined in AIR.
pragma(inline, false)
@nogc pure nothrow float addOne(float f)
{
    return f + 1.0f;
}

@nogc pure nothrow uint floatBits(float f)
{
    return *cast(uint*)&f;
}

@nogc pure nothrow float floatFromBits(uint bits)
{
    float f;
    *cast(uint*)&f = bits;
    return f;
}
