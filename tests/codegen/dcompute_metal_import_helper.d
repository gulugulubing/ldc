// Metal: imported @compute(hostAndDevice) helpers must be defined in AIR,
// not left as declare-only symbols (metallib link failure otherwise).

// REQUIRES: system-darwin
// RUN: %ldc -c -mdcompute-targets=metal-30 -m64 -mdcompute-file-prefix=import_helper -output-ll -I%S/inputs %s
// RUN: FileCheck %s --check-prefix=AIR < import_helper_metal30_64.air.ll

@compute(CompileFor.deviceOnly) module dcompute_metal_import_helper;
import ldc.dcompute;
import dcompute.std.bfloat16 : BFloat16, addOne;

@kernel()
void bf16AddOne(GlobalPointer!BFloat16 data, size_t n) {
    if (n == 0)
        return;
    BFloat16 b = data[0];
    float v = addOne(b.toFloat());
    data[0] = BFloat16.fromFloat(v);
}

// AIR: define void @bf16AddOne(
// AIR-NOT: declare float @_D8dcompute3std8bfloat166addOneFNaNbNifZf(
// AIR: define float @_D8dcompute3std8bfloat166addOneFNaNbNifZf(
