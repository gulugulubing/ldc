// Metal: rvalue CondExp must load GlobalPointer elements in addrspace(1).

// REQUIRES: system-darwin
// RUN: %ldc -c -mdcompute-targets=metal-30 -m64 -mdcompute-file-prefix=condexp -output-ll -output-o %s
// RUN: FileCheck %s --check-prefix=AIR < condexp_metal30_64.air.ll

@compute(CompileFor.deviceOnly) module dcompute_metal_condexp;
import ldc.dcompute;

@kernel()
void loadViaTernary(GlobalPointer!(const(int)) input, GlobalPointer!int output,
                    uint idx) {
    int val = (idx < 4) ? input[idx] : 0;
    output[idx] = val;
}

// AIR: define void @loadViaTernary(
// AIR: load i32, ptr addrspace(1)
// AIR-NOT: load i32, ptr {{[^a].*}}
