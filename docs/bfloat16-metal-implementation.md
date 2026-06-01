# BF16 on Metal — Option B implementation

This document summarizes the **Brain Float16 (BF16)** work for LDC Metal dcompute:
mapping `dcompute.std.bfloat16.BFloat16` to LLVM **`bfloat`**, using native
`fpext` / `fptrunc` conversions instead of `i32 ↔ float` bitcasts (which broke
`metallib`).

**Status:** Option B implemented and validated (May 2026). Step 2 of the llm.c
roadmap is **mostly complete** — load/store, metadata, device add-one, and
**imported `@compute(hostAndDevice)` helpers in AIR** work; native BF16 GEMM
is still open.

---

## Goal

| Before (broken) | After (Option B) |
|-----------------|------------------|
| `BFloat16` → `{ i16 }` struct, `load i16` | Struct body `{ bfloat }`, buffer GEP uses `bfloat` |
| `toFloat()` via `floatBits` / bitcast | `fpext bfloat → float` on Metal device |
| `fromFloat()` via `>> 16` on `uint` | `fptrunc float → bfloat` on Metal device |
| `metallib`: Invalid bitcode | `metallib` succeeds |

Host code is unchanged: still uses 16-bit truncate / zero-extend via IEEE bit
pattern (`floatBits >> 16`). Only the **Metal device codegen path** uses native
BF16 IR.

---

## Architecture (three layers)

```text
  User D                          LDC                          Apple GPU
  ───────                         ───                          ─────────
  BFloat16.toFloat()     →   __ldc_bfloat16_to_float()   →   fpext bfloat to float
  BFloat16.fromFloat()   →   __ldc_float_to_bfloat16()   →   fptrunc float to bfloat
  GlobalPointer!BFloat16 →   DtoMemType → bfloat GEP     →   load/store bfloat AS1
  Buffer metadata        →   air.arg_type_name = "bfloat"  →   pipeline creation
```

### 1. Type recognition (`isDComputeBFloat16`)

**Files:** `gen/dcompute/druntime.h`, `gen/dcompute/druntime.cpp`

Recognizes:

- `dcompute.std.bfloat16.BFloat16` (canonical module path), or
- Any 2-byte struct named `BFloat16` / `DeviceBFloat16` with a single `ushort rep`
  field (codegen tests and device-local kernel structs).

### 2. LLVM type layout

**File:** `ir/irtypestruct.cpp`

On Metal device codegen, `BFloat16` struct IR is:

```llvm
%dcompute.std.bfloat16.BFloat16 = type { bfloat }
```

(not `{ i16 }`).

**File:** `gen/tollvm.cpp` — `DtoMemType()`

For buffer indexing (`data[i]`), GEP element type is **`bfloat`**, so AIR contains:

```llvm
%v = load bfloat, ptr addrspace(1) %p, align 1
store bfloat %w, ptr addrspace(1) %p, align 1
```

Struct assignment (`BFloat16 b = data[i]`) still uses the normal struct/memcpy
path; only **pointer element** typing uses scalar `bfloat`.

### 3. Conversion compiler hooks

**Declaration (D):** `runtime/druntime/src/ldc/dcompute.d`

```d
extern(C) float __ldc_bfloat16_to_float(ushort bits) pure @nogc nothrow;
extern(C) ushort __ldc_float_to_bfloat16(float f) pure @nogc nothrow;
```

No D body — LDC emits LLVM IR at the call site.

**Lowering (C++):** `gen/tocall.cpp` — `DtoLowerDComputeBFloat16Builtin()`

Called from `DtoCallFunction()` when `fd->ident == Id::ldcBfloat16ToFloat` (or
`ldcFloatToBfloat16`) and target is Metal:

| Hook | LLVM emitted |
|------|----------------|
| `__ldc_bfloat16_to_float(bits)` | `zext` → `bitcast i16→bfloat` → **`fpext`** to float |
| `__ldc_float_to_bfloat16(f)` | **`fptrunc`** to bfloat → `bitcast` → i16 |

**Semantic allowlist:** `gen/semantic-dcompute.cpp` — these may be called from
`@compute` code even though `ldc.dcompute` is not an `@compute` module (same
pattern as `__dcompute_reflect`).

**Identifier registry:** `dmd/id.d` + `dmd/id.h`

Entries `ldcBfloat16ToFloat` / `ldcFloatToBfloat16` map to the string names
`__ldc_bfloat16_to_float` / `__ldc_float_to_bfloat16`. The backend compares
`fd->ident == Id::ldcBfloat16ToFloat` (interned `Identifier*`, not string compare).

### 4. AIR buffer metadata

**File:** `gen/dcompute/targetMetal.cpp` — `metalBufferElementTypeName()`

For `GlobalPointer!BFloat16` kernel arguments:

- `air.arg_type_size` = 2
- `air.arg_type_name` = **`"bfloat"`** (was `"BFloat16"`)

### 5. User-facing library

**File:** `dcompute/source/dcompute/std/bfloat16.d`

```d
float toFloat() const {
    if (__dcompute_reflect(ReflectTarget.Metal, 0))
        return __ldc_bfloat16_to_float(rep);
    return floatFromBits(cast(uint)rep << 16);  // host
}

static BFloat16 fromFloat(float f) {
    if (__dcompute_reflect(ReflectTarget.Metal, 0))
        return BFloat16(__ldc_float_to_bfloat16(f));
    return BFloat16(cast(ushort)(floatBits(f) >> 16));  // host
}
```

`__dcompute_reflect(Metal)` is folded at **device codegen time** (see
`gen/statements.cpp`); host builds never call the `__ldc_*` hooks.

---

## Who calls what?

| Compile pass | Calls `__ldc_*`? | Conversion used |
|--------------|------------------|-----------------|
| Host executable | **No** | `floatBits` / bit-shift in D |
| Metal device (`-mdcompute-targets=metal-*`) | **Yes** (via `toFloat` / `fromFloat`) | `fpext` / `fptrunc` in AIR |

Users write `b.toFloat()` / `BFloat16.fromFloat(f)` in kernels; they do **not**
call `__ldc_*` directly unless needed.

---

## Cross-module AIR emit fix (May 2026)

### Problem

A **`deviceOnly`** kernel that `import`s **`@compute(hostAndDevice)`** modules
(e.g. `dcompute.std.bfloat16`) could leave helper symbols as **`declare` only**
in the AIR module → `metallib` failed with undefined D-mangled symbols.

Typical symptom: `BFloat16.__ctor`, `toFloat`, or other struct methods appeared
as `declare` in `*.air.ll` with no matching `define`.

**Workaround (removed):** duplicate `DeviceBFloat16` inside the kernel module so
definitions lived in the same AIR module as `@kernel` entry points.

### Root cause (two parts)

1. **Imported modules never device-codegen'd** — only command-line root modules
   were pushed into `computeModules` in `codegenModules()`. Imported modules
   (e.g. `dcompute.std.bfloat16`) were semantically analyzed via
   `Module::amodules` but never passed to `DComputeCodeGenManager::emit()`.

2. **`skipCodegen()` suppressed non-root symbols** — functions in imported
   modules have `inNonRoot() == true`. That is correct for host `.o` linking,
   but wrong for dcompute: all reachable `@compute` code must be merged into
   one shared AIR/PTX/SPIR-V module.

A third edge case: when **`hostAndDevice` helpers emit before the kernel**,
`semantic3` might not have run yet on unused struct methods → ICE unless
semantic is run on demand during device codegen.

### Fix (LDC)

| File | Change |
|------|--------|
| `driver/main.cpp` | After collecting root `@compute` modules, also walk `Module::amodules` and add any imported `@compute(hostAndDevice)` / `@compute(deviceOnly)` modules to the device emit list. Emit `hostAndDevice` helpers before `deviceOnly` kernels when possible. |
| `gen/function-inlining.cpp` | On the dcompute device path (`gIR->dcomputetarget`), do **not** skip codegen for functions in `@compute` modules just because they are `inNonRoot()`. |
| `gen/functions.cpp` | In `DtoDefineFunction()`, run `functionSemantic3()` on demand when emitting imported `@compute` helpers whose semantic3 has not completed yet. |

### Validation

| Test | What it checks |
|------|----------------|
| `tests/codegen/dcompute_metal_import_helper.d` | `deviceOnly` kernel imports `dcompute.std.bfloat16`; non-inlinable `addOne` helper has **`define`**, not `declare`, in AIR |
| `MetalTest/BFloat16Test/bf16_kernel.d` | Uses `import dcompute.std.bfloat16 : BFloat16` (no local `DeviceBFloat16` duplicate); `metallib` + GPU `bf16AddOne` PASS |

With default optimization, small helpers (`toFloat`, `fromFloat`, `__ctor`) may
be fully inlined into the kernel and disappear from AIR — that is fine. The
regression test uses `pragma(inline, false)` on `addOne` to force a visible
`define`.

---

## Tests

| Test | Location | What it checks |
|------|----------|----------------|
| Codegen (inline) | `tests/codegen/dcompute_metal_bfloat16.d` | `load bfloat`, `store bfloat`, no `bitcast float` |
| Codegen (import) | `tests/codegen/dcompute_metal_import_helper.d` | imported `hostAndDevice` helper **`define`** in AIR |
| E2E | `/Users/qiugaofei/MetalTest/BFloat16Test/` | `bf16Passthrough`, **`bf16AddOne`** via `import dcompute.std.bfloat16`, legacy FP32 path |

```bash
# Rebuild LDC
cd /Users/qiugaofei/dlang/ldc/build-ldc && ninja

# E2E
cd ~/MetalTest/BFloat16Test
LDC=~/dlang/ldc/build-ldc/bin/ldc2 ./build_and_test.sh
```

Expected: all three PASS; AIR shows `load bfloat` and `air.arg_type_name = "bfloat"`.

After LLVM cleanup, a simple `+1` kernel may fold to **`fadd bfloat`** instead of
explicit `fpext`/`fptrunc` — both are valid.

---

## Files changed (summary)

| Area | Files |
|------|--------|
| Type hook | `gen/dcompute/druntime.{h,cpp}`, `ir/irtypestruct.cpp`, `gen/tollvm.cpp` |
| Conversions | `gen/tocall.cpp`, `gen/semantic-dcompute.cpp`, `runtime/druntime/src/ldc/dcompute.d` |
| Metadata | `gen/dcompute/targetMetal.cpp` |
| **AIR import fix** | `driver/main.cpp`, `gen/function-inlining.cpp`, `gen/functions.cpp` |
| Identifiers | `dmd/id.d`, `dmd/id.h` |
| D library | `dcompute/source/dcompute/std/bfloat16.d` |
| Tests | `tests/codegen/dcompute_metal_bfloat16.d`, `tests/codegen/dcompute_metal_import_helper.d`, `tests/codegen/inputs/dcompute/std/bfloat16.d`, `MetalTest/BFloat16Test/*` |

---

## Not done yet (follow-ups)

- [ ] Native BF16 element-wise / GEMM kernels (Phase 2–3 of llm.c roadmap)
- [ ] Round-to-nearest-even for `fromFloat` (MLX-style; `fptrunc` may differ slightly)
- [ ] Option A fallback (text IR pipeline) if other LLVM 22 bitcode patterns still break `metallib`

---

## Related docs

- `docs/gpu-concepts.md` — BF16 vs FP16 mental model
- `docs/dcompute-metal-progress.md` — overall Metal pipeline
- `.claude.md` — project progress and roadmap
