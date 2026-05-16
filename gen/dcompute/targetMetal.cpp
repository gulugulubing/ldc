//===-- gen/dcompute/targetMetal.cpp --------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license.
// See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//
//
// Metal backend for dcompute.
// Writes LLVM bitcode to `*.air` via `llvm::WriteBitcodeToFile` (same linkage
// of IR → bitcode as the external `llvm-as` tool uses for textual `.ll`; see
// e.g. [PR draft Metal support](https://github.com/ldc-developers/ldc/pull/5118)).
// Optionally keep a textual `*.air.ll` beside the output for debugging.
// Packaging to a `.metallib` is performed by the usual `xcrun metallib` step.
//
// Thread-index lowering:
//
//   The dcompute library maps index helpers such as GlobalIndex.x directly to
//   AIR's dimension-taking intrinsics (`air.get_global_id.i32(0)` etc.).
//   No synthetic kernel parameters or index metadata are needed.
//
// After emission, kernel functions get Apple-style fnattrs, then a small LLVM
// cleanup pipeline (module inliner, SROA / mem2reg / instcombine / DCE,
// GlobalDCE), Apple bitcode compatibility fixes, and removal of dead non-kernel
// helpers so IR resembles Metal-generated kernels;
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_SUPPORTED_TARGET_METAL

#include "gen/dcompute/target.h"
#include "gen/dcompute/druntime.h"
#include "gen/abi/targets.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "driver/cl_options.h"
#include "driver/targetmachine.h"
#include "dmd/declaration.h"
#include "dmd/errors.h"
#include "dmd/globals.h"
#include "dmd/identifier.h"
#include "dmd/mangle.h"
#include "dmd/root/filename.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include <string>

using namespace dmd;

namespace {

// AIR triple / SDK banner in module metadata (`metallib` / active Xcode Metal).
constexpr char kMetalAirTargetTriple[] = "air64_v28-apple-macosx26.0.0";
constexpr int kMetalSdkVersionMajor = 26;
constexpr int kMetalSdkVersionMinor = 4;

/// Match Apple Metal AIR kernel attributes (see e.g. MetalHello/hello.ll):
/// `no-builtins`, fast-math strings, and `memory(argmem: write)` for typical
/// kernels that only touch buffer arguments.
static void applyMetalKernelAttributes(llvm::LLVMContext &C, llvm::Function *F) {
  llvm::AttrBuilder B(C);
  B.addAttribute(llvm::Attribute::MustProgress);
  B.addAttribute(llvm::Attribute::NoFree);
  B.addAttribute(llvm::Attribute::NoSync);
  B.addAttribute(llvm::Attribute::NoUnwind);
  B.addAttribute(llvm::Attribute::WillReturn);
  // Apple uses the *string* attribute "no-builtins" (not the LLVM enum
  // `nobuiltin`).  The enum form encodes differently in bitcode and crashes
  // Apple's Metal compiler ("internal error").
  B.addAttribute(llvm::Attribute::get(C, "no-builtins"));
  // Deliberately omit memory(argmem: write) — Apple's air-as / metallib
  // toolchain uses an older LLVM fork that can't parse it, and it's only
  // an optimisation hint.
  B.addAttribute(llvm::Attribute::get(C, "approx-func-fp-math", "true"));
  B.addAttribute(llvm::Attribute::get(C, "min-legal-vector-width", "0"));
  B.addAttribute(llvm::Attribute::get(C, "no-infs-fp-math", "true"));
  B.addAttribute(llvm::Attribute::get(C, "no-nans-fp-math", "true"));
  B.addAttribute(llvm::Attribute::get(C, "no-signed-zeros-fp-math", "true"));
  B.addAttribute(llvm::Attribute::get(C, "no-trapping-math", "true"));
  B.addAttribute(llvm::Attribute::get(C, "stack-protector-buffer-size", "8"));
  B.addAttribute(llvm::Attribute::get(C, "unsafe-fp-math", "true"));
  F->addFnAttrs(B);
  F->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
  // Kernels are not inlinable entry points; clear if the inliner marked them.
  if (F->hasFnAttribute(llvm::Attribute::AlwaysInline))
    F->removeFnAttr(llvm::Attribute::AlwaysInline);
}

/// Run a narrow LLVM cleanup pipeline on the AIR module.
///
/// Execution order: first `ModuleInlinerWrapperPass` (cost-based inlining),
/// then a function-pass adaptor running SROA / mem2reg / instcombine / LICM /
/// DCE for each function, then `GlobalDCE`. That tends to fold small
/// `dcompute.std` helpers into kernels, promote stack-like patterns to SSA,
/// hoist loop-invariant address math, and drop obvious dead code. This is not
/// the full `ldc_optimize_module` path used for host/CUDA output. Like SPIR-V,
/// Metal AIR is ultimately optimized by the consumer toolchain; this small pass
/// set is only for AIR-friendly cleanup before handing bitcode to Apple's Metal
/// compiler tools. On LLVM 20+, a final walk strips newer IR constructs (e.g.
/// `icmp samesign`, `zext nneg`, GEP no-wrap flags) that those tools reject
/// when ingesting our IR.
static void runMetalAIRCleanupPasses(llvm::Module &M) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::FunctionPassManager FPM;
  FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
  FPM.addPass(llvm::PromotePass());
  FPM.addPass(llvm::InstCombinePass());
  FPM.addPass(llvm::createFunctionToLoopPassAdaptor(
      llvm::LICMPass(128, 128, false), /*UseMemorySSA=*/true));
  FPM.addPass(llvm::DCEPass());

  llvm::ModulePassManager MPM;
  MPM.addPass(llvm::ModuleInlinerWrapperPass());
  MPM.addPass(
      llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.addPass(llvm::GlobalDCEPass());
  MPM.run(M, MAM);

  // InstCombine can emit newer LLVM IR constructs that Apple's Metal compiler
  // tools reject before producing a `.metallib` (`xcrun metallib` for AIR
  // bitcode, and `xcrun metal -x ir` for textual IR):
  // - `icmp samesign` (LLVM 20+)
  // - `zext nneg` (LLVM 20+)
  // - explicit GEP no-wrap flags such as `nuw` (not arithmetic `nuw`/`nsw`)
  // Strip those while preserving older, Apple-accepted forms such as
  // `getelementptr inbounds`.
#if LDC_LLVM_VER >= 2000
  for (llvm::Function &F : M) {
    if (F.isDeclaration())
      continue;
    for (llvm::BasicBlock &BB : F) {
      llvm::SmallVector<llvm::ICmpInst *, 16> todo;
      for (llvm::Instruction &I : BB) {
        auto *icmp = llvm::dyn_cast<llvm::ICmpInst>(&I);
        if (icmp && icmp->hasSameSign()) {
          todo.push_back(icmp);
        }

        if (auto *zext = llvm::dyn_cast<llvm::ZExtInst>(&I)) {
          if (zext->hasNonNeg()) {
            zext->setNonNeg(false);
          }
        }

        if (auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&I)) {
          if (gep->getNoWrapFlags() != llvm::GEPNoWrapFlags::none()) {
            gep->setNoWrapFlags(gep->isInBounds()
                                    ? llvm::GEPNoWrapFlags::inBounds()
                                    : llvm::GEPNoWrapFlags::none());
          }
        }
      }
      for (llvm::ICmpInst *icmp : todo) {
        llvm::IRBuilder<> b(icmp);
        llvm::Value *rep =
            b.CreateICmp(icmp->getPredicate(), icmp->getOperand(0),
                         icmp->getOperand(1));
        rep->takeName(icmp);
        icmp->replaceAllUsesWith(rep);
        icmp->eraseFromParent();
      }
    }
  }
#endif // LDC_LLVM_VER >= 2000
}

/// Drop leftover helper definitions after inlining.
///
/// D template helpers from `dcompute.std` are emitted as `weak_odr` functions.
/// LLVM's GlobalDCE keeps unused externally visible definitions because a normal
/// object-file link may still need them. A Metal AIR module is different: the
/// only definitions that should remain externally visible are kernel entry
/// points. Other unused definitions are just inlined device helpers and can
/// confuse the AIR output, so remove them once all kernel bodies are finalized.
static void eraseUnusedNonKernelFunctions(
    llvm::Module &M, llvm::ArrayRef<llvm::Function *> keepKernels) {
  llvm::SmallPtrSet<llvm::Function *, 8> keep(keepKernels.begin(),
                                               keepKernels.end());
  llvm::SmallVector<llvm::Function *, 16> dead;
  for (auto &F : M) {
    if (F.isDeclaration() || keep.count(&F))
      continue;
    if (F.use_empty())
      dead.push_back(&F);
  }
  for (auto *F : dead)
    F->eraseFromParent();
}

// ═══════════════════════════════════════════════════════════════════════════
// TargetMetal
// ═══════════════════════════════════════════════════════════════════════════

class TargetMetal : public DComputeTarget {
public:
  TargetMetal(llvm::LLVMContext &c, int version)
      : DComputeTarget(c, version, ID::Metal, "metal", "air",
                       createAirABI(),
                       // DCompute logical -> AIR physical address space:
                       //   Private(0)  -> 0
                       //   Global(1)   -> 1  (device)
                       //   Shared(2)   -> 3  (threadgroup)
                       //   Constant(3) -> 2  (constant)
                       //   Generic(4)  -> 4
                       {{0, 1, 3, 2, 4}}) {

    _ir = new IRState("dcomputeTargetMetal", ctx);

    // Align with Metallib expectations for Xcode 26-era Metal toolchain.
    std::string tripleStr(kMetalAirTargetTriple);
#if LDC_LLVM_VER >= 2100
    _ir->module.setTargetTriple(llvm::Triple(tripleStr));
#else
    _ir->module.setTargetTriple(tripleStr);
#endif

    _ir->module.setDataLayout(
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
        "-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64"
        "-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256"
        "-v512:512:512-v1024:1024:1024-n8:16:32");

    // AIR has no real LLVM backend; use AArch64 for internal analysis only.
    auto floatABI = ::FloatABI::Hard;
    targetMachine = createTargetMachine(
        "aarch64-apple-macosx26.0.0", "aarch64", "", {},
        ExplicitBitness::M64, floatABI,
        llvm::Reloc::Static, llvm::CodeModel::Small,
        codeGenOptLevel(), false);

    _ir->dcomputetarget = this;
  }

  // ── Module-level metadata ─────────────────────────────────────────────

  void addMetadata() override {
    addModuleFlags();
    addCompileOptions();
    addVersionInfo();
    addSourceFileName();
    addLLVMIdent();
  }

  // ── Per-kernel registration (deferred — metadata is emitted later) ────

  void addKernelMetadata(FuncDeclaration *fd, llvm::Function *llf,
                         StructLiteralExp * /*kernAttr*/) override {
    pendingKernels_.push_back({fd, llf});
  }

  // ── Write .air bitcode file ───────────────────────────────────────────

  void writeModule() override {
    // Step 1: Collect kernel functions for metadata and cleanup.
    struct KernelInfo {
      FuncDeclaration *fd;
      llvm::Function *fn;
    };
    llvm::SmallVector<KernelInfo, 4> kernels;

    for (auto &rec : pendingKernels_) {
      if (!rec.fn)
        continue;

      kernels.push_back({rec.fd, rec.fn});
    }

    // Step 2: Rename kernel functions to their plain D names.
    // Metal's host API (newFunctionWithName:) looks up kernels by name, so
    // expose `void mmul(...)` as `mmul` instead of the D-mangled symbol.
    for (auto &ki : kernels) {
      ki.fn->setName(ki.fd->ident->toChars());
    }

    // Step 3: Emit module-level metadata.
    addMetadata();

    // Step 4: Emit per-kernel metadata.
    for (auto &ki : kernels)
      emitKernelMD(ki.fd, ki.fn);

    // Step 5: Kernel function attributes + IR cleanup (Apple AIR parity).
    llvm::SmallVector<llvm::Function *, 4> kernelFns;
    for (auto &ki : kernels)
      kernelFns.push_back(ki.fn);
    for (auto *kf : kernelFns)
      applyMetalKernelAttributes(ctx, kf);
    runMetalAIRCleanupPasses(_ir->module);
    eraseUnusedNonKernelFunctions(_ir->module, kernelFns);

    // Step 6: Build output path.
    std::string filename;
    llvm::raw_string_ostream os(filename);
    os << opts::dcomputeFilePrefix << '_' << short_name << tversion
       << "_64." << binSuffix;

    const char *outPath =
        FileName::combine(global.params.objdir.ptr, os.str().c_str());

    // Step 7: Verify the module before writing.
    if (llvm::verifyModule(_ir->module, &llvm::errs())) {
      error(Loc(), "Metal AIR module verification failed");
      fatal();
    }

    const auto directory = llvm::sys::path::parent_path(outPath);
    if (!directory.empty()) {
      if (auto ec = llvm::sys::fs::create_directories(directory)) {
        error(Loc(), "failed to create output directory: %s\n%s",
              directory.str().c_str(), ec.message().c_str());
        fatal();
      }
    }

    Logger::println("Writing Metal AIR bitcode and textual IR beside output: "
                    "%s",
                    outPath);

    // Debugging: raw Module print (upstream LLVM dialect; same representation
    // that `llvm-as` would ingest from a textual file).
    const std::string llPathStr = std::string(outPath) + ".ll";
    {
      std::error_code wec;
      llvm::raw_fd_ostream llOs(llPathStr, wec, llvm::sys::fs::OF_Text);
      if (wec) {
        error(Loc(), "cannot write Metal LLVM IR file '%s': %s",
              llPathStr.c_str(), wec.message().c_str());
        fatal();
      }
      _ir->module.print(llOs, nullptr);
    }

    {
      std::error_code ec;
      llvm::raw_fd_ostream airOs(outPath, ec, llvm::sys::fs::OF_None);
      if (ec) {
        error(Loc(), "cannot write Metal AIR bitcode '%s': %s", outPath,
              ec.message().c_str());
        fatal();
      }
      llvm::WriteBitcodeToFile(_ir->module, airOs);
    }

    delete _ir;
    _ir = nullptr;
  }

private:
  struct KernelRecord {
    FuncDeclaration *fd;
    llvm::Function *fn;
  };
  llvm::SmallVector<KernelRecord, 4> pendingKernels_;

  // ── Module flags (matches hello.ll !llvm.module.flags) ────────────────

  void addModuleFlags() {
    auto &mod = _ir->module;
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);

    llvm::Constant *sdkVer[] = {
        llvm::ConstantInt::get(i32Ty, kMetalSdkVersionMajor),
        llvm::ConstantInt::get(i32Ty, kMetalSdkVersionMinor)};
    mod.addModuleFlag(llvm::Module::Warning, "SDK Version",
                      llvm::ConstantArray::get(
                          llvm::ArrayType::get(i32Ty, 2), sdkVer));

    mod.addModuleFlag(llvm::Module::Error, "wchar_size", 4);
    mod.addModuleFlag(llvm::Module::Max, "frame-pointer", 2);
    mod.addModuleFlag(llvm::Module::Max, "air.max_device_buffers", 31);
    mod.addModuleFlag(llvm::Module::Max, "air.max_constant_buffers", 31);
    mod.addModuleFlag(llvm::Module::Max, "air.max_threadgroup_buffers", 31);
    mod.addModuleFlag(llvm::Module::Max, "air.max_textures", 128);
    mod.addModuleFlag(llvm::Module::Max, "air.max_read_write_textures", 8);
    mod.addModuleFlag(llvm::Module::Max, "air.max_samplers", 16);
  }

  // ── Compile options ───────────────────────────────────────────────────

  void addCompileOptions() {
    llvm::NamedMDNode *node =
        _ir->module.getOrInsertNamedMetadata("air.compile_options");
    auto addOpt = [&](llvm::StringRef opt) {
      node->addOperand(
          llvm::MDTuple::get(ctx, {llvm::MDString::get(ctx, opt)}));
    };
    addOpt("air.compile.denorms_disable");
    addOpt("air.compile.fast_math_enable");
    addOpt("air.compile.framebuffer_fetch_enable");
  }

  // ── AIR / Metal language version ──────────────────────────────────────

  void addVersionInfo() {
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);
    auto ci = [&](int v) -> llvm::Metadata * {
      return llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32Ty, v));
    };

    // AIR ABI version tracked by Xcode / Metal toolchain.  Older toolchains
    // accepted 2.7.x; current Apple `metal`/metallib expects 2.8 for macOS 26 /
    // Metal toolchain 32023-era SDKs—keep aligned with `-triple air64_v28-...`.
    {
      llvm::Metadata *v[] = {ci(2), ci(8), ci(0)};
      _ir->module.getOrInsertNamedMetadata("air.version")
          ->addOperand(llvm::MDTuple::get(ctx, v));
    }
    {
      llvm::Metadata *v[] = {llvm::MDString::get(ctx, "Metal"),
                             ci(3), ci(2), ci(0)};
      _ir->module.getOrInsertNamedMetadata("air.language_version")
          ->addOperand(llvm::MDTuple::get(ctx, v));
    }
  }

  // ── Source file name ──────────────────────────────────────────────────

  void addSourceFileName() {
    auto srcName = _ir->module.getSourceFileName();
    llvm::NamedMDNode *node =
        _ir->module.getOrInsertNamedMetadata("air.source_file_name");
    node->addOperand(llvm::MDTuple::get(
        ctx, {llvm::MDString::get(ctx, srcName)}));
  }

  // ── Producer string (parity with Apple bitcode) ───────────────────────

  void addLLVMIdent() {
    llvm::NamedMDNode *ident =
        _ir->module.getOrInsertNamedMetadata("llvm.ident");
    ident->addOperand(llvm::MDTuple::get(
        ctx, {llvm::MDString::get(ctx, "LDC DCompute Metal (AIR)")}));
  }

  // ── Per-kernel metadata emission ──────────────────────────────────────

  void emitKernelMD(FuncDeclaration *fd, llvm::Function *llf) {
    llvm::NamedMDNode *airKernel =
        _ir->module.getOrInsertNamedMetadata("air.kernel");

    llvm::SmallVector<llvm::Metadata *, 8> inputArgs;
    unsigned bufferIdx = 0;

    // Original D parameters → buffer / scalar metadata.
    VarDeclarations *params = fd->parameters;
    if (params) {
      for (unsigned i = 0; i < params->length; ++i) {
        VarDeclaration *v = (*params)[i];
        Type *baseTy = v->type->toBasetype();

        std::optional<DcomputePointer> ptr;
        if (baseTy->ty == TY::Tstruct &&
            (ptr = toDcomputePointer(
                 static_cast<TypeStruct *>(baseTy)->sym))) {
          inputArgs.push_back(buildBufferParamMD(i, bufferIdx, *ptr, v));
          // Match Apple-generated kernel signatures: buffer pointer params
          // carry nocapture, noundef, writeonly/readonly, and air-buffer-no-alias.
          llf->addParamAttr(i, llvm::Attribute::NoUndef);
#if LDC_LLVM_VER >= 2100
          {
            llvm::AttrBuilder ab(ctx);
            ab.addCapturesAttr(llvm::CaptureInfo::none());
            llf->addParamAttrs(i, ab);
          }
#else
          llf->addParamAttr(i, llvm::Attribute::NoCapture);
#endif
          {
            bool isConst = (ptr->type->mod & (MODconst | MODimmutable)) != 0;
            if (isConst)
              llf->addParamAttr(i, llvm::Attribute::ReadOnly);
            else
              llf->addParamAttr(i, llvm::Attribute::WriteOnly);
          }
          llf->addParamAttr(i,
              llvm::Attribute::get(ctx, "air-buffer-no-alias"));
          ++bufferIdx;
        } else {
          inputArgs.push_back(buildScalarParamMD(i, bufferIdx, v));
          llf->addParamAttr(i, llvm::Attribute::NoUndef);
          ++bufferIdx;
        }
      }
    }

    llvm::Metadata *outputsMD = llvm::MDTuple::get(ctx, {});
    llvm::Metadata *inputsMD  = llvm::MDTuple::get(ctx, inputArgs);

    llvm::Metadata *entry[] = {
        llvm::ConstantAsMetadata::get(llf),
        outputsMD,
        inputsMD};
    airKernel->addOperand(llvm::MDTuple::get(ctx, entry));
  }

  // ── Build per-parameter AIR metadata nodes ────────────────────────────

  llvm::Metadata *buildBufferParamMD(unsigned paramIdx, unsigned bufIdx,
                                     const DcomputePointer &ptr,
                                     VarDeclaration *v) {
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);
    auto ci = [&](int val) -> llvm::Metadata * {
      return llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(i32Ty, val));
    };
    auto ms = [&](llvm::StringRef s) -> llvm::Metadata * {
      return llvm::MDString::get(ctx, s);
    };

    int physAS = mapping[ptr.addrspace];
    unsigned elemSize = size(ptr.type);
    unsigned elemAlign = ptr.type->alignsize();

    bool isConst = (ptr.type->mod & (MODconst | MODimmutable)) != 0;
    const char *accessMode = isConst ? "air.read" : "air.read_write";

    std::string typeName = ptr.type->toChars();

    llvm::Metadata *fields[] = {
        ci(paramIdx),
        ms("air.buffer"),
        ms("air.location_index"), ci(bufIdx),
        ci(1),
        ms(accessMode),
        ms("air.address_space"),   ci(physAS),
        ms("air.arg_type_size"),   ci(elemSize),
        ms("air.arg_type_align_size"), ci(elemAlign),
        ms("air.arg_type_name"),   ms(typeName),
        ms("air.arg_name"),        ms(v->ident->toChars())};
    return llvm::MDTuple::get(ctx, fields);
  }

  llvm::Metadata *buildScalarParamMD(unsigned paramIdx, unsigned bufIdx,
                                     VarDeclaration *v) {
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);
    auto ci = [&](int val) -> llvm::Metadata * {
      return llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(i32Ty, val));
    };
    auto ms = [&](llvm::StringRef s) -> llvm::Metadata * {
      return llvm::MDString::get(ctx, s);
    };

    Type *baseTy = v->type->toBasetype();
    unsigned sz = size(baseTy);
    unsigned al = baseTy->alignsize();
    std::string typeName = baseTy->toChars();

    llvm::Metadata *fields[] = {
        ci(paramIdx),
        ms("air.buffer"),
        ms("air.location_index"), ci(bufIdx),
        ci(1),
        ms("air.read"),
        ms("air.address_space"),   ci(2),
        ms("air.arg_type_size"),   ci(sz),
        ms("air.arg_type_align_size"), ci(al),
        ms("air.arg_type_name"),   ms(typeName),
        ms("air.arg_name"),        ms(v->ident->toChars())};
    return llvm::MDTuple::get(ctx, fields);
  }
};

} // anonymous namespace

DComputeTarget *createMetalTarget(llvm::LLVMContext &c, int version) {
  return new TargetMetal(c, version);
}

#endif // LDC_LLVM_SUPPORTED_TARGET_METAL
