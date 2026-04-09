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
// Generates LLVM IR that conforms to Apple's AIR format, then emits it as
// LLVM bitcode (.air).  The resulting file can be packaged into a .metallib
// via `xcrun metallib`.
//
// Thread-index lowering (Option 2 — on-demand synthetic intrinsic):
//
//   The dcompute library declares placeholder functions such as
//   @air.thread_position_in_grid.x via pragma(mangle, ...).  During
//   writeModule(), after running AlwaysInlinerPass to inline GlobalIndex.x
//   etc. into kernels, we scan each kernel body for calls to these
//   placeholders, inject <3 x i32> parameters for each attribute actually
//   used, replace the calls with extractelement, and emit the matching
//   AIR metadata.
//
// After emission, kernel functions get Apple-style fnattrs, then SROA /
// mem2reg / instcombine / DCE (+ dead helper removal) so IR resembles
// Metal-generated kernels; see docs/dcompute-metal-progress.md.
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
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include <string>

using namespace dmd;

namespace {

// ═══════════════════════════════════════════════════════════════════════════
// Metal attribute parameter injection (Option 2)
// ═══════════════════════════════════════════════════════════════════════════

// AIR attribute names that correspond to Metal kernel parameter attributes.
// Each one maps to a family of three placeholder functions: <base>.{x,y,z}
static const llvm::StringRef metalAttrNames[] = {
    "air.thread_position_in_grid",
    "air.thread_position_in_threadgroup",
    "air.threadgroup_position_in_grid",
    "air.threads_per_grid",
    "air.threads_per_threadgroup",
    "air.threadgroups_per_grid",
};

struct InjectedAttr {
  llvm::StringRef airName;
  unsigned paramIdx = 0;
  llvm::SmallVector<llvm::CallInst *, 4> calls[3]; // [0]=x [1]=y [2]=z
};

// Scan a function body for calls to Metal attribute placeholder functions.
// Groups them by attribute base name.
static void collectAttrCalls(llvm::Function *F,
                             llvm::SmallVector<InjectedAttr> &result) {
  llvm::StringMap<unsigned> nameToIdx;

  for (auto &BB : *F) {
    for (auto &I : BB) {
      auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
      if (!CI)
        continue;
      auto *callee = CI->getCalledFunction();
      if (!callee)
        continue;

      llvm::StringRef name = callee->getName();
      for (const auto &attrName : metalAttrNames) {
        if (!name.starts_with(attrName))
          continue;
        llvm::StringRef suffix = name.drop_front(attrName.size());
        int comp = -1;
        if (suffix == ".x")
          comp = 0;
        else if (suffix == ".y")
          comp = 1;
        else if (suffix == ".z")
          comp = 2;
        if (comp < 0)
          continue;

        auto it = nameToIdx.find(attrName);
        if (it == nameToIdx.end()) {
          nameToIdx[attrName] = result.size();
          result.push_back({attrName, 0, {}});
          it = nameToIdx.find(attrName);
        }
        result[it->second].calls[comp].push_back(CI);
        break;
      }
    }
  }
}

// Rewrite a kernel's LLVM Function to append <3 x i32> parameters for each
// Metal attribute that was called.  Returns the new Function (or the
// original if no injection was needed).
static llvm::Function *
injectAttrParams(llvm::Function *oldF,
                 llvm::SmallVector<InjectedAttr> &attrs,
                 llvm::LLVMContext &ctx) {
  if (attrs.empty())
    return oldF;

  auto *i32Ty = llvm::Type::getInt32Ty(ctx);
  auto *vec3Ty = llvm::FixedVectorType::get(i32Ty, 3);

  // New parameter type list = original params + one <3 x i32> per attribute.
  llvm::SmallVector<llvm::Type *, 8> newParamTypes;
  for (auto &arg : oldF->args())
    newParamTypes.push_back(arg.getType());

  unsigned firstNew = oldF->arg_size();
  for (unsigned i = 0; i < attrs.size(); ++i) {
    attrs[i].paramIdx = firstNew + i;
    newParamTypes.push_back(vec3Ty);
  }

  auto *newFTy =
      llvm::FunctionType::get(oldF->getReturnType(), newParamTypes, false);
  auto *newF = llvm::Function::Create(newFTy, oldF->getLinkage(), "",
                                      oldF->getParent());
  newF->takeName(oldF);
  newF->copyAttributesFrom(oldF);

  // Move basic blocks from old → new (cheap pointer splice).
  newF->splice(newF->begin(), oldF);

  // Remap old arguments → corresponding new arguments.
  {
    auto newArgIt = newF->arg_begin();
    for (auto &oldArg : oldF->args()) {
      newArgIt->takeName(&oldArg);
      oldArg.replaceAllUsesWith(&*newArgIt);
      ++newArgIt;
    }
  }

  // Replace placeholder calls with extractelement from the injected params.
  llvm::IRBuilder<> builder(ctx);
  for (auto &attr : attrs) {
    auto *param = newF->getArg(attr.paramIdx);
    param->setName(attr.airName);

    for (unsigned comp = 0; comp < 3; ++comp) {
      for (auto *CI : attr.calls[comp]) {
        builder.SetInsertPoint(CI);
        auto *elem =
            builder.CreateExtractElement(param, builder.getInt32(comp));
        CI->replaceAllUsesWith(elem);
        CI->eraseFromParent();
      }
    }
  }

  oldF->eraseFromParent();
  return newF;
}

// Remove module-level declarations of placeholder functions that have no
// remaining uses (all calls were replaced by extractelement).
static void cleanupPlaceholderDecls(llvm::Module &mod) {
  llvm::SmallVector<llvm::Function *, 16> toErase;
  for (const auto &attrName : metalAttrNames) {
    const char *suffixes[] = {".x", ".y", ".z"};
    for (const char *sfx : suffixes) {
      std::string name = (attrName.str() + sfx);
      if (auto *F = mod.getFunction(name)) {
        if (F->use_empty())
          toErase.push_back(F);
      }
    }
  }
  for (auto *F : toErase)
    F->eraseFromParent();
}

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
  B.addAttribute(llvm::Attribute::NoBuiltin);
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

/// SROA + mem2reg + instcombine + DCE to drop `GlobalPointer` alloca wrappers
/// and approach SSA like Apple-generated kernels.
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
  FPM.addPass(llvm::DCEPass());

  llvm::ModulePassManager MPM;
  MPM.addPass(
      llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.addPass(llvm::GlobalDCEPass());
  MPM.run(M, MAM);
}

/// Remove uninlined device helpers (e.g. `GlobalIndex.x` templates) so
/// `cleanupPlaceholderDecls` can drop unused `declare @air.*` lines.
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

    std::string tripleStr = "air64-apple-macosx15.0.0";
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
        "aarch64-apple-macosx15.0.0", "aarch64", "", {},
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

  void addKernelMetadata(FuncDeclaration *fd, llvm::Function * /*llf*/,
                         StructLiteralExp * /*kernAttr*/) override {
    pendingKernels_.push_back({fd, fd->mangleString});
  }

  // ── Write .air bitcode file ───────────────────────────────────────────

  void writeModule() override {
    // Step 1: Inline functions that (transitively) call Metal attribute
    // placeholders like @air.thread_position_in_grid.x into kernel bodies.
    inlineMetalPlaceholderCallers();

    // Step 2: For each kernel, detect placeholder calls and inject
    // <3 x i32> parameters for each Metal attribute actually used.
    struct KernelInfo {
      FuncDeclaration *fd;
      llvm::Function *fn;
      llvm::SmallVector<InjectedAttr> attrs;
    };
    llvm::SmallVector<KernelInfo, 4> kernels;

    for (auto &rec : pendingKernels_) {
      auto *fn = _ir->module.getFunction(rec.mangledName);
      if (!fn)
        continue;

      llvm::SmallVector<InjectedAttr> attrs;
      collectAttrCalls(fn, attrs);
      fn = injectAttrParams(fn, attrs, ctx);
      kernels.push_back({rec.fd, fn, std::move(attrs)});
    }

    cleanupPlaceholderDecls(_ir->module);

    // Step 3: Emit module-level metadata.
    addMetadata();

    // Step 4: Emit per-kernel metadata (now aware of injected attr params).
    for (auto &ki : kernels)
      emitKernelMD(ki.fd, ki.fn, ki.attrs);

    // Step 5: Kernel function attributes + IR cleanup (Apple AIR parity).
    llvm::SmallVector<llvm::Function *, 4> kernelFns;
    for (auto &ki : kernels)
      kernelFns.push_back(ki.fn);
    for (auto *kf : kernelFns)
      applyMetalKernelAttributes(ctx, kf);
    runMetalAIRCleanupPasses(_ir->module);
    eraseUnusedNonKernelFunctions(_ir->module, kernelFns);
    cleanupPlaceholderDecls(_ir->module);

    // Step 6: Build output path.
    std::string filename;
    llvm::raw_string_ostream os(filename);
    os << opts::dcomputeFilePrefix << '_' << short_name << tversion
       << "_64." << binSuffix;

    const char *path =
        FileName::combine(global.params.objdir.ptr, os.str().c_str());

    // Step 7: Verify the module before writing.
    if (llvm::verifyModule(_ir->module, &llvm::errs())) {
      error(Loc(), "Metal AIR module verification failed");
      fatal();
    }

    const auto directory = llvm::sys::path::parent_path(path);
    if (!directory.empty()) {
      if (auto ec = llvm::sys::fs::create_directories(directory)) {
        error(Loc(), "failed to create output directory: %s\n%s",
              directory.str().c_str(), ec.message().c_str());
        fatal();
      }
    }

    Logger::println("Writing Metal AIR bitcode to: %s", path);

    std::error_code errinfo;
    llvm::ToolOutputFile bos(path, errinfo, llvm::sys::fs::OF_None);
    if (bos.os().has_error()) {
      error(Loc(), "cannot write Metal AIR file '%s': %s", path,
            errinfo.message().c_str());
      fatal();
    }
    llvm::WriteBitcodeToFile(_ir->module, bos.os());
    bos.keep();

    delete _ir;
    _ir = nullptr;
  }

private:
  struct KernelRecord {
    FuncDeclaration *fd;
    const char *mangledName;
  };
  llvm::SmallVector<KernelRecord, 4> pendingKernels_;

  // ── Inline functions that transitively call Metal attribute placeholders ──

  void inlineMetalPlaceholderCallers() {
    // Mark any function that directly calls an air.* placeholder as
    // alwaysinline so the AlwaysInlinerPass will fold it into callers.
    // Repeat until no new functions are marked (handles transitive chains).
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &F : _ir->module) {
        if (F.isDeclaration() || F.hasFnAttribute(llvm::Attribute::AlwaysInline))
          continue;
        for (auto &BB : F) {
          for (auto &I : BB) {
            auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
            if (!CI || !CI->getCalledFunction())
              continue;
            llvm::StringRef name = CI->getCalledFunction()->getName();
            bool isPlaceholder = false;
            for (const auto &attrName : metalAttrNames) {
              if (name.starts_with(attrName)) {
                isPlaceholder = true;
                break;
              }
            }
            // Also treat calls to already-marked alwaysinline functions
            // as needing propagation.
            if (isPlaceholder ||
                (CI->getCalledFunction()->hasFnAttribute(
                    llvm::Attribute::AlwaysInline) &&
                 !CI->getCalledFunction()->isDeclaration())) {
              F.addFnAttr(llvm::Attribute::AlwaysInline);
              changed = true;
              goto next_function;
            }
          }
        }
      next_function:;
      }
    }

    // Now run AlwaysInlinerPass to actually inline them.
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::PassBuilder pb;
    pb.registerModuleAnalyses(mam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.registerCGSCCAnalyses(cgam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm;
    mpm.addPass(llvm::AlwaysInlinerPass());
    mpm.run(_ir->module, mam);
  }

  // ── Module flags (matches hello.ll !llvm.module.flags) ────────────────

  void addModuleFlags() {
    auto &mod = _ir->module;
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);

    llvm::Constant *sdkVer[] = {
        llvm::ConstantInt::get(i32Ty, 15),
        llvm::ConstantInt::get(i32Ty, 5)};
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

    {
      llvm::Metadata *v[] = {ci(2), ci(7), ci(0)};
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

  // ── Per-kernel metadata emission (after attribute injection) ──────────

  void emitKernelMD(FuncDeclaration *fd, llvm::Function *llf,
                    const llvm::SmallVector<InjectedAttr> &injected) {
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
          // Try to match Apple kernels: pointer args are noundef + nocapture.
          // For the Metal API, this can affect pipeline validation.
          llf->addParamAttr(i, llvm::Attribute::NoUndef);
          ++bufferIdx;
        } else {
          inputArgs.push_back(buildScalarParamMD(i, bufferIdx, v));
          llf->addParamAttr(i, llvm::Attribute::NoUndef);
          ++bufferIdx;
        }
      }
    }

    // Injected Metal attribute parameters.
    for (const auto &attr : injected) {
      inputArgs.push_back(buildAttrParamMD(attr));
      llf->addParamAttr(attr.paramIdx, llvm::Attribute::NoUndef);
    }

    llvm::Metadata *outputsMD = llvm::MDTuple::get(ctx, {});
    llvm::Metadata *inputsMD  = llvm::MDTuple::get(ctx, inputArgs);

    llvm::Metadata *entry[] = {
        llvm::ConstantAsMetadata::get(llf),
        outputsMD,
        inputsMD};
    airKernel->addOperand(llvm::MDTuple::get(ctx, entry));
  }

  // ── Build metadata for an injected Metal attribute parameter ──────────
  //
  // Shape: !{i32 <paramIdx>, !"<air_attribute_name>"}

  llvm::Metadata *buildAttrParamMD(const InjectedAttr &attr) {
    auto *i32Ty = llvm::Type::getInt32Ty(ctx);
    llvm::Metadata *fields[] = {
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(i32Ty, attr.paramIdx)),
        llvm::MDString::get(ctx, attr.airName)};
    return llvm::MDTuple::get(ctx, fields);
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
