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
// Generates LLVM IR that conforms to Apple's AIR format.
// The generated IR can be assembled with llvm-as and packaged with metallib.
//
//===----------------------------------------------------------------------===//

#include "gen/dcompute/target.h"
#include "gen/dcompute/druntime.h"
#include "gen/abi/targets.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "driver/targetmachine.h"
#include "dmd/declaration.h"
#include "dmd/identifier.h"
#include "dmd/mangle.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Format.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include <string>

using namespace dmd;

namespace {

// Helper to create a pointer to an i8 constant string in address space 0.
// Metal expects strings in the default address space for metadata.
static llvm::Constant *getCStringInAddrSpace0(llvm::Module &mod, llvm::LLVMContext &ctx,
                                              llvm::StringRef str) {
  llvm::Constant *c = llvm::ConstantDataArray::getString(ctx, str, true);
  auto *global = new llvm::GlobalVariable(mod, c->getType(), true,
                                          llvm::GlobalValue::PrivateLinkage, c,
                                          ".str");
  global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  global->setAlignment(llvm::Align(1));
  return llvm::ConstantExpr::getPointerCast(global, llvm::PointerType::get(ctx, 0));
}


// -----------------------------------------------------------------------------
// Metal target implementation
// -----------------------------------------------------------------------------
class TargetMetal : public DComputeTarget {
public:
  TargetMetal(llvm::LLVMContext &c, int version)
      : DComputeTarget(c, version, ID::Metal, "metal", "air",
                       createAirABI(),
                       // Logical -> physical address space mapping:
                       // DComputeAddrSpace::private  -> 0
                       // DComputeAddrSpace::Global   -> 1 (device)
                       // DComputeAddrSpace::Shared   -> 3 (threadgroup)
                       // DComputeAddrSpace::Constant -> 2 (constant)
                       // DComputeAddrSpace::Generic  -> 0
                       {{0, 1, 3, 2, 4}}) {

    _ir = new IRState("dcomputeTargetMetal", ctx);

    // Target triple and data layout are taken from a valid AIR file.
    // This matches the output of Apple's metal compiler.
    std::string triple = "air64-apple-macosx15.0.0";
    _ir->module.setTargetTriple(llvm::Triple(triple));

    std::string dataLayout =
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-"
        "f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-"
        "v96:128:128-v128:128:128-v192:256:256-v256:256:256-"
        "v512:512:512-v1024:1024:1024-n8:16:32";
    _ir->module.setDataLayout(dataLayout);

    // Actually generating code for Metal is not currently supported,
    // just set up the target machine to be compilable.
    auto floatABI = ::FloatABI::Hard;
    targetMachine = createTargetMachine(
        "aarch64-apple-macosx15.0.0",
        "aarch64", "", {},
        ExplicitBitness::M64, floatABI,
        llvm::Reloc::Static, llvm::CodeModel::Small,
        codeGenOptLevel(), false);

    _ir->dcomputetarget = this;
  }
  void addMetadata() override {

  }
  void addKernelMetadata(FuncDeclaration *df, llvm::Function *llf,
                         StructLiteralExp *kernAttr) override {
  }


private:

};

} // anonymous namespace

// -----------------------------------------------------------------------------
// Factory function
// -----------------------------------------------------------------------------
DComputeTarget *createMetalTarget(llvm::LLVMContext &c, int version) {
  return new TargetMetal(c, version);
}