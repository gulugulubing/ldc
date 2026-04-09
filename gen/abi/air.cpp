//===-- gen/abi/air.cpp - AIR (Metal) ABI description -----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license.
// See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/abi/abi.h"
#include "gen/dcompute/druntime.h"
#include "gen/uda.h"
#include "dmd/declaration.h"
#include "gen/tollvm.h"
#include "gen/dcompute/abi-rewrites.h"

using namespace dmd;

struct AirTargetABI : TargetABI {
  DComputePointerRewrite pointerRewite;

  llvm::CallingConv::ID callingConv(LINK l) override {
    assert(l == LINK::c);
    return llvm::CallingConv::C;
  }

  llvm::CallingConv::ID callingConv(FuncDeclaration *fdecl) override {
    return llvm::CallingConv::C;
  }

  bool passByVal(TypeFunction *, Type *t) override {
    return DtoIsInMemoryOnly(t) && isPOD(t) && size(t) > 64;
  }

  bool returnInArg(TypeFunction *tf, bool) override {
    Type *retty = tf->next->toBasetype();
    if (retty->ty == TY::Tsarray)
      return true;
    else if (auto st = retty->isTypeStruct())
      return !toDcomputePointer(st->sym);
    else
      return false;
  }

  void rewriteArgument(IrFuncTy &fty, IrFuncTyArg &arg) override {
    TargetABI::rewriteArgument(fty, arg);
    if (arg.rewrite)
      return;

    Type *ty = arg.type->toBasetype();
    std::optional<DcomputePointer> ptr;
    if (ty->ty == TY::Tstruct &&
        (ptr = toDcomputePointer(static_cast<TypeStruct *>(ty)->sym))) {
      pointerRewite.applyTo(arg);
      arg.attrs.addAttribute("air-buffer-no-alias");
    }
  }
};

TargetABI *createAirABI() { return new AirTargetABI(); }
