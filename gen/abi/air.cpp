#include "gen/abi/abi.h"
#include "gen/dcompute/druntime.h"
#include "gen/uda.h"
#include "dmd/declaration.h"
#include "gen/tollvm.h"
#include "gen/dcompute/abi-rewrites.h"

using namespace dmd;

struct AirTargetABI : TargetABI {
  DComputePointerRewrite pointerRewite;  // Used to rewrite GlobalPointer

  // Calling convention: Metal device functions typically have no special
  // conventions; the default C calling convention can be used.
  // Kernels may need special marking (e.g., via attributes rather than
  // calling convention)
  llvm::CallingConv::ID callingConv(LINK l) override {
      assert(l == LINK::c);
      // For ordinary device functions, use the default CC (e.g., llvm::CallingConv::C)
      return llvm::CallingConv::C;
  }

  llvm::CallingConv::ID callingConv(FuncDeclaration *fdecl) override {
    // Metal kernels don't need special calling conventions, but we still need to distinguish kernel functions
    // so that metadata can be attached in addKernelMetadata. Here we can return the same CC.
    // If needed, attributes (e.g., "kernel") can be set on kernel functions instead of changing the calling convention.
    return llvm::CallingConv::C;
  }

  bool passByVal(TypeFunction *, Type *t) override {
    // Determine whether to pass by pointer. Metal kernel parameters are usually
    // pointers or simple integers; large structs are unlikely. For generality,
    // keep the heuristic used by OpenCL.
    return DtoIsInMemoryOnly(t) && isPOD(t) && size(t) > 64;
  }

  bool returnInArg(TypeFunction *tf, bool) override {
    // Metal kernels return void, but device functions may return values.
    // For structs or static arrays, an implicit pointer may be needed.
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
    // If the parameter is a DComputePointer struct (e.g., GlobalPointer!T),
    // rewrite it to a pointer.
    if (ty->ty == TY::Tstruct &&
        (ptr = toDcomputePointer(static_cast<TypeStruct *>(ty)->sym))) {
      pointerRewite.applyTo(arg);
      // Note: The correct address space must be set according to the mapping table.
      // pointerRewite internally uses the address space mapping provided by TargetABI.
      // For Metal
      arg.byref = true;
      arg.attrs.addAttribute(llvm::Attribute::NoAlias);
      arg.attrs.addAttribute(llvm::Attribute::WriteOnly);
      arg.attrs.addAttribute("air-buffer-no-alias");
    }
  }
};

TargetABI *createAirABI() { return new AirTargetABI(); }