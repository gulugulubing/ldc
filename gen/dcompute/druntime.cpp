//===-- gen/dcompute/druntime.cpp -----------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/dcompute/druntime.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/dsymbol.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/identifier.h"
#include "dmd/module.h"
#include "dmd/template.h"
#include <cstring>

using namespace dmd;

bool isFromLDC_Mod(Dsymbol *sym, Identifier* id) {
  auto mod = sym->getModule();
  if (!mod)
    return false;
  auto moduleDecl = mod->md;
  if (!moduleDecl)
    return false;

  if (moduleDecl->packages.length != 1)
    return false;
  if (moduleDecl->packages.ptr[0] != Id::ldc)
    return false;

  return moduleDecl->id == id;
}

bool isFromLDC_DCompute(Dsymbol *sym) {
  return isFromLDC_Mod(sym,Id::dcompute);
}
bool isFromLDC_OpenCL(Dsymbol *sym) {
  return isFromLDC_Mod(sym,Id::opencl);
}

static bool isFromDComputeStdMod(Dsymbol *sym, Identifier *leaf) {
  Module *mod = sym->getModule();
  if (!mod)
    return false;
  ModuleDeclaration *md = mod->md;
  if (!md || md->id != leaf)
    return false;
  if (md->packages.length != 2)
    return false;
  return md->packages.ptr[0] == Id::dcompute && md->packages.ptr[1] == Id::std;
}

bool isDComputeBFloat16(StructDeclaration *sd) {
  if (sd->ident != Id::BFloat16 && sd->ident != Id::DeviceBFloat16)
    return false;
  if (isFromDComputeStdMod(sd, Id::bfloat16))
    return true;
  // Codegen tests may embed an equivalent `BFloat16` in a different module.
  if (sd->sizeok != Sizeok::done || sd->structsize != 2 ||
      sd->fields.length != 1)
    return false;
  VarDeclaration *rep = sd->fields[0];
  return std::strcmp(rep->ident->toChars(), "rep") == 0 &&
         rep->type->toBasetype()->ty == TY::Tuns16;
}

bool isDComputeBFloat16Type(Type *t) {
  t = t->toBasetype();
  if (t->ty != TY::Tstruct)
    return false;
  return isDComputeBFloat16(static_cast<TypeStruct *>(t)->sym);
}

std::optional<DcomputePointer> toDcomputePointer(StructDeclaration *sd) {
  if (sd->ident != Id::dcPointer || !isFromLDC_DCompute(sd)) {
    return std::optional<DcomputePointer>(std::nullopt);
  }

  TemplateInstance *ti = sd->isInstantiated();
  int addrspace = isExpression((*ti->tiargs)[0])->toInteger();
  Type *type = isType((*ti->tiargs)[1]);
  return std::optional<DcomputePointer>(DcomputePointer(addrspace, type));
}
