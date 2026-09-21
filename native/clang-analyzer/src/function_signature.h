// Function declaration printing adapted from Clang 18's DeclPrinter.cpp.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/AttrLeftSideCanPrintList.inc"
#include "clang/Basic/AttrLeftSideMustPrintList.inc"
#include "llvm/Support/raw_ostream.h"

namespace cpp_context {

inline bool functionAttributeOnLeft(const clang::Attr *attribute,
                                    const clang::FunctionDecl *function) {
  using namespace clang;
  if (attribute->isDeclspecAttribute())
    return true;
#ifdef CLANG_ATTR_LIST_MustPrintOnLeft
  switch (attribute->getKind()) {
    CLANG_ATTR_LIST_MustPrintOnLeft
    return true;
  default:
    break;
  }
#endif
  if (attribute->isStandardAttributeSyntax())
    return false;
  switch (attribute->getKind()) {
    CLANG_ATTR_LIST_CanPrintOnLeft
    return function->isThisDeclarationADefinition();
  default:
    return false;
  }
}

inline bool printFunctionSignature(const clang::NamedDecl *decl,
                                   const clang::ASTContext &context,
                                   llvm::raw_ostream &out) {
  const auto *functionTemplate = llvm::dyn_cast<clang::FunctionTemplateDecl>(decl);
  const auto *function = functionTemplate ? functionTemplate->getTemplatedDecl()
                                         : llvm::dyn_cast<clang::FunctionDecl>(decl);
  if (!function)
    return false;

  // TerseOutput also erases lambdas in defaults, noexcept, types and constraints.
  // Keep Clang's full subprinters; omit only the outer body and ctor initializers.
  auto policy = context.getPrintingPolicy();
  policy.TerseOutput = false;
  const auto printExpression = [&](const clang::Expr *expression,
                                    llvm::raw_ostream &stream) {
    expression->printPretty(stream, nullptr, policy, 0, "\n", &context);
  };
  const auto printTemplateParameters = [&](const clang::TemplateParameterList *parameters) {
    parameters->print(out, context, policy);
    if (const auto *constraint = parameters->getRequiresClause()) {
      out << "requires ";
      printExpression(constraint, out);
      out << ' ';
    }
  };
  const auto printAttributes = [&](llvm::raw_ostream &stream, bool left) {
    if (policy.PolishForDeclaration)
      return;
    for (const auto *attribute : function->attrs())
      if (!attribute->isInherited() && !attribute->isImplicit() &&
          functionAttributeOnLeft(attribute, function) == left)
        attribute->printPretty(stream, policy);
  };

  if (!policy.PolishForDeclaration &&
      (functionTemplate || (!function->getDescribedFunctionTemplate() &&
                            !function->isFunctionTemplateSpecialization()))) {
    for (const auto *attribute : function->attrs()) {
      switch (attribute->getKind()) {
#define ATTR(X)
#define PRAGMA_SPELLING_ATTR(X) case clang::attr::X: attribute->printPretty(out, policy); break;
#include "clang/Basic/AttrList.inc"
      default:
        break;
      }
    }
  }
  if (!functionTemplate && function->isFunctionTemplateSpecialization()) {
    out << "template<> ";
  } else if (functionTemplate || !function->getDescribedFunctionTemplate()) {
    for (unsigned index = 0; index < function->getNumTemplateParameterLists(); ++index)
      printTemplateParameters(function->getTemplateParameterList(index));
  }
  if (functionTemplate)
    printTemplateParameters(functionTemplate->getTemplateParameters());

  std::string leftAttributes;
  llvm::raw_string_ostream leftStream(leftAttributes);
  printAttributes(leftStream, true);
  if (!leftAttributes.empty())
    out << llvm::StringRef(leftAttributes).ltrim() << ' ';
  if (!policy.SuppressSpecifiers) {
    if (function->getStorageClass() != clang::SC_None)
      out << clang::VarDecl::getStorageClassSpecifierString(function->getStorageClass()) << ' ';
    if (function->isInlineSpecified()) out << "inline ";
    if (function->isVirtualAsWritten()) out << "virtual ";
    if (function->isModulePrivate()) out << "__module_private__ ";
    if (function->isConstexprSpecified() && !function->isExplicitlyDefaulted())
      out << "constexpr ";
    if (function->isConsteval()) out << "consteval ";
    else if (function->isImmediateFunction()) out << "immediate ";
    clang::ExplicitSpecifier explicitSpecifier;
    if (const auto *constructor = llvm::dyn_cast<clang::CXXConstructorDecl>(function))
      explicitSpecifier = constructor->getExplicitSpecifier();
    else if (const auto *conversion = llvm::dyn_cast<clang::CXXConversionDecl>(function))
      explicitSpecifier = conversion->getExplicitSpecifier();
    else if (const auto *guide = llvm::dyn_cast<clang::CXXDeductionGuideDecl>(function))
      explicitSpecifier = guide->getExplicitSpecifier();
    if (explicitSpecifier.isSpecified()) {
      out << "explicit";
      if (const auto *expression = explicitSpecifier.getExpr()) {
        out << '(';
        printExpression(expression, out);
        out << ')';
      }
      out << ' ';
    }
  }

  std::string prototype;
  const auto *guide = llvm::dyn_cast<clang::CXXDeductionGuideDecl>(function);
  {
    llvm::raw_string_ostream stream(prototype);
    if (guide) {
      stream << guide->getDeducedTemplate()->getDeclName();
    } else if (policy.FullyQualifiedName) {
      stream << function->getQualifiedNameAsString();
    } else {
      if (!policy.SuppressScope)
        if (const auto *qualifier = function->getQualifier())
          qualifier->print(stream, policy);
      function->getNameInfo().printName(stream, policy);
    }
    if (function->isFunctionTemplateSpecialization()) {
      const auto *written = function->getTemplateSpecializationArgsAsWritten();
      if (written && !policy.PrintCanonicalTypes)
        clang::printTemplateArgumentList(stream, written->arguments(), policy);
      else if (const auto *arguments = function->getTemplateSpecializationArgs())
        clang::printTemplateArgumentList(stream, arguments->asArray(), policy);
    }
  }
  auto type = function->getType();
  while (const auto *parenthesized = llvm::dyn_cast<clang::ParenType>(type)) {
    prototype = '(' + prototype + ')';
    type = parenthesized->getInnerType();
  }
  const auto *functionType = type->getAs<clang::FunctionType>();
  const auto *parameterType = function->hasWrittenPrototype()
                                  ? llvm::dyn_cast<clang::FunctionProtoType>(functionType)
                                  : nullptr;
  {
    llvm::raw_string_ostream stream(prototype);
    stream << '(';
    auto parameterPolicy = policy;
    parameterPolicy.SuppressSpecifiers = false;
    for (unsigned index = 0; index < function->getNumParams(); ++index) {
      if (index) stream << ", ";
      if (parameterType)
        function->getParamDecl(index)->print(stream, parameterPolicy);
      else
        stream << function->getParamDecl(index)->getName();
    }
    if (parameterType && parameterType->isVariadic()) {
      if (function->getNumParams()) stream << ", ";
      stream << "...";
    } else if (parameterType && !function->getNumParams() && !context.getLangOpts().CPlusPlus) {
      stream << "void";
    }
    stream << ')';
    if (parameterType) {
      const auto qualifiers = parameterType->getMethodQuals().getAsString(policy);
      if (!qualifiers.empty()) stream << ' ' << qualifiers;
      if (parameterType->getRefQualifier() == clang::RQ_LValue) stream << " &";
      if (parameterType->getRefQualifier() == clang::RQ_RValue) stream << " &&";
      parameterType->printExceptionSpecification(stream, policy);
    }
  }
  if (llvm::isa<clang::CXXConstructorDecl, clang::CXXDestructorDecl,
                clang::CXXConversionDecl>(function)) {
    out << prototype;
  } else if (parameterType && parameterType->hasTrailingReturn()) {
    if (!guide) out << "auto ";
    out << prototype << " -> ";
    functionType->getReturnType().print(out, policy);
  } else {
    functionType->getReturnType().print(out, policy, prototype);
  }
  if (const auto *constraint = function->getTrailingRequiresClause()) {
    out << " requires ";
    printExpression(constraint, out);
  }
  printAttributes(out, false);
  if (function->isPureVirtual()) out << " = 0";
  else if (function->isDeletedAsWritten()) out << " = delete";
  else if (function->isExplicitlyDefaulted()) out << " = default";
  if (functionTemplate && function->hasAttr<clang::OMPDeclareTargetDeclAttr>())
    out << "#pragma omp end declare target\n";
  return true;
}

} // namespace cpp_context
