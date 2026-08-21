#include <algorithm>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace {

static_assert(CLANG_VERSION_MAJOR == 18, "cpp-context-clang-analyzer requires Clang 18");

constexpr llvm::StringLiteral kProtocol = "cpp-context-clang-facts";
constexpr std::int64_t kProtocolVersion = 3;
constexpr std::int64_t kClangMajor = 18;

const std::vector<std::string> kCapabilities = {
    "direct_calls",       "full_ast",          "function_cfg_v1",
    "includes",
    "inherits",           "lambda_metadata",   "macro_provenance",
    "occurrences",        "overrides",         "pp_callbacks",
    "source_manager",     "symbols",           "template_metadata",
    "uses_type",          "callsites_v1",     "dispatch_targets_v1",
    "macro_expansion_stack", "template_relationships_v1"};

void emit(llvm::json::Object object) {
  llvm::outs() << llvm::formatv("{0}\n", llvm::json::Value(std::move(object)));
  llvm::outs().flush();
}

void emit(std::initializer_list<llvm::json::Object::KV> properties) {
  emit(llvm::json::Object(properties));
}

void emitError(llvm::StringRef code, llvm::StringRef message) {
  emit({{"type", "error"}, {"code", code}, {"message", message}});
}

std::optional<std::string> requiredString(const llvm::json::Object &object,
                                          llvm::StringRef name) {
  if (auto value = object.getString(name))
    return value->str();
  return std::nullopt;
}

class FactSink {
public:
  void add(std::string sortKey, llvm::json::Object fact) {
    fact["type"] = "fact";
    entries_.emplace_back(std::move(sortKey), std::move(fact));
  }

  void add(std::string sortKey,
           std::initializer_list<llvm::json::Object::KV> properties) {
    add(std::move(sortKey), llvm::json::Object(properties));
  }

  void flush() {
    std::stable_sort(entries_.begin(), entries_.end(),
                     [](const auto &left, const auto &right) {
                       return left.first < right.first;
                     });
    std::string previous;
    bool first = true;
    for (auto &entry : entries_) {
      if (!first && entry.first == previous)
        continue;
      previous = entry.first;
      first = false;
      emit(std::move(entry.second));
    }
    entries_.clear();
  }

private:
  std::vector<std::pair<std::string, llvm::json::Object>> entries_;
};

struct MacroExpansionRecord {
  std::string name;
  std::string key;
  clang::SourceRange definitionRange;
  clang::SourceRange expansionRange;
};

class SourceFacts {
public:
  SourceFacts(clang::SourceManager &sourceManager, const clang::LangOptions &langOptions,
              std::filesystem::path projectRoot)
      : sourceManager_(sourceManager), langOptions_(langOptions),
        projectRoot_(canonical(std::move(projectRoot))) {}

  static std::filesystem::path canonical(std::filesystem::path path) {
    std::error_code error;
    auto result = std::filesystem::weakly_canonical(path, error);
    return error ? path.lexically_normal() : result;
  }

  std::optional<std::filesystem::path> path(clang::SourceLocation location,
                                            bool spelling = true) const {
    if (location.isInvalid())
      return std::nullopt;
    const auto resolved = spelling ? sourceManager_.getSpellingLoc(location)
                                   : sourceManager_.getExpansionLoc(location);
    auto filename = sourceManager_.getFilename(resolved);
    if (filename.empty())
      return std::nullopt;
    return canonical(filename.str());
  }

  bool isProjectPath(const std::filesystem::path &candidate) const {
    auto relative = canonical(candidate).lexically_relative(projectRoot_);
    return !relative.empty() && *relative.begin() != "..";
  }

  std::optional<std::string> relative(clang::SourceLocation location,
                                      bool spelling = true) const {
    auto candidate = path(location, spelling);
    if (!candidate || !isProjectPath(*candidate))
      return std::nullopt;
    return canonical(*candidate).lexically_relative(projectRoot_).generic_string();
  }

  std::optional<llvm::json::Object> span(clang::SourceRange range,
                                        bool spelling = true) const {
    clang::SourceLocation begin = spelling ? sourceManager_.getSpellingLoc(range.getBegin())
                                           : sourceManager_.getExpansionLoc(range.getBegin());
    clang::SourceLocation end = spelling ? sourceManager_.getSpellingLoc(range.getEnd())
                                         : sourceManager_.getExpansionLoc(range.getEnd());
    if (begin.isInvalid() || end.isInvalid())
      return std::nullopt;
    bool endIsExclusive = false;
    if (!sourceManager_.isWrittenInSameFile(begin, end)) {
      // A declaration ending in a macro can otherwise combine source and definition files.
      auto fileRange = clang::Lexer::makeFileCharRange(
          clang::CharSourceRange::getTokenRange(range), sourceManager_, langOptions_);
      fileRange = clang::Lexer::getAsCharRange(fileRange, sourceManager_, langOptions_);
      if (fileRange.isInvalid())
        return std::nullopt;
      begin = fileRange.getBegin();
      end = fileRange.getEnd();
      endIsExclusive = true;
    }
    auto candidate = path(begin, spelling);
    if (!candidate || !isProjectPath(*candidate))
      return std::nullopt;
    if (!endIsExclusive) {
      auto endToken = clang::Lexer::getLocForEndOfToken(end, 0, sourceManager_, langOptions_);
      if (endToken.isValid())
        end = endToken;
    }
    return llvm::json::Object{
        {"path", canonical(*candidate).string()},
        {"start_line", static_cast<std::int64_t>(sourceManager_.getSpellingLineNumber(begin))},
        {"start_column",
         static_cast<std::int64_t>(sourceManager_.getSpellingColumnNumber(begin))},
        {"end_line", static_cast<std::int64_t>(sourceManager_.getSpellingLineNumber(end))},
        {"end_column", static_cast<std::int64_t>(sourceManager_.getSpellingColumnNumber(end))}};
  }

  std::string source(clang::SourceRange range) const {
    bool invalid = false;
    auto text = clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(range),
                                            sourceManager_, langOptions_, &invalid);
    return invalid ? std::string{} : text.str();
  }

  std::int64_t offset(clang::SourceLocation location, bool spelling = true) const {
    auto resolved = spelling ? sourceManager_.getSpellingLoc(location)
                             : sourceManager_.getExpansionLoc(location);
    return resolved.isValid()
               ? static_cast<std::int64_t>(sourceManager_.getFileOffset(resolved))
               : 0;
  }

  std::pair<std::int64_t, std::int64_t> offsets(clang::SourceRange range) const {
    auto fileRange = clang::Lexer::makeFileCharRange(
        clang::CharSourceRange::getTokenRange(range), sourceManager_, langOptions_);
    fileRange = clang::Lexer::getAsCharRange(fileRange, sourceManager_, langOptions_);
    if (fileRange.isInvalid()) {
      auto end = sourceManager_.getSpellingLoc(range.getEnd());
      auto endToken = clang::Lexer::getLocForEndOfToken(end, 0, sourceManager_, langOptions_);
      return {offset(range.getBegin()), offset(endToken.isValid() ? endToken : end)};
    }
    return {static_cast<std::int64_t>(sourceManager_.getFileOffset(fileRange.getBegin())),
            static_cast<std::int64_t>(sourceManager_.getFileOffset(fileRange.getEnd()))};
  }

  llvm::json::Array
  expansionStack(clang::SourceLocation location,
                 const std::vector<MacroExpansionRecord> &records) const {
    llvm::json::Array result;
    auto current = location;
    for (unsigned depth = 0; current.isMacroID() && depth < 64; ++depth) {
      auto immediate = sourceManager_.getImmediateExpansionRange(current).getAsRange();
      const MacroExpansionRecord *matched = nullptr;
      for (auto iterator = records.rbegin(); iterator != records.rend(); ++iterator) {
        if (iterator->expansionRange.getBegin() == immediate.getBegin()) {
          matched = &*iterator;
          break;
        }
      }
      if (!matched) {
        for (auto iterator = records.rbegin(); iterator != records.rend(); ++iterator) {
          if (offset(iterator->expansionRange.getBegin(), false) ==
                  offset(immediate.getBegin(), false) &&
              path(iterator->expansionRange.getBegin(), false) ==
                  path(immediate.getBegin(), false)) {
            matched = &*iterator;
            break;
          }
        }
      }
      if (matched) {
        auto spelling = span(matched->definitionRange, true);
        auto expansion = span(immediate, false);
        if (spelling && expansion)
          result.push_back(llvm::json::Object{{"macro_key", matched->key},
                                              {"name", matched->name},
                                              {"spelling_span", std::move(*spelling)},
                                              {"expansion_span", std::move(*expansion)}});
      }
      auto next = sourceManager_.getImmediateMacroCallerLoc(current);
      if (next == current)
        break;
      current = next;
    }
    return result;
  }

  std::string fileKey(const std::filesystem::path &path) const {
    return "file:" + canonical(path).lexically_relative(projectRoot_).generic_string();
  }

  std::string declKey(const clang::NamedDecl *decl, llvm::StringRef kind) const {
    if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(decl)) {
      if (method->isLambdaStaticInvoker() ||
          (method->getParent() && method->getParent()->isLambda() &&
           method->getOverloadedOperator() == clang::OO_Call)) {
        const auto loc = sourceManager_.getSpellingLoc(method->getParent()->getBeginLoc());
        auto rel = relative(loc).value_or("unknown");
        auto key = "lambda:" + rel + ":" +
                   std::to_string(sourceManager_.getSpellingLineNumber(loc)) + ":" +
                   std::to_string(sourceManager_.getSpellingColumnNumber(loc)) + ":operator()";
        if (method->getTemplateSpecializationKind() != clang::TSK_Undeclared) {
          llvm::SmallString<256> specializationUsr;
          if (!clang::index::generateUSRForDecl(method, specializationUsr) &&
              !specializationUsr.empty())
            key += ":specialization:" + specializationUsr.str().str();
        }
        return key;
      }
    }
    llvm::SmallString<256> usr;
    if (!clang::index::generateUSRForDecl(decl, usr) && !usr.empty())
      return "usr:" + usr.str().str();
    auto rel = relative(decl->getLocation()).value_or("unknown");
    auto loc = sourceManager_.getSpellingLoc(decl->getLocation());
    return "fallback:" + kind.str() + ":" + rel + ":" +
           decl->getQualifiedNameAsString() + ":" +
           std::to_string(sourceManager_.getSpellingLineNumber(loc)) + ":" +
           std::to_string(sourceManager_.getSpellingColumnNumber(loc));
  }

  std::string macroKey(llvm::StringRef name, clang::SourceLocation location) const {
    llvm::SmallString<256> usr;
    if (!clang::index::generateUSRForMacro(name, location, sourceManager_, usr) && !usr.empty())
      return "usr:" + usr.str().str();
    auto candidate = path(location).value_or(projectRoot_ / "unknown");
    return "macro:" + fileKey(candidate) + ":" + name.str() + ":" +
           std::to_string(sourceManager_.getSpellingLineNumber(location)) + ":" +
           std::to_string(sourceManager_.getSpellingColumnNumber(location));
  }

private:
  clang::SourceManager &sourceManager_;
  const clang::LangOptions &langOptions_;
  std::filesystem::path projectRoot_;
};

class Collector;

class PreprocessorCollector final : public clang::PPCallbacks {
public:
  PreprocessorCollector(FactSink &sink, SourceFacts &source,
                        std::vector<MacroExpansionRecord> &macroExpansions)
      : sink_(sink), source_(source), macroExpansions_(macroExpansions) {}

  void InclusionDirective(clang::SourceLocation hashLoc, const clang::Token &,
                          llvm::StringRef fileName, bool isAngled,
                          clang::CharSourceRange filenameRange,
                          clang::OptionalFileEntryRef file, llvm::StringRef,
                          llvm::StringRef, const clang::Module *,
                          clang::SrcMgr::CharacteristicKind) override {
    auto from = source_.path(hashLoc);
    if (!from || !source_.isProjectPath(*from))
      return;
    llvm::json::Object fact{{"fact", "include"},
                            {"source_key", source_.fileKey(*from)},
                            {"written", fileName.str()},
                            {"angled", isAngled}};
    if (auto span = source_.span(filenameRange.getAsRange()))
      fact["span"] = std::move(*span);
    if (file) {
      auto target = SourceFacts::canonical(std::filesystem::path(file->getName().str()));
      if (source_.isProjectPath(target)) {
        fact["target_key"] = source_.fileKey(target);
        fact["resolved_path"] = target.string();
        sink_.add("include:" + from->string() + ":" + target.string() + ":" +
                      std::to_string(source_.offset(hashLoc, false)),
                  std::move(fact));
      }
    }
  }

  void MacroDefined(const clang::Token &macroName, const clang::MacroDirective *directive) override {
    const auto *info = directive ? directive->getMacroInfo() : nullptr;
    if (!info || !macroName.getIdentifierInfo())
      return;
    auto path = source_.path(info->getDefinitionLoc());
    if (!path || !source_.isProjectPath(*path))
      return;
    auto range = clang::SourceRange(info->getDefinitionLoc(), info->getDefinitionEndLoc());
    auto span = source_.span(range);
    if (!span)
      return;
    auto name = macroName.getIdentifierInfo()->getName().str();
    auto key = source_.macroKey(name, info->getDefinitionLoc());
    llvm::json::Object metadata{{"is_definition", true},
                                {"analysis_backend", "clang-libtooling"},
                                {"advanced_facts_complete", true}};
    llvm::json::Object symbol{{"fact", "symbol"},
                              {"key", key},
                              {"qualified_name", name},
                              {"kind", "macro"},
                              {"span", std::move(*span)},
                              {"signature", name},
                              {"source_text", source_.source(range)},
                              {"metadata", std::move(metadata)}};
    sink_.add("symbol:" + key, std::move(symbol));
    sink_.add("occurrence:" + key,
              {{"fact", "occurrence"},
               {"symbol_key", key},
               {"kind", "definition"},
               {"span", std::move(*source_.span(range))}});
  }

  void MacroExpands(const clang::Token &macroName, const clang::MacroDefinition &definition,
                    clang::SourceRange range, const clang::MacroArgs *) override {
    const auto *info = definition.getMacroInfo();
    if (!info || !macroName.getIdentifierInfo())
      return;
    auto definitionPath = source_.path(info->getDefinitionLoc());
    auto usePath = source_.path(range.getBegin(), false);
    if (!definitionPath || !usePath || !source_.isProjectPath(*definitionPath) ||
        !source_.isProjectPath(*usePath))
      return;
    auto name = macroName.getIdentifierInfo()->getName().str();
    auto key = source_.macroKey(name, info->getDefinitionLoc());
    macroExpansions_.push_back(
        {name, key,
         clang::SourceRange(info->getDefinitionLoc(), info->getDefinitionEndLoc()), range});
    auto spelling = source_.span(range, true);
    auto expansion = source_.span(range, false);
    if (!spelling || !expansion)
      return;
    sink_.add("macro-expansion:" + key + ":" + usePath->string() + ":" +
                  std::to_string(source_.offset(range.getBegin(), false)),
              {{"fact", "occurrence"},
               {"symbol_key", key},
               {"kind", "macro_expansion"},
               {"span", std::move(*expansion)},
               {"spelling_span", std::move(*spelling)},
               {"expansion_span", std::move(*source_.span(range, false))}});
  }

private:
  FactSink &sink_;
  SourceFacts &source_;
  std::vector<MacroExpansionRecord> &macroExpansions_;
};

class Collector final : public clang::RecursiveASTVisitor<Collector> {
public:
  Collector(clang::ASTContext &context, FactSink &sink, std::filesystem::path projectRoot,
            const std::vector<MacroExpansionRecord> &macroExpansions)
      : context_(context), sink_(sink),
        source_(context.getSourceManager(), context.getLangOpts(), std::move(projectRoot)),
        macroExpansions_(macroExpansions) {}

  SourceFacts &sourceFacts() { return source_; }

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }

  bool VisitNamedDecl(clang::NamedDecl *decl) {
    auto kind = symbolKind(decl);
    if (!kind || !source_.relative(decl->getLocation()))
      return true;
    if (decl->isImplicit() && !isRequiredImplicit(decl))
      return true;
    if (const auto *owner = enclosingDecl(decl);
        owner && owner->isImplicit() && !isRequiredImplicit(owner))
      return true;
    emitSymbol(decl, *kind);
    emitTemplateRelationship(decl);
    return true;
  }

  bool VisitFunctionDecl(clang::FunctionDecl *function) {
    if (!function->isThisDeclarationADefinition() || !function->hasBody() ||
        !source_.relative(function->getLocation()) ||
        (function->isImplicit() && !isRequiredImplicit(function)))
      return true;
    emitCFG(function);
    emitTemplateRelationship(function);
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *expression) {
    auto *callee = expression->getDirectCallee();
    auto *owner = enclosingCallable(expression);
    if (!owner || !source_.relative(expression->getExprLoc(), false))
      return true;
    if (callee && source_.relative(callee->getLocation())) {
      emitSymbol(callee, llvm::isa<clang::CXXMethodDecl>(callee) ? "method" : "function");
      emitRelationship(owner, callee, "calls", expression->getSourceRange(), "call");
    }
    emitCallsite(expression, owner, callee);
    return true;
  }

  bool VisitCXXConstructExpr(clang::CXXConstructExpr *expression) {
    auto *constructor = expression->getConstructor();
    auto *owner = enclosingCallable(expression);
    if (constructor && owner && source_.relative(expression->getExprLoc()) &&
        source_.relative(constructor->getLocation())) {
      emitSymbol(constructor, "method");
      emitRelationship(owner, constructor, "calls", expression->getSourceRange(), "call");
    }
    if (constructor && owner && source_.relative(expression->getExprLoc(), false))
      emitResolvedCallsite(expression, owner, constructor, constructor, "constructor", true,
                           "", "certain", 1.0, "direct_constructor",
                           "Clang resolved the constructed type and constructor directly");
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *expression) {
    auto *target = llvm::dyn_cast<clang::NamedDecl>(expression->getDecl());
    auto *owner = enclosingCallable(expression);
    if (target && owner && source_.relative(target->getLocation()))
      emitRelationship(owner, target, "references", expression->getSourceRange(), "reference");
    return true;
  }

  bool VisitMemberExpr(clang::MemberExpr *expression) {
    auto *target = expression->getMemberDecl();
    auto *owner = enclosingCallable(expression);
    if (target && owner && source_.relative(target->getLocation()))
      emitRelationship(owner, target, "references", expression->getSourceRange(), "reference");
    return true;
  }

  bool VisitTypeLoc(clang::TypeLoc location) {
    const auto *tag = location.getType()->getAsTagDecl();
    if (!tag || !source_.relative(location.getBeginLoc()) || !source_.relative(tag->getLocation()))
      return true;
    auto *owner = enclosingNamed(location);
    if (owner)
      emitRelationship(owner, tag, "uses_type", location.getSourceRange(), "type");
    return true;
  }

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *record) {
    if (!record->isThisDeclarationADefinition() || !source_.relative(record->getLocation()))
      return true;
    emitSymbol(record, record->isStruct() ? "struct" : "class");
    for (const auto &base : record->bases()) {
      const auto *baseRecord = base.getType()->getAsCXXRecordDecl();
      if (baseRecord && source_.relative(baseRecord->getLocation()))
        emitRelationship(record, baseRecord, "inherits", base.getSourceRange(), "type");
    }
    return true;
  }

  bool VisitCXXMethodDecl(clang::CXXMethodDecl *method) {
    if (!source_.relative(method->getLocation()) ||
        (method->isImplicit() && !isRequiredImplicit(method)))
      return true;
    emitSymbol(method, "method");
    for (const auto *overridden : method->overridden_methods()) {
      if (source_.relative(overridden->getLocation()))
        emitRelationship(method, overridden, "overrides", method->getSourceRange(),
                         "reference", false);
    }
    return true;
  }

private:
  void emitCallsite(const clang::CallExpr *expression, const clang::FunctionDecl *owner,
                    const clang::FunctionDecl *staticTarget) {
    if (!staticTarget) {
      const auto *calleeExpression = expression->getCallee();
      const bool dependent = expression->isTypeDependent() || expression->isValueDependent() ||
                             calleeExpression->isTypeDependent() ||
                             calleeExpression->isValueDependent();
      emitResolvedCallsite(expression, owner, nullptr, nullptr,
                           dependent ? "dependent_template" : "unresolved_indirect", false,
                           dependent ? "dependent_or_uninstantiated_template"
                                     : "indirect_function_or_member_pointer_not_resolved",
                           "", 0.0, "", "");
      return;
    }

    const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(staticTarget);
    if (!method) {
      emitResolvedCallsite(expression, owner, staticTarget, staticTarget, "direct", true, "",
                           "certain", 1.0, "direct_ast",
                           "CallExpr::getDirectCallee selected one non-virtual function");
      return;
    }
    const auto *parent = method->getParent();
    if (parent && parent->isLambda() && method->getOverloadedOperator() == clang::OO_Call) {
      const bool generic = method->getPrimaryTemplate() != nullptr;
      emitResolvedCallsite(expression, owner, staticTarget, staticTarget,
                           generic ? "generic_lambda" : "lambda", true, "", "certain", 1.0,
                           "direct_ast",
                           generic ? "Clang selected a concrete generic-lambda specialization"
                                   : "Clang selected the concrete lambda call operator");
      return;
    }
    if (method->getOverloadedOperator() == clang::OO_Call) {
      emitResolvedCallsite(expression, owner, staticTarget, staticTarget, "functor", true, "",
                           "certain", 1.0, "direct_ast",
                           "Clang selected the concrete function-object call operator");
      return;
    }
    if (!method->isVirtual()) {
      emitResolvedCallsite(expression, owner, staticTarget, staticTarget, "direct", true, "",
                           "certain", 1.0, "direct_ast",
                           "CallExpr::getDirectCallee selected one non-virtual method");
      return;
    }

    const auto *memberCall = llvm::dyn_cast<clang::CXXMemberCallExpr>(expression);
    const auto *member = llvm::dyn_cast<clang::MemberExpr>(
        expression->getCallee()->IgnoreParenImpCasts());
    if (member && member->hasQualifier()) {
      emitResolvedCallsite(expression, owner, staticTarget, staticTarget, "direct", true, "",
                           "certain", 1.0, "qualified_direct_ast",
                           "a qualified member call suppresses virtual dispatch");
      return;
    }
    if (method->hasAttr<clang::FinalAttr>() || (parent && parent->isEffectivelyFinal())) {
      emitResolvedCallsite(expression, owner, staticTarget, staticTarget, "devirtualized", true,
                           "", "certain", 1.0, "final_dispatch",
                           "the selected method or its declaring class is final");
      return;
    }
    if (memberCall) {
      const auto *base = memberCall->getImplicitObjectArgument();
      if (base) {
        if (const auto *devirtualized = method->getDevirtualizedMethod(base, false)) {
          emitResolvedCallsite(expression, owner, staticTarget, devirtualized,
                               "devirtualized", true, "", "certain", 1.0,
                               "clang_devirtualized",
                               "CXXMethodDecl::getDevirtualizedMethod proved one target");
          return;
        }
      }
    }
    emitResolvedCallsite(expression, owner, staticTarget, staticTarget, "virtual", false,
                         "open_world_external_overrides_possible", "possible", 0.75,
                         "static_virtual_candidate",
                         "the statically selected method is a candidate; confidence is a "
                         "deterministic ranking value, not a probability");
  }

  template <typename Expression>
  void emitResolvedCallsite(const Expression *expression, const clang::FunctionDecl *owner,
                            const clang::FunctionDecl *staticTarget,
                            const clang::FunctionDecl *resolvedTarget,
                            llvm::StringRef dispatchKind, bool complete,
                            llvm::StringRef unresolvedReason, llvm::StringRef certainty,
                            double confidence, llvm::StringRef derivation,
                            llvm::StringRef confidenceReason) {
    auto spelling = source_.span(expression->getSourceRange(), true);
    auto expansion = source_.span(expression->getSourceRange(), false);
    auto ownerKind = symbolKind(owner);
    if (!spelling || !expansion || !ownerKind)
      return;
    emitSymbol(owner, *ownerKind);
    const bool staticTargetIndexed =
        staticTarget && source_.relative(staticTarget->getLocation()).has_value();
    const bool resolvedTargetIndexed =
        resolvedTarget && source_.relative(resolvedTarget->getLocation()).has_value();
    if (staticTargetIndexed)
      emitSymbol(staticTarget, llvm::isa<clang::CXXMethodDecl>(staticTarget) ? "method"
                                                                           : "function");
    if (resolvedTargetIndexed)
      emitSymbol(resolvedTarget, llvm::isa<clang::CXXMethodDecl>(resolvedTarget) ? "method"
                                                                                : "function");
    if ((staticTarget && !staticTargetIndexed) || (resolvedTarget && !resolvedTargetIndexed)) {
      complete = false;
      unresolvedReason = "resolved_target_external_or_unindexed";
      certainty = "";
      derivation = "";
      confidenceReason = "";
    }
    auto ownerKey = source_.declKey(owner, *ownerKind);
    auto range = expression->getSourceRange();
    auto callsiteKey = "callsite:" + ownerKey + ":" + expression->getStmtClassName() + ":" +
                       std::to_string(source_.offset(range.getBegin(), true)) + ":" +
                       std::to_string(source_.offset(range.getBegin(), false)) + ":" +
                       std::to_string(source_.offset(range.getEnd(), false));
    llvm::json::Object site{{"fact", "callsite_v1"},
                            {"key", callsiteKey},
                            {"owner_key", ownerKey},
                            {"dispatch_kind", dispatchKind.str()},
                            {"spelling_span", std::move(*spelling)},
                            {"expansion_span", std::move(*expansion)},
                            {"expansion_stack",
                             source_.expansionStack(range.getBegin(), macroExpansions_)},
                            {"target_set_complete", complete},
                            {"unresolved_reason", unresolvedReason.str()},
                            {"callee_text", source_.source(range)}};
    if (staticTargetIndexed)
      site["static_target_key"] = source_.declKey(
          staticTarget, llvm::isa<clang::CXXMethodDecl>(staticTarget) ? "method" : "function");
    sink_.add("callsite:" + callsiteKey, std::move(site));

    if (resolvedTargetIndexed && !certainty.empty()) {
      auto targetKey = source_.declKey(
          resolvedTarget, llvm::isa<clang::CXXMethodDecl>(resolvedTarget) ? "method" : "function");
      sink_.add("call-target:" + callsiteKey + ":" + targetKey,
                {{"fact", "call_target_v1"},
                 {"callsite_key", callsiteKey},
                 {"target_key", targetKey},
                 {"certainty", certainty.str()},
                 {"confidence", confidence},
                 {"confidence_reason", confidenceReason.str()},
                 {"derivation", derivation.str()},
                 {"evidence_span", std::move(*source_.span(range, false))}});
    }
    for (const auto &frame : macroExpansions_) {
      auto stack = source_.expansionStack(range.getBegin(), macroExpansions_);
      for (const auto &value : stack) {
        const auto *object = value.getAsObject();
        auto macroKey = object ? object->getString("macro_key") : std::nullopt;
        if (!macroKey || *macroKey != frame.key)
          continue;
        sink_.add("edge:generated_by_macro:" + ownerKey + ":" + frame.key + ":" +
                      std::to_string(source_.offset(range.getBegin(), false)),
                  {{"fact", "edge"},
                   {"source_key", ownerKey},
                   {"target_key", frame.key},
                   {"relation", "generated_by_macro"},
                   {"span", std::move(*source_.span(range, false))}});
        break;
      }
    }
  }

  static clang::CFG::BuildOptions cfgBuildOptions(const clang::LangOptions &language) {
    clang::CFG::BuildOptions options;
    options.PruneTriviallyFalseEdges = false;
    options.AddEHEdges = language.CXXExceptions;
    options.AddInitializers = true;
    options.AddImplicitDtors = true;
    options.AddLifetime = true;
    options.AddLoopExit = true;
    options.AddTemporaryDtors = true;
    options.AddScopes = true;
    options.AddStaticInitBranches = true;
    options.AddCXXNewAllocator = true;
    options.AddCXXDefaultInitExprInCtors = true;
    options.AddCXXDefaultInitExprInAggregates = true;
    options.AddRichCXXConstructors = true;
    options.MarkElidedCXXConstructors = true;
    options.AddVirtualBaseBranches = true;
    options.OmitImplicitValueInitializers = false;
    options.setAllAlwaysAdd();
    return options;
  }

  static llvm::StringRef cfgElementKind(clang::CFGElement::Kind kind) {
    switch (kind) {
    case clang::CFGElement::Initializer:
      return "initializer";
    case clang::CFGElement::ScopeBegin:
      return "scope_begin";
    case clang::CFGElement::ScopeEnd:
      return "scope_end";
    case clang::CFGElement::NewAllocator:
      return "new_allocator";
    case clang::CFGElement::LifetimeEnds:
      return "lifetime_end";
    case clang::CFGElement::LoopExit:
      return "loop_exit";
    case clang::CFGElement::Statement:
      return "statement";
    case clang::CFGElement::Constructor:
      return "constructor";
    case clang::CFGElement::CXXRecordTypedCall:
      return "record_typed_call";
    case clang::CFGElement::AutomaticObjectDtor:
      return "automatic_object_destructor";
    case clang::CFGElement::DeleteDtor:
      return "delete_destructor";
    case clang::CFGElement::BaseDtor:
      return "base_destructor";
    case clang::CFGElement::MemberDtor:
      return "member_destructor";
    case clang::CFGElement::TemporaryDtor:
      return "temporary_destructor";
    case clang::CFGElement::CleanupFunction:
      return "cleanup_function";
    }
    return "unknown";
  }

  const clang::Stmt *elementStatement(const clang::CFGElement &element) const {
    if (auto statement = element.getAs<clang::CFGStmt>())
      return statement->getStmt();
    if (auto allocator = element.getAs<clang::CFGNewAllocator>())
      return allocator->getAllocatorExpr();
    if (auto lifetime = element.getAs<clang::CFGLifetimeEnds>())
      return lifetime->getTriggerStmt();
    if (auto loop = element.getAs<clang::CFGLoopExit>())
      return loop->getLoopStmt();
    if (auto scope = element.getAs<clang::CFGScopeBegin>())
      return scope->getTriggerStmt();
    if (auto scope = element.getAs<clang::CFGScopeEnd>())
      return scope->getTriggerStmt();
    if (auto destructor = element.getAs<clang::CFGAutomaticObjDtor>())
      return destructor->getTriggerStmt();
    if (auto destructor = element.getAs<clang::CFGDeleteDtor>())
      return destructor->getDeleteExpr();
    if (auto destructor = element.getAs<clang::CFGTemporaryDtor>())
      return destructor->getBindTemporaryExpr();
    return nullptr;
  }

  std::optional<clang::SourceRange> elementRange(const clang::CFGElement &element) const {
    if (const auto *statement = elementStatement(element))
      return statement->getSourceRange();
    if (auto initializer = element.getAs<clang::CFGInitializer>())
      return initializer->getInitializer()->getSourceRange();
    if (auto destructor = element.getAs<clang::CFGBaseDtor>())
      return destructor->getBaseSpecifier()->getSourceRange();
    if (auto destructor = element.getAs<clang::CFGMemberDtor>())
      return destructor->getFieldDecl()->getSourceRange();
    if (auto cleanup = element.getAs<clang::CFGCleanupFunction>())
      return cleanup->getVarDecl()->getSourceRange();
    return std::nullopt;
  }

  llvm::json::Object cfgElementMetadata(const clang::CFGElement &element) const {
    llvm::json::Object metadata{{"implicit", element.getKind() < clang::CFGElement::STMT_BEGIN ||
                                                element.getKind() > clang::CFGElement::STMT_END}};
    if (auto initializer = element.getAs<clang::CFGInitializer>()) {
      const auto *value = initializer->getInitializer();
      metadata["initializer_kind"] = value->isBaseInitializer()      ? "base"
                                     : value->isMemberInitializer() ? "member"
                                     : value->isDelegatingInitializer() ? "delegating"
                                                                        : "other";
      if (value->isMemberInitializer())
        metadata["declaration"] = value->getMember()->getQualifiedNameAsString();
    } else if (auto lifetime = element.getAs<clang::CFGLifetimeEnds>()) {
      metadata["declaration"] = lifetime->getVarDecl()->getQualifiedNameAsString();
    } else if (auto destructor = element.getAs<clang::CFGAutomaticObjDtor>()) {
      metadata["declaration"] = destructor->getVarDecl()->getQualifiedNameAsString();
    } else if (auto destructor = element.getAs<clang::CFGMemberDtor>()) {
      metadata["declaration"] = destructor->getFieldDecl()->getQualifiedNameAsString();
    } else if (auto cleanup = element.getAs<clang::CFGCleanupFunction>()) {
      metadata["declaration"] = cleanup->getVarDecl()->getQualifiedNameAsString();
      metadata["cleanup_function"] =
          cleanup->getFunctionDecl()->getQualifiedNameAsString();
    }
    return metadata;
  }

  void addRange(llvm::json::Object &fact, clang::SourceRange range,
                llvm::StringRef spellingName, llvm::StringRef expansionName) const {
    if (auto spelling = source_.span(range, true))
      fact[spellingName] = std::move(*spelling);
    if (auto expansion = source_.span(range, false))
      fact[expansionName] = std::move(*expansion);
  }

  static bool blockContains(const clang::CFGBlock &block, clang::Stmt::StmtClass kind) {
    for (const auto &element : block) {
      if (auto statement = element.getAs<clang::CFGStmt>();
          statement && statement->getStmt()->getStmtClass() == kind)
        return true;
    }
    return false;
  }

  static llvm::StringRef edgeKind(const clang::CFGBlock &source,
                                  const clang::CFGBlock &target,
                                  unsigned successorIndex,
                                  const llvm::SmallPtrSetImpl<const clang::CFGBlock *> &tryBlocks) {
    const auto *terminator = source.getTerminatorStmt();
    const auto *label = target.getLabel();
    if (llvm::isa_and_nonnull<clang::CXXCatchStmt>(label) || tryBlocks.contains(&source) ||
        tryBlocks.contains(&target) ||
        blockContains(source, clang::Stmt::CXXThrowExprClass))
      return "exception";
    if (llvm::isa_and_nonnull<clang::DefaultStmt>(label))
      return "default";
    if (llvm::isa_and_nonnull<clang::CaseStmt>(label))
      return "case";
    if (llvm::isa_and_nonnull<clang::BreakStmt>(terminator))
      return "break";
    if (llvm::isa_and_nonnull<clang::ContinueStmt>(terminator))
      return "continue";
    if (llvm::isa_and_nonnull<clang::GotoStmt, clang::IndirectGotoStmt>(terminator))
      return "goto";
    if (blockContains(source, clang::Stmt::ReturnStmtClass))
      return "return";
    if (source.getLoopTarget())
      return "loop_back";
    if (llvm::isa_and_nonnull<clang::IfStmt, clang::WhileStmt, clang::ForStmt,
                             clang::DoStmt, clang::ConditionalOperator>(terminator))
      return successorIndex == 0 ? "true" : "false";
    if (const auto *binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(terminator);
        binary && binary->isLogicalOp())
      return successorIndex == 0 ? "true" : "false";
    return "fallthrough";
  }

  void emitCFG(const clang::FunctionDecl *function) {
    auto body = function->getBody();
    auto functionKind = llvm::isa<clang::CXXMethodDecl>(function) ? "method" : "function";
    auto functionKey = source_.declKey(function, functionKind);
    auto options = cfgBuildOptions(context_.getLangOpts());
    auto cfg = clang::CFG::buildCFG(function, body, &context_, options);
    if (!cfg)
      return;

    emitSymbol(function, functionKind);
    auto graphKey = "cfg:" + functionKey;
    auto blockKey = [&](const clang::CFGBlock &block) {
      return graphKey + ":block:" + std::to_string(block.getBlockID());
    };

    llvm::SmallPtrSet<const clang::CFGBlock *, 16> tryBlocks;
    for (const auto *block : cfg->try_blocks())
      tryBlocks.insert(block);

    llvm::SmallPtrSet<const clang::CFGBlock *, 32> reachable;
    std::deque<const clang::CFGBlock *> pending{&cfg->getEntry()};
    reachable.insert(&cfg->getEntry());
    while (!pending.empty()) {
      const auto *block = pending.front();
      pending.pop_front();
      for (const auto &successor : block->succs()) {
        const auto *target = successor.getReachableBlock();
        if (target && reachable.insert(target).second)
          pending.push_back(target);
      }
    }

    llvm::json::Object buildOptions{
        {"prune_trivially_false_edges", false},
        {"add_eh_edges", options.AddEHEdges},
        {"add_initializers", true},
        {"add_implicit_dtors", true},
        {"add_lifetime", true},
        {"add_loop_exit", true},
        {"add_temporary_dtors", true},
        {"add_scopes", true},
        {"add_static_init_branches", true},
        {"add_cxx_new_allocator", true},
        {"add_cxx_default_init_expr_in_ctors", true},
        {"add_cxx_default_init_expr_in_aggregates", true},
        {"add_rich_cxx_constructors", true},
        {"mark_elided_cxx_constructors", true},
        {"add_virtual_base_branches", true},
        {"omit_implicit_value_initializers", false},
        {"always_add_all_statements", true}};
    sink_.add("cfg-graph:" + functionKey,
              {{"fact", "cfg_graph_v1"},
               {"key", graphKey},
               {"function_key", functionKey},
               {"entry_block_key", blockKey(cfg->getEntry())},
               {"normal_exit_block_key", blockKey(cfg->getExit())},
               {"exceptional_exit_block_key", nullptr},
               {"clang_major", kClangMajor},
               {"fact_schema_version", 1},
               {"build_options", std::move(buildOptions)}});

    std::vector<const clang::CFGBlock *> blocks(cfg->begin(), cfg->end());
    std::sort(blocks.begin(), blocks.end(), [](const auto *left, const auto *right) {
      return left->getBlockID() < right->getBlockID();
    });
    for (const auto *block : blocks) {
      std::string role = "normal";
      if (block == &cfg->getEntry())
        role = "entry";
      else if (block == &cfg->getExit())
        role = "normal_exit";
      llvm::json::Object blockFact{{"fact", "cfg_block_v1"},
                                   {"key", blockKey(*block)},
                                   {"graph_key", graphKey},
                                   {"index", static_cast<std::int64_t>(block->getBlockID())},
                                   {"role", role},
                                   {"reachable", reachable.contains(block)}};
      if (const auto *terminator = block->getTerminatorStmt()) {
        blockFact["terminator_kind"] = terminator->getStmtClassName();
        blockFact["terminator_text"] = source_.source(terminator->getSourceRange());
        addRange(blockFact, terminator->getSourceRange(), "terminator_spelling_span",
                 "terminator_expansion_span");
      }
      if (const auto *label = block->getLabel()) {
        blockFact["label_kind"] = label->getStmtClassName();
        blockFact["label_text"] = source_.source(label->getSourceRange());
      }
      sink_.add("cfg-block:" + functionKey + ":" +
                    std::to_string(block->getBlockID()),
                std::move(blockFact));

      unsigned elementIndex = 0;
      for (const auto &element : *block) {
        llvm::json::Object elementFact{
            {"fact", "cfg_element_v1"},
            {"key", blockKey(*block) + ":element:" + std::to_string(elementIndex)},
            {"graph_key", graphKey},
            {"block_key", blockKey(*block)},
            {"index", static_cast<std::int64_t>(elementIndex)},
            {"kind", cfgElementKind(element.getKind())},
            {"metadata", cfgElementMetadata(element)}};
        if (const auto *statement = elementStatement(element)) {
          elementFact["statement_class"] = statement->getStmtClassName();
          elementFact["text"] = source_.source(statement->getSourceRange());
        }
        if (auto range = elementRange(element))
          addRange(elementFact, *range, "spelling_span", "expansion_span");
        sink_.add("cfg-element:" + functionKey + ":" +
                      std::to_string(block->getBlockID()) + ":" +
                      std::to_string(elementIndex),
                  std::move(elementFact));
        ++elementIndex;
      }

      unsigned successorIndex = 0;
      for (const auto &successor : block->succs()) {
        const auto emitEdge = [&](const clang::CFGBlock *target, bool feasible,
                                  llvm::StringRef suffix) {
          if (!target)
            return;
          const auto kind = edgeKind(*block, *target, successorIndex, tryBlocks);
          sink_.add("cfg-edge:" + functionKey + ":" +
                        std::to_string(block->getBlockID()) + ":" +
                        std::to_string(successorIndex) + ":" + suffix.str() + ":" +
                        std::to_string(target->getBlockID()),
                    {{"fact", "cfg_edge_v1"},
                     {"graph_key", graphKey},
                     {"source_block_key", blockKey(*block)},
                     {"target_block_key", blockKey(*target)},
                     {"kind", kind},
                     {"successor_index", static_cast<std::int64_t>(successorIndex)},
                     {"feasible", feasible}});
        };
        const auto *reachableTarget = successor.getReachableBlock();
        emitEdge(reachableTarget, true, "reachable");
        const auto *alternateTarget = successor.getPossiblyUnreachableBlock();
        if (alternateTarget != reachableTarget)
          emitEdge(alternateTarget, false, "unreachable");
        ++successorIndex;
      }
    }
  }

  static bool isRequiredImplicit(const clang::NamedDecl *decl) {
    const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(decl);
    return method && method->getParent() && method->getParent()->isLambda() &&
           method->getOverloadedOperator() == clang::OO_Call;
  }

  static std::optional<std::string> symbolKind(const clang::NamedDecl *decl) {
    if (llvm::isa<clang::CXXMethodDecl>(decl))
      return "method";
    if (llvm::isa<clang::FunctionDecl, clang::FunctionTemplateDecl>(decl))
      return "function";
    if (const auto *record = llvm::dyn_cast<clang::CXXRecordDecl>(decl))
      return record->isStruct() ? "struct" : "class";
    if (llvm::isa<clang::ClassTemplateDecl, clang::ClassTemplatePartialSpecializationDecl>(decl))
      return "class";
    if (llvm::isa<clang::EnumDecl>(decl))
      return "enum";
    if (llvm::isa<clang::NamespaceDecl>(decl))
      return "namespace";
    if (llvm::isa<clang::VarDecl, clang::FieldDecl, clang::EnumConstantDecl,
                  clang::ParmVarDecl>(decl))
      return "variable";
    if (llvm::isa<clang::TypedefNameDecl>(decl))
      return "type_alias";
    return std::nullopt;
  }

  bool isDefinition(const clang::NamedDecl *decl) const {
    if (const auto *function = llvm::dyn_cast<clang::FunctionDecl>(decl))
      return function->isThisDeclarationADefinition();
    if (const auto *tag = llvm::dyn_cast<clang::TagDecl>(decl))
      return tag->isThisDeclarationADefinition();
    if (const auto *variable = llvm::dyn_cast<clang::VarDecl>(decl))
      return variable->isThisDeclarationADefinition() != clang::VarDecl::DeclarationOnly;
    if (llvm::isa<clang::NamespaceDecl, clang::EnumConstantDecl, clang::FieldDecl,
                  clang::ParmVarDecl, clang::TypedefNameDecl>(decl))
      return true;
    return false;
  }

  std::string qualifiedName(const clang::NamedDecl *decl, llvm::StringRef kind) const {
    auto result = decl->getQualifiedNameAsString();
    if (!result.empty())
      return result;
    auto location = context_.getSourceManager().getSpellingLoc(decl->getLocation());
    return "<anonymous-" + kind.str() + "@" +
           std::to_string(context_.getSourceManager().getSpellingLineNumber(location)) + ":" +
           std::to_string(context_.getSourceManager().getSpellingColumnNumber(location)) + ">";
  }

  llvm::json::Object templateMetadata(const clang::NamedDecl *decl) const {
    llvm::json::Object metadata;
    const clang::FunctionDecl *function = llvm::dyn_cast<clang::FunctionDecl>(decl);
    const clang::ClassTemplateSpecializationDecl *record =
        llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(decl);
    if (!function && !record && !llvm::isa<clang::TemplateDecl>(decl))
      return metadata;
    std::string kind = "primary";
    llvm::json::Array arguments;
    if (function) {
      switch (function->getTemplateSpecializationKind()) {
      case clang::TSK_Undeclared:
        break;
      case clang::TSK_ImplicitInstantiation:
        kind = "implicit_instantiation";
        break;
      case clang::TSK_ExplicitSpecialization:
        kind = "explicit_specialization";
        break;
      case clang::TSK_ExplicitInstantiationDeclaration:
        kind = "explicit_instantiation_declaration";
        break;
      case clang::TSK_ExplicitInstantiationDefinition:
        kind = "explicit_instantiation_definition";
        break;
      }
      if (const auto *list = function->getTemplateSpecializationArgs()) {
        for (const auto &argument : list->asArray()) {
          std::string rendered;
          llvm::raw_string_ostream stream(rendered);
          argument.print(context_.getPrintingPolicy(), stream, true);
          arguments.push_back(stream.str());
        }
      }
      if (auto location = function->getPointOfInstantiation(); location.isValid()) {
        if (auto point = source_.span(clang::SourceRange(location, location), false))
          metadata["point_of_instantiation"] = std::move(*point);
      }
    }
    if (record) {
      switch (record->getSpecializationKind()) {
      case clang::TSK_Undeclared:
        break;
      case clang::TSK_ImplicitInstantiation:
        kind = "implicit_instantiation";
        break;
      case clang::TSK_ExplicitSpecialization:
        kind = "explicit_specialization";
        break;
      case clang::TSK_ExplicitInstantiationDeclaration:
        kind = "explicit_instantiation_declaration";
        break;
      case clang::TSK_ExplicitInstantiationDefinition:
        kind = "explicit_instantiation_definition";
        break;
      }
      for (const auto &argument : record->getTemplateArgs().asArray()) {
        std::string rendered;
        llvm::raw_string_ostream stream(rendered);
        argument.print(context_.getPrintingPolicy(), stream, true);
        arguments.push_back(stream.str());
      }
      if (auto location = record->getPointOfInstantiation(); location.isValid()) {
        if (auto point = source_.span(clang::SourceRange(location, location), false))
          metadata["point_of_instantiation"] = std::move(*point);
      }
    }
    metadata["template_kind"] = kind;
    metadata["template_arguments"] = std::move(arguments);
    return metadata;
  }

  void emitTemplateRelationship(const clang::NamedDecl *decl) {
    if (const auto *function = llvm::dyn_cast<clang::FunctionDecl>(decl)) {
      auto specializationKind = function->getTemplateSpecializationKind();
      if (specializationKind == clang::TSK_Undeclared)
        return;
      const clang::FunctionDecl *pattern = function->getTemplateInstantiationPattern();
      if (!pattern) {
        if (const auto *primary = function->getPrimaryTemplate())
          pattern = primary->getTemplatedDecl();
      }
      if (!pattern || pattern == function || !source_.relative(pattern->getLocation()))
        return;
      emitRelationship(function, pattern,
                       specializationKind == clang::TSK_ExplicitSpecialization
                           ? "specializes"
                           : "instantiates",
                       function->getSourceRange(), "reference", false);
      return;
    }
    if (const auto *record =
            llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(decl)) {
      const auto *primary = record->getSpecializedTemplate();
      const auto *pattern = primary ? primary->getTemplatedDecl() : nullptr;
      if (!pattern || pattern == record || !source_.relative(pattern->getLocation()))
        return;
      emitRelationship(record, pattern,
                       record->getSpecializationKind() == clang::TSK_ExplicitSpecialization
                           ? "specializes"
                           : "instantiates",
                       record->getSourceRange(), "reference", false);
    }
  }

  void emitFile(const std::filesystem::path &path) {
    auto key = source_.fileKey(path);
    sink_.add("file:" + path.string(),
              {{"fact", "file"}, {"key", key}, {"path", path.string()}});
  }

  void emitSymbol(const clang::NamedDecl *decl, llvm::StringRef kind) {
    auto span = source_.span(decl->getSourceRange());
    auto path = source_.path(decl->getLocation());
    if (!span || !path || !source_.isProjectPath(*path))
      return;
    emitFile(*path);
    auto key = source_.declKey(decl, kind);
    const auto [startOffset, endOffset] = source_.offsets(decl->getSourceRange());
    llvm::json::Object metadata{{"is_definition", isDefinition(decl)},
                                {"analysis_backend", "clang-libtooling"},
                                {"advanced_facts_complete", true},
                                {"start_offset", startOffset},
                                {"end_offset_exclusive", endOffset}};
    auto templateInfo = templateMetadata(decl);
    for (auto &entry : templateInfo)
      metadata[entry.first] = std::move(entry.second);
    if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(decl)) {
      if (method->getParent() && method->getParent()->isLambda() &&
          method->getOverloadedOperator() == clang::OO_Call) {
        metadata["is_lambda_call_operator"] = true;
        metadata["stable_lambda_key"] = key;
      }
    }
    std::string signature;
    llvm::raw_string_ostream signatureStream(signature);
    decl->print(signatureStream, context_.getPrintingPolicy());
    signatureStream.flush();
    std::string documentation;
    if (const auto *comment = context_.getRawCommentForDeclNoCache(decl))
      documentation = comment->getRawText(context_.getSourceManager()).str();
    llvm::json::Object fact{{"fact", "symbol"},
                            {"key", key},
                            {"qualified_name", qualifiedName(decl, kind)},
                            {"kind", kind.str()},
                            {"span", std::move(*span)},
                            {"signature", signature},
                            {"documentation", documentation},
                            {"source_text", source_.source(decl->getSourceRange())},
                            {"metadata", std::move(metadata)}};
    sink_.add("symbol:" + key + (isDefinition(decl) ? ":0" : ":1"), std::move(fact));
    llvm::StringRef occurrenceKind = isDefinition(decl) ? "definition" : "declaration";
    auto occurrencePath = source_.path(decl->getLocation(), false).value_or(*path);
    sink_.add("occurrence:" + key + ":" + occurrenceKind.str() + ":" +
                  occurrencePath.string() + ":" +
                  std::to_string(source_.offset(decl->getLocation(), false)),
              {{"fact", "occurrence"},
               {"symbol_key", key},
               {"kind", occurrenceKind.str()},
               {"span", std::move(*source_.span(decl->getSourceRange()))},
               {"enclosing_key", enclosingDeclKey(decl)}});
    auto owner = enclosingDecl(decl);
    auto sourceKey = owner ? source_.declKey(owner, symbolKind(owner).value_or("unknown"))
                           : source_.fileKey(*path);
    if (sourceKey != key)
      sink_.add("edge:contains:" + sourceKey + ":" + key,
                {{"fact", "edge"},
                 {"source_key", sourceKey},
                 {"target_key", key},
                 {"relation", "contains"}});
  }

  const clang::NamedDecl *enclosingDecl(const clang::Decl *decl) const {
    const clang::DeclContext *context = decl ? decl->getDeclContext() : nullptr;
    while (context) {
      if (auto *named = llvm::dyn_cast<clang::NamedDecl>(context)) {
        if (symbolKind(named))
          return named;
      }
      context = context->getParent();
    }
    return nullptr;
  }

  std::string enclosingDeclKey(const clang::Decl *decl) const {
    auto *owner = enclosingDecl(decl);
    return owner ? source_.declKey(owner, symbolKind(owner).value_or("unknown")) : "";
  }

  template <typename Node> const clang::FunctionDecl *enclosingCallable(const Node *node) const {
    if (!node)
      return nullptr;
    clang::DynTypedNode current = clang::DynTypedNode::create(*node);
    for (unsigned depth = 0; depth < 64; ++depth) {
      auto parents = context_.getParents(current);
      if (parents.empty())
        return nullptr;
      if (const auto *function = parents[0].template get<clang::FunctionDecl>())
        return function;
      current = parents[0];
    }
    return nullptr;
  }

  template <typename Node> const clang::NamedDecl *enclosingNamed(const Node &node) const {
    clang::DynTypedNode current = clang::DynTypedNode::create(node);
    for (unsigned depth = 0; depth < 64; ++depth) {
      auto parents = context_.getParents(current);
      if (parents.empty())
        return nullptr;
      if (const auto *named = parents[0].template get<clang::NamedDecl>())
        return named;
      current = parents[0];
    }
    return nullptr;
  }

  void emitRelationship(const clang::NamedDecl *source, const clang::NamedDecl *target,
                        llvm::StringRef relation, clang::SourceRange range,
                        llvm::StringRef occurrenceKind, bool addReference = true) {
    auto sourceKind = symbolKind(source);
    auto targetKind = symbolKind(target);
    auto span = source_.span(range, false);
    if (!sourceKind || !targetKind || !span)
      return;
    emitSymbol(source, *sourceKind);
    emitSymbol(target, *targetKind);
    auto sourceKey = source_.declKey(source, *sourceKind);
    auto targetKey = source_.declKey(target, *targetKind);
    std::string evidence = std::to_string(source_.offset(range.getBegin(), false));
    sink_.add("edge:" + relation.str() + ":" + sourceKey + ":" + targetKey + ":" + evidence,
              {{"fact", "edge"},
               {"source_key", sourceKey},
               {"target_key", targetKey},
               {"relation", relation.str()},
               {"span", std::move(*span)}});
    if (addReference && relation != "references")
      sink_.add("edge:references:" + sourceKey + ":" + targetKey + ":" + evidence,
                {{"fact", "edge"},
                 {"source_key", sourceKey},
                 {"target_key", targetKey},
                 {"relation", "references"},
                 {"span", std::move(*source_.span(range, false))}});
    sink_.add("occurrence:" + occurrenceKind.str() + ":" + targetKey + ":" + evidence,
              {{"fact", "occurrence"},
               {"symbol_key", targetKey},
               {"enclosing_key", sourceKey},
               {"kind", occurrenceKind.str()},
               {"span", std::move(*source_.span(range, false))}});
  }

  clang::ASTContext &context_;
  FactSink &sink_;
  SourceFacts source_;
  const std::vector<MacroExpansionRecord> &macroExpansions_;
};

class Consumer final : public clang::ASTConsumer {
public:
  Consumer(clang::ASTContext &context, FactSink &sink, const std::filesystem::path &root,
           const std::vector<MacroExpansionRecord> &macroExpansions)
      : collector_(context, sink, root, macroExpansions) {}
  void HandleTranslationUnit(clang::ASTContext &context) override {
    collector_.TraverseDecl(context.getTranslationUnitDecl());
  }

private:
  Collector collector_;
};

class Action final : public clang::ASTFrontendAction {
public:
  Action(FactSink &sink, std::filesystem::path root) : sink_(sink), root_(std::move(root)) {}

  bool BeginSourceFileAction(clang::CompilerInstance &compiler) override {
    source_ = std::make_unique<SourceFacts>(compiler.getSourceManager(), compiler.getLangOpts(),
                                            root_);
    compiler.getPreprocessor().addPPCallbacks(
        std::make_unique<PreprocessorCollector>(sink_, *source_, macroExpansions_));
    return true;
  }

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &compiler, llvm::StringRef) override {
    return std::make_unique<Consumer>(compiler.getASTContext(), sink_, root_,
                                      macroExpansions_);
  }

private:
  FactSink &sink_;
  std::filesystem::path root_;
  std::unique_ptr<SourceFacts> source_;
  std::vector<MacroExpansionRecord> macroExpansions_;
};

class ActionFactory final : public clang::tooling::FrontendActionFactory {
public:
  ActionFactory(FactSink &sink, std::filesystem::path root)
      : sink_(sink), root_(std::move(root)) {}
  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<Action>(sink_, root_);
  }

private:
  FactSink &sink_;
  std::filesystem::path root_;
};

bool handleHello(const llvm::json::Object &request) {
  auto protocol = request.getString("protocol");
  auto version = request.getInteger("protocol_version");
  auto requiredMajor = request.getInteger("required_clang_major");
  if (!protocol || *protocol != kProtocol || !version || *version != kProtocolVersion) {
    emitError("protocol_mismatch", "unsupported analyzer protocol");
    return false;
  }
  if (!requiredMajor || *requiredMajor != kClangMajor) {
    emitError("clang_major_mismatch", "the analyzer requires Clang major 18");
    return false;
  }
  llvm::json::Array capabilities;
  for (const auto &capability : kCapabilities)
    capabilities.push_back(capability);
  emit({{"type", "hello"},
        {"protocol", kProtocol},
        {"protocol_version", kProtocolVersion},
        {"analyzer_version", CPP_CONTEXT_ANALYZER_VERSION},
        {"clang_major", kClangMajor},
        {"capabilities", std::move(capabilities)}});
  return true;
}

bool handleAnalyze(const llvm::json::Object &request) {
  auto requestId = requiredString(request, "request_id");
  auto root = requiredString(request, "project_root");
  auto source = requiredString(request, "source_path");
  auto directory = requiredString(request, "directory");
  const auto *argumentsValue = request.getArray("arguments");
  if (!requestId || !root || !source || !directory || !argumentsValue) {
    emitError("invalid_request", "analyze requires bounded project and compiler inputs");
    return false;
  }
  std::vector<std::string> arguments;
  for (const auto &entry : *argumentsValue) {
    auto argument = entry.getAsString();
    if (!argument) {
      emitError("invalid_request", "compiler arguments must be strings");
      return false;
    }
    arguments.push_back(argument->str());
  }
  emit({{"type", "begin"}, {"request_id", *requestId}});
  FactSink sink;
  clang::tooling::FixedCompilationDatabase database(*directory, arguments);
  std::vector<std::string> sources{*source};
  clang::tooling::ClangTool tool(database, sources);
  ActionFactory factory(sink, *root);
  const int result = tool.run(&factory);
  if (result != 0) {
    emitError("analysis_failed", "Clang rejected the translation unit");
    return false;
  }
  sink.flush();
  emit({{"type", "complete"}, {"request_id", *requestId}, {"success", true}});
  return true;
}

} // namespace

int main() {
  std::string line;
  bool ready = false;
  while (std::getline(std::cin, line)) {
    auto parsed = llvm::json::parse(line);
    if (!parsed) {
      emitError("invalid_json", "input must be one JSON object per line");
      return 2;
    }
    auto *request = parsed->getAsObject();
    if (!request) {
      emitError("invalid_request", "input record must be an object");
      return 2;
    }
    auto type = request->getString("type");
    if (!ready) {
      if (!type || *type != "hello" || !handleHello(*request))
        return 2;
      ready = true;
      continue;
    }
    if (!type || *type != "analyze" || !handleAnalyze(*request))
      return 2;
  }
  return ready ? 0 : 2;
}
