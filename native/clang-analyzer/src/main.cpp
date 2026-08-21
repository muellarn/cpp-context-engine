#include <algorithm>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
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
constexpr std::int64_t kProtocolVersion = 5;
constexpr std::int64_t kClangMajor = 18;

const std::vector<std::string> kCapabilities = {
    "direct_calls",       "full_ast",          "function_cfg_v1",
    "includes",
    "inherits",           "lambda_metadata",   "macro_provenance",
    "occurrences",        "overrides",         "pp_callbacks",
    "source_manager",     "symbols",           "template_metadata",
    "uses_type",          "callsites_v1",     "dispatch_targets_v1",
    "macro_expansion_stack", "template_relationships_v1",
    "intraprocedural_dataflow_v1", "points_to_v1", "function_summaries_v1",
    "interprocedural_bindings_v1"};

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

constexpr unsigned kDataFlowMaxIterations = 64;
constexpr unsigned kDataFlowMaxAliasTargets = 64;
constexpr unsigned kDataFlowMaxAccessPathDepth = 8;
constexpr unsigned kDataFlowMaxLocations = 4096;

struct PointsToValue {
  std::set<const clang::FunctionDecl *> functions;
  std::set<std::string> locations;
  bool complete = true;
  bool includesNull = false;

  bool operator==(const PointsToValue &other) const {
    return functions == other.functions && locations == other.locations &&
           complete == other.complete && includesNull == other.includesNull;
  }
};

struct DataFlowState {
  std::map<std::string, std::set<std::string>> definitions;
  std::map<std::string, bool> definitionsComplete;
  std::map<std::string, PointsToValue> pointsTo;

  bool operator==(const DataFlowState &other) const {
    return definitions == other.definitions &&
           definitionsComplete == other.definitionsComplete && pointsTo == other.pointsTo;
  }
};

struct MemoryLocationRecord {
  std::string key;
  std::string kind;
  std::string name;
  std::string typeName;
  std::string declarationKey;
  std::string baseKey;
  std::vector<std::string> accessPath;
  bool isVolatile = false;
  bool isAtomic = false;
  bool tracksPointsTo = false;
};

struct DataAccessRecord {
  std::string key;
  std::string blockKey;
  std::string elementKey;
  std::string locationKey;
  std::string kind;
  unsigned sequence = 0;
  const clang::Stmt *statement = nullptr;
  const clang::Expr *assignedExpression = nullptr;
  std::string expression;
  std::vector<const clang::FunctionDecl *> pointees;
  bool pointsToComplete = true;
};

struct IndirectCallRecord {
  const clang::CallExpr *expression = nullptr;
  const clang::FunctionDecl *owner = nullptr;
  std::string blockKey;
  const clang::Expr *calleeExpression = nullptr;
  unsigned sequence = 0;
};

struct SummaryCallRecord {
  std::string callsiteKey;
  std::vector<std::string> argumentLocations;
  std::vector<bool> writebackCandidates;
  std::string resultLocation;
  std::string resultAccess;
};

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
    emitDataFlow(function, *cfg, graphKey);
  }

  void emitDataFlow(const clang::FunctionDecl *function, const clang::CFG &cfg,
                    const std::string &graphKey) {
    const std::string analysisKey = "data-flow:" + graphKey;
    const auto blockKey = [&](const clang::CFGBlock &block) {
      return graphKey + ":block:" + std::to_string(block.getBlockID());
    };
    std::set<std::string> incompleteReasons;
    std::map<std::string, MemoryLocationRecord> locations;
    std::map<std::string, std::vector<DataAccessRecord>> accessesByBlock;
    std::map<std::string, unsigned> sequences;
    std::vector<IndirectCallRecord> indirectCalls;
    std::vector<SummaryCallRecord> summaryCalls;
    std::set<const clang::Expr *> handledExpressions;
    std::vector<std::string> parameterLocations;
    std::vector<std::string> parameterModes;

    const std::string unknownKey = analysisKey + ":memory:unknown";
    locations.emplace(unknownKey,
                      MemoryLocationRecord{unknownKey, "unknown", "$unknown", "", "", "",
                                           {}, false, false, false});

    const auto typeName = [&](clang::QualType type) {
      return type.getAsString(context_.getPrintingPolicy());
    };
    const auto tracksPointsTo = [](clang::QualType type) {
      if (type.isNull())
        return false;
      if (type->isReferenceType() || type->isMemberFunctionPointerType())
        return true;
      if (type->getAs<clang::PointerType>())
        return true;
      return false;
    };
    const auto parameterMode = [](clang::QualType type) {
      if (type->isRValueReferenceType())
        return std::string("rvalue_reference");
      if (type->isReferenceType())
        return type.getNonReferenceType().isConstQualified()
                   ? std::string("const_reference")
                   : std::string("reference");
      if (type->isPointerType())
        return type->getPointeeType().isConstQualified() ? std::string("const_pointer")
                                                         : std::string("pointer");
      return std::string("value");
    };
    const auto qualifyLocation = [&](MemoryLocationRecord &record, clang::QualType type) {
      if (type.isNull())
        return;
      record.typeName = typeName(type);
      record.isVolatile = type.isVolatileQualified();
      record.isAtomic = type->isAtomicType() || record.typeName.find("std::atomic") != std::string::npos;
      record.tracksPointsTo = tracksPointsTo(type);
      if (record.isVolatile)
        incompleteReasons.insert("volatile_access");
      if (record.isAtomic)
        incompleteReasons.insert("atomic_access");
    };

    std::function<std::string(const clang::ValueDecl *)> locationForDecl;
    std::function<std::string(const clang::Expr *)> locationForLValue;
    locationForDecl = [&](const clang::ValueDecl *decl) -> std::string {
      if (!decl)
        return unknownKey;
      auto kind = symbolKind(decl).value_or("variable");
      const bool indexed = source_.relative(decl->getLocation()).has_value();
      const std::string declarationKey = indexed ? source_.declKey(decl, kind) : "";
      const std::string key = analysisKey + ":memory:decl:" +
                              (declarationKey.empty()
                                   ? decl->getQualifiedNameAsString() + ":" +
                                         std::to_string(source_.offset(decl->getLocation()))
                                   : declarationKey);
      if (locations.count(key))
        return key;
      std::string locationKind = "local";
      if (llvm::isa<clang::ParmVarDecl>(decl))
        locationKind = "parameter";
      else if (const auto *variable = llvm::dyn_cast<clang::VarDecl>(decl);
               variable && variable->hasGlobalStorage())
        locationKind = "global";
      MemoryLocationRecord record{key,
                                  locationKind,
                                  decl->getQualifiedNameAsString().empty()
                                      ? decl->getNameAsString()
                                      : decl->getQualifiedNameAsString(),
                                  "",
                                  declarationKey,
                                  "",
                                  {},
                                  false,
                                  false,
                                  false};
      if (record.name.empty())
        record.name = "<anonymous-storage>";
      qualifyLocation(record, decl->getType());
      if (indexed) {
        emitSymbol(decl, kind);
      } else if (locationKind == "global") {
        incompleteReasons.insert("external_global_storage");
      }
      locations.emplace(key, std::move(record));
      if (locations.size() > kDataFlowMaxLocations) {
        locations.erase(key);
        incompleteReasons.insert("location_cap_exceeded");
        return unknownKey;
      }
      return key;
    };

    const auto addDerivedLocation = [&](std::string key, std::string kind, std::string name,
                                        clang::QualType type, std::string base,
                                        std::vector<std::string> path) -> std::string {
      if (locations.count(key))
        return key;
      if (path.size() > kDataFlowMaxAccessPathDepth) {
        incompleteReasons.insert("access_path_cap_exceeded");
        return unknownKey;
      }
      MemoryLocationRecord record{std::move(key), std::move(kind), std::move(name), "", "",
                                  std::move(base), std::move(path), false, false, false};
      qualifyLocation(record, type);
      const auto result = record.key;
      locations.emplace(result, std::move(record));
      if (locations.size() > kDataFlowMaxLocations) {
        locations.erase(result);
        incompleteReasons.insert("location_cap_exceeded");
        return unknownKey;
      }
      return result;
    };

    locationForLValue = [&](const clang::Expr *raw) -> std::string {
      if (!raw)
        return unknownKey;
      const clang::Expr *expression = raw->IgnoreParenImpCasts();
      handledExpressions.insert(expression);
      if (const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression)) {
        if (const auto *value = llvm::dyn_cast<clang::ValueDecl>(reference->getDecl()))
          return locationForDecl(value);
        return unknownKey;
      }
      if (llvm::isa<clang::CXXThisExpr>(expression)) {
        const auto key = analysisKey + ":memory:this";
        if (!locations.count(key)) {
          MemoryLocationRecord record{key, "parameter", "this", "", "", "", {}, false,
                                      false, true};
          qualifyLocation(record, expression->getType());
          locations.emplace(key, std::move(record));
        }
        return key;
      }
      if (const auto *member = llvm::dyn_cast<clang::MemberExpr>(expression)) {
        const auto *field = llvm::dyn_cast<clang::FieldDecl>(member->getMemberDecl());
        if (!field)
          return unknownKey;
        std::string base = locationForLValue(member->getBase());
        if (member->isArrow()) {
          auto dereference = analysisKey + ":memory:deref:" + base;
          // Preserve the base path here: resetting every `->` to `*` hid deep
          // field chains from the deterministic access-path budget.
          auto dereferencePath = locations.at(base).accessPath;
          dereferencePath.push_back("*");
          dereference = addDerivedLocation(dereference, "dereference", "*(" +
                                               locations.at(base).name + ")",
                                           member->getBase()->getType()->getPointeeType(), base,
                                           std::move(dereferencePath));
          base = dereference;
        }
        auto path = locations.at(base).accessPath;
        path.push_back(field->getNameAsString());
        if (field->getParent() && field->getParent()->isUnion())
          incompleteReasons.insert("union_storage");
        const auto fieldKey = source_.declKey(field, "variable");
        emitSymbol(field, "variable");
        return addDerivedLocation(analysisKey + ":memory:field:" + base + ":" + fieldKey,
                                  "field", locations.at(base).name + "." +
                                               field->getNameAsString(),
                                  field->getType(), base, std::move(path));
      }
      if (const auto *unary = llvm::dyn_cast<clang::UnaryOperator>(expression);
          unary && unary->getOpcode() == clang::UO_Deref) {
        const std::string base = locationForLValue(unary->getSubExpr());
        auto path = locations.at(base).accessPath;
        path.push_back("*");
        return addDerivedLocation(analysisKey + ":memory:deref:" + base, "dereference",
                                  "*(" + locations.at(base).name + ")", expression->getType(),
                                  base, std::move(path));
      }
      if (llvm::isa<clang::ArraySubscriptExpr>(expression)) {
        incompleteReasons.insert("pointer_arithmetic_or_unknown_index");
        return unknownKey;
      }
      incompleteReasons.insert("unknown_lvalue");
      return unknownKey;
    };

    std::function<std::string(const clang::Expr *)> summaryLocationForExpr;
    summaryLocationForExpr = [&](const clang::Expr *raw) -> std::string {
      if (!raw)
        return unknownKey;
      const auto *expression = raw->IgnoreParenImpCasts();
      if (const auto *address = llvm::dyn_cast<clang::UnaryOperator>(expression);
          address && address->getOpcode() == clang::UO_AddrOf)
        return locationForLValue(address->getSubExpr());
      if (llvm::isa<clang::DeclRefExpr, clang::MemberExpr>(expression) ||
          (llvm::isa<clang::UnaryOperator>(expression) &&
           llvm::cast<clang::UnaryOperator>(expression)->getOpcode() == clang::UO_Deref))
        return locationForLValue(expression);
      std::string result;
      for (const auto *child : expression->children()) {
        const auto *childExpression = llvm::dyn_cast_or_null<clang::Expr>(child);
        if (!childExpression)
          continue;
        const auto candidate = summaryLocationForExpr(childExpression);
        if (candidate == unknownKey)
          continue;
        if (!result.empty() && result != candidate)
          return unknownKey;
        result = candidate;
      }
      return result.empty() ? unknownKey : result;
    };

    const auto cfgElementKey = [&](const clang::CFGBlock &block, unsigned index) {
      return blockKey(block) + ":element:" + std::to_string(index);
    };
    std::map<const clang::Stmt *, std::pair<std::string, std::string>> statementAnchors;
    std::vector<const clang::CFGBlock *> blocks(cfg.begin(), cfg.end());
    std::sort(blocks.begin(), blocks.end(), [](const auto *left, const auto *right) {
      return left->getBlockID() < right->getBlockID();
    });
    for (const auto *block : blocks) {
      unsigned index = 0;
      for (const auto &element : *block) {
        if (const auto *statement = elementStatement(element))
          statementAnchors.emplace(statement,
                                   std::make_pair(blockKey(*block), cfgElementKey(*block, index)));
        ++index;
      }
    }

    const std::string entryBlockKey = blockKey(cfg.getEntry());
    const auto addAccess = [&](const std::string &block, const std::string &element,
                               const std::string &location, llvm::StringRef kind,
                               const clang::Stmt *statement,
                               const clang::Expr *assigned = nullptr) -> DataAccessRecord & {
      const unsigned sequence = sequences[block]++;
      std::string key = analysisKey + ":access:" + block + ":" +
                        std::to_string(sequence) + ":" + kind.str() + ":" + location;
      auto &records = accessesByBlock[block];
      records.push_back(DataAccessRecord{key, block, element, location, kind.str(), sequence,
                                         statement, assigned,
                                         statement ? source_.source(statement->getSourceRange())
                                                   : std::string{},
                                         {}, true});
      return records.back();
    };

    for (const auto *parameter : function->parameters()) {
      const auto location = locationForDecl(parameter);
      parameterLocations.push_back(location);
      parameterModes.push_back(parameterMode(parameter->getType()));
      auto &access =
          addAccess(entryBlockKey, "", location, "parameter_definition", nullptr);
      if (locations.at(location).tracksPointsTo) {
        access.pointsToComplete = false;
        incompleteReasons.insert("external_parameter_points_to");
      }
    }
    if (llvm::isa<clang::CXXMethodDecl>(function)) {
      const auto thisKey = analysisKey + ":memory:this";
      if (!locations.count(thisKey)) {
        MemoryLocationRecord record{thisKey, "parameter", "this", "", "", "", {}, false,
                                    false, true};
        locations.emplace(thisKey, std::move(record));
      }
      addAccess(entryBlockKey, "", thisKey, "parameter_definition", nullptr);
    }

    std::function<void(const clang::Expr *, llvm::StringRef, const std::string &,
                       const std::string &, const clang::Stmt *)>
        addReads;
    addReads = [&](const clang::Expr *raw, llvm::StringRef kind, const std::string &block,
                   const std::string &element, const clang::Stmt *anchor) {
      if (!raw)
        return;
      const auto *expression = raw->IgnoreParenImpCasts();
      if (const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression)) {
        handledExpressions.insert(expression);
        if (llvm::isa<clang::FunctionDecl>(reference->getDecl()))
          return;
        if (const auto *value = llvm::dyn_cast<clang::ValueDecl>(reference->getDecl()))
          addAccess(block, element, locationForDecl(value), kind, anchor);
        return;
      }
      if (const auto *member = llvm::dyn_cast<clang::MemberExpr>(expression)) {
        handledExpressions.insert(expression);
        if (llvm::isa<clang::FieldDecl>(member->getMemberDecl()))
          addAccess(block, element, locationForLValue(member), kind, anchor);
        addReads(member->getBase(), "read", block, element, anchor);
        return;
      }
      if (const auto *unary = llvm::dyn_cast<clang::UnaryOperator>(expression);
          unary && unary->getOpcode() == clang::UO_Deref) {
        handledExpressions.insert(expression);
        addReads(unary->getSubExpr(), "read", block, element, anchor);
        addAccess(block, element, locationForLValue(unary), kind, anchor);
        return;
      }
      if (llvm::isa<clang::CXXReinterpretCastExpr>(expression))
        incompleteReasons.insert("reinterpret_cast");
      if (const auto *binary = llvm::dyn_cast<clang::BinaryOperator>(expression);
          binary && (binary->getOpcode() == clang::BO_Add ||
                     binary->getOpcode() == clang::BO_Sub) &&
          (binary->getLHS()->getType()->isPointerType() ||
           binary->getRHS()->getType()->isPointerType()))
        incompleteReasons.insert("pointer_arithmetic");
      for (const auto *child : expression->children())
        if (const auto *childExpression = llvm::dyn_cast_or_null<clang::Expr>(child))
          addReads(childExpression, kind, block, element, anchor);
    };

    const auto returnLocation = [&]() {
      const std::string key = analysisKey + ":memory:return";
      if (!locations.count(key)) {
        MemoryLocationRecord record{key, "return", "$return", "", "", "", {}, false,
                                    false, tracksPointsTo(function->getReturnType())};
        qualifyLocation(record, function->getReturnType());
        locations.emplace(key, std::move(record));
      }
      return key;
    };

    for (const auto *block : blocks) {
      const auto currentBlockKey = blockKey(*block);
      unsigned elementIndex = 0;
      for (const auto &element : *block) {
        const auto *statement = elementStatement(element);
        const auto elementKey = cfgElementKey(*block, elementIndex++);
        if (!statement)
          continue;
        if (const auto *declaration = llvm::dyn_cast<clang::DeclStmt>(statement)) {
          for (const auto *decl : declaration->decls()) {
            const auto *variable = llvm::dyn_cast<clang::VarDecl>(decl);
            if (!variable)
              continue;
            const auto location = locationForDecl(variable);
            if (const auto *initializer = variable->getInit()) {
              addReads(initializer, "read", currentBlockKey, elementKey, statement);
              addAccess(currentBlockKey, elementKey, location, "initialization", statement,
                        initializer);
            } else {
              // A declaration without an initializer is not a reaching definition.
              if (locations.at(location).tracksPointsTo)
                incompleteReasons.insert("uninitialized_pointer_or_reference");
            }
          }
          continue;
        }
        if (const auto *binary = llvm::dyn_cast<clang::BinaryOperator>(statement);
            binary && binary->isAssignmentOp()) {
          handledExpressions.insert(binary->getLHS()->IgnoreParenImpCasts());
          const auto location = locationForLValue(binary->getLHS());
          if (binary->isCompoundAssignmentOp())
            addReads(binary->getLHS(), "read", currentBlockKey, elementKey, statement);
          addReads(binary->getRHS(), "read", currentBlockKey, elementKey, statement);
          addAccess(currentBlockKey, elementKey, location,
                    binary->isCompoundAssignmentOp() ? "compound_assignment" : "assignment",
                    statement, binary->getRHS());
          continue;
        }
        if (const auto *unary = llvm::dyn_cast<clang::UnaryOperator>(statement);
            unary && unary->isIncrementDecrementOp()) {
          handledExpressions.insert(unary->getSubExpr()->IgnoreParenImpCasts());
          const auto location = locationForLValue(unary->getSubExpr());
          addAccess(currentBlockKey, elementKey, location, "read", statement);
          addAccess(currentBlockKey, elementKey, location,
                    unary->isIncrementOp() ? "increment" : "decrement", statement);
          continue;
        }
        if (const auto *returned = llvm::dyn_cast<clang::ReturnStmt>(statement)) {
          addReads(returned->getRetValue(), "return_value", currentBlockKey, elementKey,
                   statement);
          addAccess(currentBlockKey, elementKey, returnLocation(), "assignment", statement,
                    returned->getRetValue());
          continue;
        }
        if (const auto *call = llvm::dyn_cast<clang::CallExpr>(statement)) {
          const auto ownerKind = symbolKind(function).value_or("function");
          const auto ownerKey = source_.declKey(function, ownerKind);
          const auto callRange = call->getSourceRange();
          const auto callsiteKey =
              "callsite:" + ownerKey + ":" + call->getStmtClassName() + ":" +
              std::to_string(source_.offset(callRange.getBegin(), true)) + ":" +
              std::to_string(source_.offset(callRange.getBegin(), false)) + ":" +
              std::to_string(source_.offset(callRange.getEnd(), false));
          SummaryCallRecord summaryCall;
          summaryCall.callsiteKey = callsiteKey;
          for (const auto *argument : call->arguments())
            addReads(argument, "call_argument", currentBlockKey, elementKey, statement);
          if (!call->getType()->isVoidType()) {
            const auto callLocation = addDerivedLocation(
                analysisKey + ":memory:call-return:" +
                    std::to_string(source_.offset(call->getExprLoc(), false)),
                "return", "$call-return@" +
                              std::to_string(source_.offset(call->getExprLoc(), false)),
                call->getType(), "", {});
            auto &result =
                addAccess(currentBlockKey, elementKey, callLocation, "call_return", statement);
            summaryCall.resultLocation = callLocation;
            summaryCall.resultAccess = result.key;
          }
          if (!call->getDirectCallee()) {
            addReads(call->getCallee(), "read", currentBlockKey, elementKey, statement);
            // Dependent templates have no concrete runtime callee expression yet;
            // treating them as pointer calls overwrote their more precise reason.
            if (!call->isTypeDependent() && !call->isValueDependent() &&
                !call->getCallee()->isTypeDependent() &&
                !call->getCallee()->isValueDependent())
              indirectCalls.push_back(IndirectCallRecord{call, function, currentBlockKey,
                                                         call->getCallee(),
                                                         sequences[currentBlockKey]});
          }
          const auto *callee = call->getDirectCallee();
          const bool effectFree = callee &&
                                  (callee->hasAttr<clang::PureAttr>() ||
                                   callee->hasAttr<clang::ConstAttr>());
          bool escapingArgument = false;
          unsigned argumentIndex = 0;
          for (const auto *argument : call->arguments()) {
            auto type = argument->getType();
            if (type->isPointerType() ||
                (type->isReferenceType() &&
                 !type.getNonReferenceType().isConstQualified()))
              escapingArgument = true;
            const auto *stripped = argument->IgnoreParenImpCasts();
            const clang::Expr *escapedStorage = nullptr;
            if (const auto *address = llvm::dyn_cast<clang::UnaryOperator>(stripped);
                address && address->getOpcode() == clang::UO_AddrOf)
              escapedStorage = address->getSubExpr();
            if (callee && argumentIndex < callee->getNumParams()) {
              const auto parameterType = callee->getParamDecl(argumentIndex)->getType();
              if (parameterType->isReferenceType() &&
                  !parameterType.getNonReferenceType().isConstQualified())
                escapedStorage = argument;
            }
            const clang::Expr *boundStorage = argument;
            if (const auto *address = llvm::dyn_cast<clang::UnaryOperator>(stripped);
                address && address->getOpcode() == clang::UO_AddrOf)
              boundStorage = address->getSubExpr();
            const auto boundLocation = summaryLocationForExpr(boundStorage);
            summaryCall.argumentLocations.push_back(boundLocation);
            bool writebackCandidate = false;
            if (callee && argumentIndex < callee->getNumParams()) {
              const auto mode = parameterMode(callee->getParamDecl(argumentIndex)->getType());
              writebackCandidate = mode == "reference" || mode == "rvalue_reference" ||
                                   mode == "pointer";
            } else {
              writebackCandidate = type->isPointerType() || type->isReferenceType();
            }
            summaryCall.writebackCandidates.push_back(writebackCandidate);
            if (!effectFree && escapedStorage) {
              const auto escapedLocation = locationForLValue(escapedStorage);
              const auto found = locations.find(escapedLocation);
              if (found != locations.end() && found->second.tracksPointsTo) {
                auto &clobber = addAccess(currentBlockKey, elementKey, escapedLocation,
                                          "unknown_clobber", statement);
                clobber.pointsToComplete = false;
              }
            }
            ++argumentIndex;
          }
          summaryCalls.push_back(std::move(summaryCall));
          if (!effectFree &&
              (escapingArgument || !callee || !source_.relative(callee->getLocation()))) {
            addAccess(currentBlockKey, elementKey, unknownKey, "unknown_clobber", statement);
            incompleteReasons.insert(escapingArgument ? "address_escape_or_unknown_call_effects"
                                                      : "unknown_call_effects");
          }
          continue;
        }
        if (llvm::isa<clang::AsmStmt>(statement)) {
          addAccess(currentBlockKey, elementKey, unknownKey, "unknown_clobber", statement);
          incompleteReasons.insert("inline_assembly");
        }
      }
      if (const auto *condition =
              llvm::dyn_cast_or_null<clang::Expr>(block->getTerminatorCondition()))
        addReads(condition, "condition", currentBlockKey, "", condition);
    }

    // AlwaysAdd puts leaf expressions in the CFG. Reads not owned by a definition,
    // return, condition, or call still need an explicit fact, but covered leaves must
    // not be duplicated.
    for (const auto &[statement, anchor] : statementAnchors) {
      const auto *expression = llvm::dyn_cast<clang::Expr>(statement);
      if (!expression || handledExpressions.count(expression->IgnoreParenImpCasts()))
        continue;
      if (llvm::isa<clang::DeclRefExpr, clang::MemberExpr>(expression) ||
          (llvm::isa<clang::UnaryOperator>(expression) &&
           llvm::cast<clang::UnaryOperator>(expression)->getOpcode() == clang::UO_Deref))
        addReads(expression, "read", anchor.first, anchor.second, statement);
    }

    std::map<std::string, const clang::CFGBlock *> keyToBlock;
    std::map<std::string, std::vector<std::string>> predecessors;
    for (const auto *block : blocks) {
      const auto key = blockKey(*block);
      keyToBlock[key] = block;
      for (const auto &successor : block->succs()) {
        if (const auto *target = successor.getReachableBlock())
          predecessors[blockKey(*target)].push_back(key);
      }
    }
    for (auto &[_, values] : predecessors) {
      std::sort(values.begin(), values.end());
      values.erase(std::unique(values.begin(), values.end()), values.end());
    }

    const auto capPointsTo = [&](PointsToValue &value) {
      if (value.functions.size() + value.locations.size() <= kDataFlowMaxAliasTargets)
        return;
      incompleteReasons.insert("alias_target_cap_exceeded");
      value.complete = false;
      while (value.locations.size() > kDataFlowMaxAliasTargets)
        value.locations.erase(std::prev(value.locations.end()));
      const auto remaining = kDataFlowMaxAliasTargets - value.locations.size();
      std::vector<const clang::FunctionDecl *> functions(value.functions.begin(),
                                                          value.functions.end());
      std::sort(functions.begin(), functions.end(), [&](const auto *left, const auto *right) {
        const auto leftKind = llvm::isa<clang::CXXMethodDecl>(left) ? "method" : "function";
        const auto rightKind = llvm::isa<clang::CXXMethodDecl>(right) ? "method" : "function";
        return source_.declKey(left, leftKind) < source_.declKey(right, rightKind);
      });
      value.functions.clear();
      value.functions.insert(functions.begin(), functions.begin() +
                                                     std::min(remaining, functions.size()));
    };

    const auto functionTarget = [&](const clang::FunctionDecl *target) {
      const bool indexed = source_.relative(target->getLocation()).has_value();
      if (!indexed)
        incompleteReasons.insert("external_indirect_target");
      return PointsToValue{{target}, {}, indexed, false};
    };
    std::function<PointsToValue(const clang::Expr *, const DataFlowState &)> evaluatePointsTo;
    evaluatePointsTo = [&](const clang::Expr *raw, const DataFlowState &state) -> PointsToValue {
      if (!raw)
        return PointsToValue{{}, {}, false, false};
      const auto *expression = raw->IgnoreParenImpCasts();
      if (const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression)) {
        if (const auto *target = llvm::dyn_cast<clang::FunctionDecl>(reference->getDecl()))
          return functionTarget(target);
        if (const auto *value = llvm::dyn_cast<clang::ValueDecl>(reference->getDecl())) {
          const auto location = locationForDecl(value);
          if (const auto found = state.pointsTo.find(location); found != state.pointsTo.end())
            return found->second;
          if (!tracksPointsTo(value->getType()))
            return PointsToValue{{}, {location}, true, false};
          return PointsToValue{{}, {}, false, false};
        }
      }
      if (const auto *unary = llvm::dyn_cast<clang::UnaryOperator>(expression);
          unary && unary->getOpcode() == clang::UO_AddrOf) {
        const auto *operand = unary->getSubExpr()->IgnoreParenImpCasts();
        if (const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(operand)) {
          if (const auto *target = llvm::dyn_cast<clang::FunctionDecl>(reference->getDecl()))
            return functionTarget(target);
        }
        if (const auto *member = llvm::dyn_cast<clang::MemberExpr>(operand)) {
          if (const auto *target = llvm::dyn_cast<clang::FunctionDecl>(member->getMemberDecl()))
            return functionTarget(target);
        }
        return PointsToValue{{}, {locationForLValue(operand)}, true, false};
      }
      if (const auto *conditional = llvm::dyn_cast<clang::ConditionalOperator>(expression)) {
        auto left = evaluatePointsTo(conditional->getTrueExpr(), state);
        auto right = evaluatePointsTo(conditional->getFalseExpr(), state);
        left.functions.insert(right.functions.begin(), right.functions.end());
        left.locations.insert(right.locations.begin(), right.locations.end());
        left.complete = left.complete && right.complete;
        left.includesNull = left.includesNull || right.includesNull;
        capPointsTo(left);
        return left;
      }
      if (const auto *binary = llvm::dyn_cast<clang::BinaryOperator>(expression);
          binary && (binary->getOpcode() == clang::BO_PtrMemD ||
                     binary->getOpcode() == clang::BO_PtrMemI))
        return evaluatePointsTo(binary->getRHS(), state);
      if (llvm::isa<clang::CXXNullPtrLiteralExpr, clang::GNUNullExpr>(expression) ||
          (llvm::isa<clang::IntegerLiteral>(expression) &&
           llvm::cast<clang::IntegerLiteral>(expression)->getValue() == 0))
        return PointsToValue{{}, {}, true, true};
      if (const auto *lambda = llvm::dyn_cast<clang::LambdaExpr>(expression))
        return functionTarget(lambda->getCallOperator());
      if (llvm::isa<clang::CXXReinterpretCastExpr>(expression))
        incompleteReasons.insert("reinterpret_cast");
      if (const auto *binary = llvm::dyn_cast<clang::BinaryOperator>(expression);
          binary && (binary->getOpcode() == clang::BO_Add ||
                     binary->getOpcode() == clang::BO_Sub))
        incompleteReasons.insert("pointer_arithmetic");
      return PointsToValue{{}, {}, false, false};
    };

    const auto isDefinition = [](llvm::StringRef kind) {
      return kind == "parameter_definition" || kind == "initialization" ||
             kind == "assignment" || kind == "compound_assignment" ||
             kind == "increment" || kind == "decrement" || kind == "call_return" ||
             kind == "unknown_clobber";
    };
    const auto transfer = [&](DataFlowState state,
                              const std::vector<DataAccessRecord> &records) {
      for (const auto &access : records) {
        if (!isDefinition(access.kind))
          continue;
        state.definitions[access.locationKey] = {access.key};
        state.definitionsComplete[access.locationKey] = true;
        const auto location = locations.find(access.locationKey);
        if (location != locations.end() && location->second.tracksPointsTo) {
          const bool referenceHandle =
              (location->second.kind == "local" || location->second.kind == "parameter") &&
              location->second.typeName.find('&') != std::string::npos;
          if (referenceHandle && access.kind != "initialization" &&
              access.kind != "parameter_definition" && access.kind != "unknown_clobber")
            continue;
          auto value = access.assignedExpression
                           ? evaluatePointsTo(access.assignedExpression, state)
                           : PointsToValue{{}, {}, access.pointsToComplete, false};
          state.pointsTo[access.locationKey] = std::move(value);
        }
      }
      return state;
    };
    const auto joinState = [&](const std::vector<std::string> &incoming,
                               const std::map<std::string, DataFlowState> &outputs) {
      DataFlowState joined;
      std::vector<const DataFlowState *> states;
      for (const auto &predecessor : incoming) {
        const auto found = outputs.find(predecessor);
        if (found != outputs.end())
          states.push_back(&found->second);
      }
      std::set<std::string> pointLocations;
      std::set<std::string> definitionLocations;
      for (const auto *state : states) {
        for (const auto &[location, definitions] : state->definitions) {
          joined.definitions[location].insert(definitions.begin(), definitions.end());
          definitionLocations.insert(location);
        }
        for (const auto &[location, _] : state->definitionsComplete)
          definitionLocations.insert(location);
        for (const auto &[location, _] : state->pointsTo)
          pointLocations.insert(location);
      }
      for (const auto &location : definitionLocations) {
        bool complete = states.size() == incoming.size();
        for (const auto *state : states) {
          const auto found = state->definitionsComplete.find(location);
          complete = complete && found != state->definitionsComplete.end() && found->second;
        }
        joined.definitionsComplete[location] = complete;
      }
      for (const auto &location : pointLocations) {
        PointsToValue destination;
        for (const auto *state : states) {
          const auto found = state->pointsTo.find(location);
          if (found == state->pointsTo.end()) {
            destination.complete = false;
            continue;
          }
          destination.functions.insert(found->second.functions.begin(),
                                       found->second.functions.end());
          destination.locations.insert(found->second.locations.begin(),
                                       found->second.locations.end());
          destination.complete = destination.complete && found->second.complete;
          destination.includesNull = destination.includesNull || found->second.includesNull;
        }
        capPointsTo(destination);
        joined.pointsTo.emplace(location, std::move(destination));
      }
      return joined;
    };

    std::map<std::string, DataFlowState> inputs;
    std::map<std::string, DataFlowState> outputs;
    unsigned iterationCount = 0;
    bool converged = false;
    for (; iterationCount < kDataFlowMaxIterations; ++iterationCount) {
      bool changed = false;
      for (const auto *block : blocks) {
        const auto key = blockKey(*block);
        auto input = joinState(predecessors[key], outputs);
        auto output = transfer(input, accessesByBlock[key]);
        if (!inputs.count(key) || !(inputs[key] == input) || !outputs.count(key) ||
            !(outputs[key] == output)) {
          inputs[key] = std::move(input);
          outputs[key] = std::move(output);
          changed = true;
        }
      }
      if (!changed) {
        converged = true;
        ++iterationCount;
        break;
      }
    }
    if (!converged)
      incompleteReasons.insert("iteration_cap_exceeded");

    std::set<std::string> emittedEvidence;
    const auto emitAccessEvidence = [&](llvm::StringRef relation, llvm::StringRef certainty,
                                        llvm::StringRef reason, const std::string &sourceAccess,
                                        const std::string &targetAccess,
                                        const clang::Stmt *statement) {
      const std::string key = analysisKey + ":evidence:" + relation.str() + ":" +
                              sourceAccess + ":" + targetAccess;
      if (!emittedEvidence.insert(key).second)
        return;
      llvm::json::Object fact{{"fact", "data_flow_evidence_v1"},
                              {"key", key},
                              {"analysis_key", analysisKey},
                              {"graph_key", graphKey},
                              {"relation", relation.str()},
                              {"certainty", certainty.str()},
                              {"reason", reason.str()},
                              {"source_access_key", sourceAccess},
                              {"target_access_key", targetAccess}};
      if (statement)
        if (auto span = source_.span(statement->getSourceRange(), false))
          fact["evidence_span"] = std::move(*span);
      sink_.add("data-flow-evidence:" + key, std::move(fact));
    };
    const auto emitAliasEvidence = [&](llvm::StringRef relation, llvm::StringRef certainty,
                                       llvm::StringRef reason, const std::string &sourceLocation,
                                       const std::string &targetLocation,
                                       const clang::Stmt *statement) {
      const std::string key = analysisKey + ":evidence:" + relation.str() + ":" +
                              sourceLocation + ":" + targetLocation;
      if (!emittedEvidence.insert(key).second)
        return;
      llvm::json::Object fact{{"fact", "data_flow_evidence_v1"},
                              {"key", key},
                              {"analysis_key", analysisKey},
                              {"graph_key", graphKey},
                              {"relation", relation.str()},
                              {"certainty", certainty.str()},
                              {"reason", reason.str()},
                              {"source_location_key", sourceLocation},
                              {"target_location_key", targetLocation}};
      if (statement)
        if (auto span = source_.span(statement->getSourceRange(), false))
          fact["evidence_span"] = std::move(*span);
      sink_.add("data-flow-evidence:" + key, std::move(fact));
    };

    std::map<const clang::CallExpr *, PointsToValue> callValues;
    for (const auto *block : blocks) {
      const auto key = blockKey(*block);
      DataFlowState state = inputs[key];
      auto calls = std::vector<IndirectCallRecord>{};
      for (const auto &call : indirectCalls)
        if (call.blockKey == key)
          calls.push_back(call);
      std::sort(calls.begin(), calls.end(), [](const auto &left, const auto &right) {
        return left.sequence < right.sequence;
      });
      std::size_t nextCall = 0;
      for (auto &access : accessesByBlock[key]) {
        while (nextCall < calls.size() && calls[nextCall].sequence <= access.sequence) {
          callValues[calls[nextCall].expression] =
              evaluatePointsTo(calls[nextCall].calleeExpression, state);
          ++nextCall;
        }
        auto definitions = state.definitions[access.locationKey];
        if (!isDefinition(access.kind)) {
          const bool complete = state.definitionsComplete[access.locationKey];
          if (!complete)
            incompleteReasons.insert("reaching_definition_incomplete");
          const auto certainty = complete && definitions.size() == 1 ? "certain" : "possible";
          for (const auto &definition : definitions)
            emitAccessEvidence("reaching_definition", certainty,
                               complete && definitions.size() == 1
                                   ? "one definition reaches this use on every modeled path"
                                   : "this definition reaches the use on at least one CFG path",
                               definition, access.key, access.statement);
        } else {
          const bool complete = state.definitionsComplete[access.locationKey];
          const auto certainty = complete && definitions.size() == 1 ? "certain" : "possible";
          for (const auto &definition : definitions)
            emitAccessEvidence("overwrites", certainty,
                               complete && definitions.size() == 1
                                   ? "this definition is the unique reaching prior value"
                                   : "this definition is one of multiple reaching prior values",
                               definition, access.key, access.statement);
        }

        const auto location = locations.find(access.locationKey);
        if (location != locations.end() && !location->second.baseKey.empty() &&
            location->second.kind == "dereference") {
          const auto aliases = state.pointsTo.find(location->second.baseKey);
          if (aliases != state.pointsTo.end()) {
            const bool must = aliases->second.complete && !aliases->second.includesNull &&
                              aliases->second.functions.empty() &&
                              aliases->second.locations.size() == 1;
            for (const auto &target : aliases->second.locations)
              emitAliasEvidence(must ? "must_alias" : "may_alias",
                                must ? "certain" : "possible",
                                must ? "a complete singleton points-to set identifies this storage"
                                     : "the dereference may designate this storage",
                                access.locationKey, target, access.statement);
          }
        }

        if (isDefinition(access.kind)) {
          state.definitions[access.locationKey] = {access.key};
          state.definitionsComplete[access.locationKey] = true;
          if (location != locations.end() && location->second.tracksPointsTo) {
            const bool referenceHandle =
                (location->second.kind == "local" || location->second.kind == "parameter") &&
                location->second.typeName.find('&') != std::string::npos;
            if (referenceHandle && access.kind != "initialization" &&
                access.kind != "parameter_definition" && access.kind != "unknown_clobber")
              continue;
            auto value = access.assignedExpression
                             ? evaluatePointsTo(access.assignedExpression, state)
                             : PointsToValue{{}, {}, access.pointsToComplete, false};
            access.pointees.assign(value.functions.begin(), value.functions.end());
            access.pointsToComplete = value.complete;
            const bool must = value.complete && !value.includesNull &&
                              value.functions.empty() && value.locations.size() == 1;
            for (const auto &target : value.locations)
              emitAliasEvidence(must ? "must_alias" : "may_alias",
                                must ? "certain" : "possible",
                                must ? "a complete singleton initializer aliases this storage"
                                     : "this assignment may alias the target storage",
                                access.locationKey, target, access.statement);
            state.pointsTo[access.locationKey] = std::move(value);
          }
        }
      }
      while (nextCall < calls.size()) {
        callValues[calls[nextCall].expression] =
            evaluatePointsTo(calls[nextCall].calleeExpression, state);
        ++nextCall;
      }
    }

    for (const auto &call : indirectCalls) {
      auto value = callValues[call.expression];
      const auto range = call.expression->getSourceRange();
      const auto ownerKind = symbolKind(call.owner).value_or("function");
      const auto ownerKey = source_.declKey(call.owner, ownerKind);
      const auto callsiteKey = "callsite:" + ownerKey + ":" +
                               call.expression->getStmtClassName() + ":" +
                               std::to_string(source_.offset(range.getBegin(), true)) + ":" +
                               std::to_string(source_.offset(range.getBegin(), false)) + ":" +
                               std::to_string(source_.offset(range.getEnd(), false));
      const bool complete = value.complete;
      sink_.add("callsite-resolution:" + callsiteKey,
                {{"fact", "callsite_resolution_v1"},
                 {"callsite_key", callsiteKey},
                 {"target_set_complete", complete},
                 {"unresolved_reason", complete ? "" : "points_to_set_incomplete"}});
      // A nullable singleton is still only a possible runtime target.
      const bool certain = complete && !value.includesNull && value.functions.size() == 1;
      for (const auto *target : value.functions) {
        if (!source_.relative(target->getLocation())) {
          incompleteReasons.insert("external_indirect_target");
          continue;
        }
        const auto targetKind = llvm::isa<clang::CXXMethodDecl>(target) ? "method" : "function";
        emitSymbol(target, targetKind);
        const auto targetKey = source_.declKey(target, targetKind);
        sink_.add("call-target:data-flow:" + callsiteKey + ":" + targetKey,
                  {{"fact", "call_target_v1"},
                   {"callsite_key", callsiteKey},
                   {"target_key", targetKey},
                   {"certainty", certain ? "certain" : "possible"},
                   {"confidence", certain ? 1.0 : 0.5},
                   {"confidence_reason",
                    certain
                        ? "bounded data flow proved a complete singleton target set"
                        : "target belongs to a non-singleton or incomplete points-to set; the "
                          "value is deterministic ranking evidence, not a probability"},
                   {"derivation", certain ? "intraprocedural_singleton_points_to"
                                           : "intraprocedural_points_to_candidate"},
                   {"evidence_span", std::move(*source_.span(range, false))}});
      }
    }

    const auto functionKind = symbolKind(function).value_or("function");
    const auto functionKey = source_.declKey(function, functionKind);
    const std::string summaryKey = "function-summary:" + graphKey;
    const auto rootParameterIndex = [&](const std::string &rawLocation)
        -> std::optional<unsigned> {
      std::string location = rawLocation;
      std::set<std::string> visited;
      while (!location.empty() && visited.insert(location).second) {
        const auto parameter = std::find(parameterLocations.begin(), parameterLocations.end(),
                                         location);
        if (parameter != parameterLocations.end())
          return static_cast<unsigned>(std::distance(parameterLocations.begin(), parameter));
        const auto found = locations.find(location);
        if (found == locations.end())
          break;
        location = found->second.baseKey;
      }
      return std::nullopt;
    };
    const auto pathArray = [](const std::vector<std::string> &path) {
      llvm::json::Array result;
      for (const auto &component : path)
        result.push_back(component);
      return result;
    };

    std::set<std::string> summaryIncompleteReasons;
    const std::set<std::string> summaryCriticalReasons{
        "access_path_cap_exceeded", "alias_target_cap_exceeded", "atomic_access",
        "inline_assembly",          "iteration_cap_exceeded",  "location_cap_exceeded",
        "union_storage",            "volatile_access"};
    for (const auto &reason : incompleteReasons)
      if (summaryCriticalReasons.count(reason))
        summaryIncompleteReasons.insert(reason);

    llvm::json::Array modesJson;
    llvm::json::Array parameterLocationsJson;
    for (const auto &mode : parameterModes)
      modesJson.push_back(mode);
    for (const auto &location : parameterLocations)
      parameterLocationsJson.push_back(location);

    for (const auto &[block, records] : accessesByBlock) {
      (void)block;
      for (const auto &access : records) {
        if (access.kind == "parameter_definition" || access.kind == "call_return")
          continue;
        const auto found = locations.find(access.locationKey);
        if (found == locations.end()) {
          summaryIncompleteReasons.insert("unknown_summary_location");
          continue;
        }
        const auto parameterIndex = rootParameterIndex(access.locationKey);
        const bool retainedLocation =
            parameterIndex.has_value() || found->second.kind == "global" ||
            found->second.kind == "field" || found->second.kind == "dereference";
        if (!retainedLocation)
          continue;
        llvm::StringRef effectKind = isDefinition(access.kind) ? "write" : "read";
        if (access.kind == "unknown_clobber")
          effectKind = "escape";
        const auto effectKey = summaryKey + ":effect:" + effectKind.str() + ":" + access.key;
        llvm::json::Object effect{{"fact", "summary_effect_v1"},
                                  {"key", effectKey},
                                  {"summary_key", summaryKey},
                                  {"kind", effectKind.str()},
                                  {"location_kind", found->second.kind},
                                  {"certainty", "certain"},
                                  {"reason", "Clang CFG access contributes local summary evidence"},
                                  {"access_path", pathArray(found->second.accessPath)},
                                  {"location_key", access.locationKey},
                                  {"source_access_key", access.key}};
        if (parameterIndex)
          effect["parameter_index"] = static_cast<std::int64_t>(*parameterIndex);
        sink_.add("summary-effect:" + effectKey, std::move(effect));
        if (access.kind == "call_argument" && parameterIndex &&
            parameterModes[*parameterIndex] != "value" &&
            parameterModes[*parameterIndex] != "const_reference" &&
            parameterModes[*parameterIndex] != "const_pointer") {
          const auto escapeKey = summaryKey + ":effect:escape:" + access.key;
          sink_.add("summary-effect:" + escapeKey,
                    {{"fact", "summary_effect_v1"},
                     {"key", escapeKey},
                     {"summary_key", summaryKey},
                     {"kind", "escape"},
                     {"location_kind", found->second.kind},
                     {"certainty", "possible"},
                     {"reason", "mutable indirection is passed across a call boundary"},
                     {"parameter_index", static_cast<std::int64_t>(*parameterIndex)},
                     {"access_path", pathArray(found->second.accessPath)},
                     {"location_key", access.locationKey},
                     {"source_access_key", access.key}});
        }
      }
    }

    std::set<std::string> emittedReturnOrigins;
    std::function<void(const clang::Expr *, const DataAccessRecord &)> emitReturnOrigins;
    emitReturnOrigins = [&](const clang::Expr *raw, const DataAccessRecord &access) {
      if (!raw)
        return;
      const auto *expression = raw->IgnoreParenImpCasts();
      if (const auto *call = llvm::dyn_cast<clang::CallExpr>(expression)) {
        const auto range = call->getSourceRange();
        const auto callsiteKey =
            "callsite:" + functionKey + ":" + call->getStmtClassName() + ":" +
            std::to_string(source_.offset(range.getBegin(), true)) + ":" +
            std::to_string(source_.offset(range.getBegin(), false)) + ":" +
            std::to_string(source_.offset(range.getEnd(), false));
        const auto key = summaryKey + ":return:call:" + callsiteKey;
        if (emittedReturnOrigins.insert(key).second)
          sink_.add("summary-return:" + key,
                    {{"fact", "summary_return_origin_v1"},
                     {"key", key},
                     {"summary_key", summaryKey},
                     {"kind", "call_result"},
                     {"certainty", "certain"},
                     {"reason", "function returns a value derived from this call result"},
                     {"access_path", llvm::json::Array{}},
                     {"callsite_key", callsiteKey}});
        return;
      }
      if (llvm::isa<clang::IntegerLiteral, clang::FloatingLiteral,
                    clang::CXXBoolLiteralExpr, clang::CharacterLiteral>(expression)) {
        const auto key = summaryKey + ":return:constant:" + access.key;
        if (emittedReturnOrigins.insert(key).second)
          sink_.add("summary-return:" + key,
                    {{"fact", "summary_return_origin_v1"},
                     {"key", key},
                     {"summary_key", summaryKey},
                     {"kind", "constant"},
                     {"certainty", "certain"},
                     {"reason", "function return expression contains a literal value"},
                     {"access_path", llvm::json::Array{}}});
        return;
      }
      const bool locationExpression =
          llvm::isa<clang::DeclRefExpr, clang::MemberExpr>(expression) ||
          (llvm::isa<clang::UnaryOperator>(expression) &&
           llvm::cast<clang::UnaryOperator>(expression)->getOpcode() == clang::UO_Deref);
      if (locationExpression) {
        const auto locationKey = locationForLValue(expression);
        const auto found = locations.find(locationKey);
        if (found == locations.end()) {
          summaryIncompleteReasons.insert("unknown_return_origin");
          return;
        }
        const auto parameterIndex = rootParameterIndex(locationKey);
        const auto key = summaryKey + ":return:location:" + access.key + ":" + locationKey;
        if (emittedReturnOrigins.insert(key).second) {
          llvm::json::Object fact{{"fact", "summary_return_origin_v1"},
                                  {"key", key},
                                  {"summary_key", summaryKey},
                                  {"kind", "location"},
                                  {"certainty", "certain"},
                                  {"reason", "function return expression reads this storage"},
                                  {"location_kind", found->second.kind},
                                  {"access_path", pathArray(found->second.accessPath)},
                                  {"location_key", locationKey}};
          if (parameterIndex)
            fact["parameter_index"] = static_cast<std::int64_t>(*parameterIndex);
          sink_.add("summary-return:" + key, std::move(fact));
        }
        return;
      }
      bool foundChild = false;
      for (const auto *child : expression->children()) {
        if (const auto *childExpression = llvm::dyn_cast_or_null<clang::Expr>(child)) {
          foundChild = true;
          emitReturnOrigins(childExpression, access);
        }
      }
      if (!foundChild)
        summaryIncompleteReasons.insert("unknown_return_origin");
    };
    for (const auto &[block, records] : accessesByBlock) {
      (void)block;
      for (const auto &access : records)
        if (access.locationKey == returnLocation() && access.assignedExpression)
          emitReturnOrigins(access.assignedExpression, access);
    }

    for (const auto &call : summaryCalls) {
      for (std::size_t index = 0; index < call.argumentLocations.size(); ++index) {
        const auto &locationKey = call.argumentLocations[index];
        const auto found = locations.find(locationKey);
        const bool complete = found != locations.end() && locationKey != unknownKey;
        const auto parameterIndex = complete ? rootParameterIndex(locationKey) : std::nullopt;
        llvm::json::Object fact{{"fact", "call_argument_binding_v1"},
                                {"key", summaryKey + ":argument:" + call.callsiteKey + ":" +
                                            std::to_string(index)},
                                {"summary_key", summaryKey},
                                {"callsite_key", call.callsiteKey},
                                {"argument_index", static_cast<std::int64_t>(index)},
                                {"location_kind", complete ? found->second.kind : "unknown"},
                                {"access_path", complete
                                                    ? pathArray(found->second.accessPath)
                                                    : llvm::json::Array{}},
                                {"writeback_candidate", call.writebackCandidates[index]},
                                {"complete", complete},
                                {"incomplete_reason", complete ? "" : "unknown_argument_storage"}};
        if (complete)
          fact["location_key"] = locationKey;
        if (parameterIndex)
          fact["parameter_index"] = static_cast<std::int64_t>(*parameterIndex);
        sink_.add("call-argument-binding:" + call.callsiteKey + ":" +
                      std::to_string(index),
                  std::move(fact));
      }
      if (!call.resultLocation.empty() && !call.resultAccess.empty())
        sink_.add("call-result-binding:" + call.callsiteKey,
                  {{"fact", "call_result_binding_v1"},
                   {"key", summaryKey + ":result:" + call.callsiteKey},
                   {"summary_key", summaryKey},
                   {"callsite_key", call.callsiteKey},
                   {"location_key", call.resultLocation},
                   {"definition_access_key", call.resultAccess}});
    }

    llvm::json::Array summaryReasons;
    for (const auto &reason : summaryIncompleteReasons)
      summaryReasons.push_back(reason);
    sink_.add("function-summary:" + summaryKey,
              {{"fact", "function_summary_v1"},
               {"key", summaryKey},
               {"function_key", functionKey},
               {"graph_key", graphKey},
               {"analysis_key", analysisKey},
               {"parameter_modes", std::move(modesJson)},
               {"parameter_location_keys", std::move(parameterLocationsJson)},
               {"local_complete", summaryIncompleteReasons.empty()},
               {"local_incomplete_reasons", std::move(summaryReasons)}});

    for (const auto &[key, location] : locations) {
      llvm::json::Array accessPath;
      for (const auto &component : location.accessPath)
        accessPath.push_back(component);
      llvm::json::Object fact{{"fact", "memory_location_v1"},
                              {"key", key},
                              {"analysis_key", analysisKey},
                              {"graph_key", graphKey},
                              {"kind", location.kind},
                              {"name", location.name},
                              {"type_name", location.typeName},
                              {"access_path", std::move(accessPath)},
                              {"is_volatile", location.isVolatile},
                              {"is_atomic", location.isAtomic}};
      if (!location.declarationKey.empty())
        fact["declaration_key"] = location.declarationKey;
      if (!location.baseKey.empty())
        fact["base_key"] = location.baseKey;
      sink_.add("memory-location:" + key, std::move(fact));
    }
    for (const auto &[block, records] : accessesByBlock) {
      for (const auto &access : records) {
        std::vector<std::string> sortedPointeeKeys;
        for (const auto *pointee : access.pointees) {
          const auto targetKind = llvm::isa<clang::CXXMethodDecl>(pointee) ? "method" : "function";
          if (source_.relative(pointee->getLocation()))
            sortedPointeeKeys.push_back(source_.declKey(pointee, targetKind));
        }
        std::sort(sortedPointeeKeys.begin(), sortedPointeeKeys.end());
        llvm::json::Array pointeeKeys;
        for (const auto &key : sortedPointeeKeys)
          pointeeKeys.push_back(key);
        llvm::json::Object fact{{"fact", "data_access_v1"},
                                {"key", access.key},
                                {"analysis_key", analysisKey},
                                {"graph_key", graphKey},
                                {"block_key", block},
                                {"location_key", access.locationKey},
                                {"kind", access.kind},
                                {"sequence", static_cast<std::int64_t>(access.sequence)},
                                {"expression", access.expression},
                                {"pointee_keys", std::move(pointeeKeys)},
                                {"points_to_complete", access.pointsToComplete}};
        if (!access.elementKey.empty())
          fact["cfg_element_key"] = access.elementKey;
        if (access.statement)
          if (auto span = source_.span(access.statement->getSourceRange(), false))
            fact["span"] = std::move(*span);
        sink_.add("data-access:" + access.key, std::move(fact));
      }
    }

    llvm::json::Array reasons;
    for (const auto &reason : incompleteReasons)
      reasons.push_back(reason);
    llvm::json::Object limits{{"max_iterations",
                               static_cast<std::int64_t>(kDataFlowMaxIterations)},
                              {"max_alias_targets",
                               static_cast<std::int64_t>(kDataFlowMaxAliasTargets)},
                              {"max_access_path_depth",
                               static_cast<std::int64_t>(kDataFlowMaxAccessPathDepth)},
                              {"max_locations",
                               static_cast<std::int64_t>(kDataFlowMaxLocations)}};
    sink_.add("data-flow-analysis:" + analysisKey,
              {{"fact", "data_flow_analysis_v1"},
               {"key", analysisKey},
               {"graph_key", graphKey},
               {"complete", incompleteReasons.empty()},
               {"incomplete_reasons", std::move(reasons)},
               {"iteration_count", static_cast<std::int64_t>(iterationCount)},
               {"limits", std::move(limits)}});
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
