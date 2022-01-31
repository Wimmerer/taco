#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <algorithm>
#include <unordered_set>
#include <taco.h>

#include "taco/ir/ir_visitor.h"
#include "codegen_c.h"
#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace ir {

// Some helper functions
namespace {

// Include stdio.h for printf
// stdlib.h for malloc/realloc
// math.h for sqrt
// MIN preprocessor macro
// This *must* be kept in sync with taco_tensor_t.h
const string cHeaders =
  "@enum TacoMode dense sparse\n"
  "struct TacoTensor{Tv}\n"
  "   order::Int32         // tensor order (number of modes)\n"
  "   dimensions::Vector{Int64}    // tensor dimensions\n"
  "   mode_ordering::Vector{Int32} // mode storage ordering\n"
  "   mode_types::Vector{TacoMode}    // mode storage types\n"
  "   indices::Vector{Vector{Vector{Int64}}}       // tensor index data (per mode)\n"
  "   vals::Vector{Tv}          // tensor values\n"
  "   fill_value::Tv    // tensor fill value\n"
  "end\n"
  "binarySearchAfter(array, arrayStart, arrayEnd, target)\n"
  "  if (array[arrayStart] >= target)\n"
  "    return arrayStart\n"
  "  end\n"
  "     lowerBound = arrayStart // always < target\n"
  "     upperBound = arrayEnd // always >= target\n"
  "  while (upperBound - lowerBound > 1)\n"
  "     mid = (upperBound + lowerBound) / 2\n"
  "     midValue = array[mid]\n"
  "     if (midValue < target)\n"
  "         lowerBound = mid\n"
  "     elseif midValue > target\n"
  "         upperBound = mid\n"
  "     else\n"
  "      return mid\n"
  "     end\n"
  "  end\n"
  "  return upperBound\n"
  "end\n"
  "binarySearchBefore(array, arrayStart, arrayEnd, target)\n"
  "  if (array[arrayStart] <= target)\n"
  "    return arrayStart\n"
  "  end\n"
  "     lowerBound = arrayStart // always <= target\n"
  "     upperBound = arrayEnd // always > target\n"
  "  while (upperBound - lowerBound > 1)\n"
  "     mid = (upperBound + lowerBound) / 2\n"
  "     midValue = array[mid]\n"
  "     if (midValue < target)\n"
  "         lowerBound = mid\n"
  "     elseif midValue > target\n"
  "         upperBound = mid\n"
  "     else\n"
  "      return mid\n"
  "     end\n"
  "  end\n"
  "  return lowerBound\n"
  "end\n"
  "TacoTensor(order, dimensions, modeordering, modetypes, fill::Tv) where {Tv}\n"
  "    t = TacoTensor{Tv}(\n"
  "      order,\n"
  "      dimensions,\n"
  "      modeordering,\n"
  "      modetypes,\n"
  "      Vector{Vector{Vector{Int64}}}(undef, order),\n"
  "      fill\n"
  "    )\n"
  "\n"
  "    for i in 1:order\n"
  "        if t.modetypes[i] == dense\n"
  "            t.indices[i] = Vector{Vector{Int64}}(undef, 1)\n"
  "        else\n"
  "          t.indices[i] = Vector{Vector{Int64}}(undef, 2)\n"
  "        end\n"
  "    end\n"
  "    return t\n"
  "end\n";
} // anonymous namespace

// find variables for generating declarations
// generates a single var for each GetProperty
class CodeGen_C::FindVars : public IRVisitor {
public:
  map<Expr, string, ExprCompare> varMap;

  // the variables for which we need to add declarations
  map<Expr, string, ExprCompare> varDecls;

  vector<Expr> localVars;

  // this maps from tensor, property, mode, index to the unique var
  map<tuple<Expr, TensorProperty, int, int>, string> canonicalPropertyVar;

  // this is for convenience, recording just the properties unpacked
  // from the output tensor so we can re-save them at the end
  map<tuple<Expr, TensorProperty, int, int>, string> outputProperties;

  // TODO: should replace this with an unordered set
  vector<Expr> outputTensors;
  vector<Expr> inputTensors;

  CodeGen_C *codeGen;

  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs, CodeGen_C *codeGen)
  : codeGen(codeGen) {
    for (auto v: inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(varMap.count(var)==0) << "Duplicate input found in codegen";
      inputTensors.push_back(v);
      varMap[var] = var->name;
    }
    for (auto v: outputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(varMap.count(var)==0) << "Duplicate output found in codegen";
      outputTensors.push_back(v);
      varMap[var] = var->name;
    }
  }

protected:
  using IRVisitor::visit;

  virtual void visit(const Var *op) {
    if (varMap.count(op) == 0) {
      varMap[op] = op->is_ptr? op->name : codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const For *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
  }

  virtual void visit(const GetProperty *op) {
    if (!util::contains(inputTensors, op->tensor) &&
        !util::contains(outputTensors, op->tensor)) {
      // Don't create header unpacking code for temporaries
      return;
    }

    if (varMap.count(op) == 0) {
      auto key =
              tuple<Expr,TensorProperty,int,int>(op->tensor,op->property,
                                                 (size_t)op->mode,
                                                 (size_t)op->index);
      if (canonicalPropertyVar.count(key) > 0) {
        varMap[op] = canonicalPropertyVar[key];
      } else {
        auto unique_name = codeGen->genUniqueName(op->name);
        canonicalPropertyVar[key] = unique_name;
        varMap[op] = unique_name;
        varDecls[op] = unique_name;
        if (util::contains(outputTensors, op->tensor)) {
          outputProperties[key] = unique_name;
        }
      }
    }
  }
};

CodeGen_C::CodeGen_C(std::ostream &dest, OutputKind outputKind, bool simplify)
    : CodeGen(dest, false, simplify, C), out(dest), outputKind(outputKind) {}

CodeGen_C::~CodeGen_C() {}

void CodeGen_C::compile(Stmt stmt, bool isFirst) {
  varMap = {};
  localVars = {};

  if (isFirst) {
    // output the headers
    out << cHeaders;
  }
  out << endl;
  // generate code for the Stmt
  stmt.accept(this);
}

void CodeGen_C::visit(const Function* func) {

  int numYields = countYields(func);
  emittingCoroutine = (numYields > 0);
  funcName = func->name;
  labelCount = 0;

  resetUniqueNameCounters();
  FindVars inputVarFinder(func->inputs, {}, this);
  func->body.accept(&inputVarFinder);
  FindVars outputVarFinder({}, func->outputs, this);
  func->body.accept(&outputVarFinder);

  // output function declaration
  doIndent();
  out << printFuncName(func, inputVarFinder.varDecls, outputVarFinder.varDecls);

  out << "\n";

  indent++;

  // find all the vars that are not inputs or outputs and declare them
  resetUniqueNameCounters();
  FindVars varFinder(func->inputs, func->outputs, this);
  func->body.accept(&varFinder);
  varMap = varFinder.varMap;
  localVars = varFinder.localVars;

  // Print variable declarations
  out << printDecls(varFinder.varDecls, func->inputs, func->outputs) << endl;

  if (emittingCoroutine) {
    out << printContextDeclAndInit(varMap, localVars, numYields, func->name)
        << endl;
  }

  // output body
  print(func->body);

  // output repack only if we allocated memory
  if (checkForAlloc(func))
    out << endl << printPack(varFinder.outputProperties, func->outputs);

  if (emittingCoroutine) {
    out << printCoroutineFinish(numYields, funcName);
  }

  doIndent();
  out << "return nothing\n";
  indent--;

  doIndent();
  out << "end\n";
}

void CodeGen_C::visit(const Load* op) {
  parentPrecedence = Precedence::LOAD;
  op->arr.accept(this);
  stream << "[";
  parentPrecedence = Precedence::LOAD;
  if(isa<Literal>(op->loc)){
    auto lit = op->loc.as<Literal>();
    if(lit->type.getKind() == Int32) {
      auto val = lit->getValue<int>();
      stream << val + 1;
    }
    else if(lit->type.getKind() == UInt32) {
        auto val = lit->getValue<unsigned int>();
        stream << val + 1;
    }
    else
      op->loc.accept(this);
    }
  else {
    op->loc.accept(this);
  }
  stream << "]";
}

void CodeGen_C::visit(const Literal* op) {
  switch (op->type.getKind()) {
    case Datatype::Complex64: {
      std::complex<float> val = op->getValue<std::complex<float>>();
      stream << val.real() << "+ " << val.imag() << "im";
    }
    break;
    case Datatype::Complex128: {
      std::complex<double> val = op->getValue<std::complex<double>>();
      stream << val.real() << "+ " << val.imag() << "im";
    }
    break;
    default: {
      IRPrinter::visit(op);
    }
  }
}

// We typically don't need to cast anymore for Julia, although this needs to be double checked, we can always do this as a reinterp.
void CodeGen_C::visit(const Cast* op) {
  // stream << "(" << keywordString(util::toString(op->type)) << ")";
  // parentPrecedence = Precedence::CAST;
  // op->a.accept(this);
  stream << "";
}


void CodeGen_C::visit(const IfThenElse* op) {
  taco_iassert(op->cond.defined());
  taco_iassert(op->then.defined());
  doIndent();
  stream << keywordString("if ");
  stream << " ";
  parentPrecedence = Precedence::TOP;
  op->cond.accept(this);

  Stmt scopedStmt = Stmt(to<Scope>(op->then)->scopedStmt);
  if (isa<Block>(scopedStmt)) {
    stream << "\n" << endl;
    op->then.accept(this);
    doIndent();
    stream << "end";
  }
  else if (isa<Assign>(scopedStmt)) {
    int tmp = indent;
    indent = 0;
    stream << " ";
    scopedStmt.accept(this);
    indent = tmp;
  }
  else {
    stream << endl;
    op->then.accept(this);
  }

  if (op->otherwise.defined()) {
    stream << "\n";
    doIndent();
    stream << keywordString("else");
    stream << "\n";
    op->otherwise.accept(this);
    doIndent();
    stream << "end";
  }
  stream << endl;
}

void CodeGen_C::visit(const VarDecl* op) {
  doIndent();
  if (emittingCoroutine) {
    op->var.accept(this);
    parentPrecedence = Precedence::TOP;
    stream << " = ";
    op->rhs.accept(this);
    stream << endl;
  } 
  else {
    string varName = varNameGenerator.getUniqueName(util::toString(op->var));
    varNames.insert({op->var, varName});
    op->var.accept(this);
    stream << "::";
    stream << printCType(op->var.type(), to<Var>(op->var)->is_ptr);
    taco_iassert(isa<Var>(op->var));
    parentPrecedence = Precedence::TOP;
    stream << " = ";
    op->rhs.accept(this);
    stream << endl;
  }
}

void CodeGen_C::visit(const Yield* op) {
  printYield(op, localVars, varMap, labelCount, funcName);
}

// For Vars, we replace their names with the generated name,
// since we match by reference (not name)
void CodeGen_C::visit(const Var* op) {
  taco_iassert(varMap.count(op) > 0) <<
      "Var " << op->name << " not found in varMap";
  if (emittingCoroutine) {
//    out << "TACO_DEREF(";
  }
  out << varMap[op];
  if (emittingCoroutine) {
//    out << ")";
  }
}

static string genVectorizePragma(int width) {
  stringstream ret;
  ret << "#pragma clang loop interleave(enable) ";
  //if (!width)
  //  ret << "vectorize(enable)";
  //else
  //  ret << "vectorize_width(" << width << ")";

  return ret.str();
}

static string getParallelizePragma(LoopKind kind) {
  stringstream ret;
  ret << "Threads.@threads";
  // switch (kind) {
  //   case LoopKind::Static:
  //     ret << "(static, 1)";
  //     break;
  //   case LoopKind::Dynamic:
  //     ret << "(dynamic, 1)";
  //     break;
  //   case LoopKind::Runtime:
  //     ret << "(runtime)";
  //     break;
  //   case LoopKind::Static_Chunked:
  //     ret << "(static)";
  //     break;
  //   default:
  //     break;
  // }
  return ret.str();
}

static string getUnrollPragma(size_t unrollFactor) {
  return "#pragma unroll " + std::to_string(unrollFactor);
}

static string getAtomicPragma() {
  return "#pragma omp atomic";
}

// The next two need to output the correct pragmas depending
// on the loop kind (Serial, Static, Dynamic, Vectorized)
//
// Docs for vectorization pragmas:
// http://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-loop-hint-optimizations
void CodeGen_C::visit(const For* op) {
  switch (op->kind) {
    case LoopKind::Vectorized:
      doIndent();
      out << genVectorizePragma(op->vec_width);
      out << "\n";
      break;
    case LoopKind::Static:
    case LoopKind::Dynamic:
    case LoopKind::Runtime:
    case LoopKind::Static_Chunked:
      doIndent();
      out << getParallelizePragma(op->kind);
      break;
    default:
      if (op->unrollFactor > 0) {
        doIndent();
        out << getUnrollPragma(op->unrollFactor) << endl;
      }
      break;
  }
  
  doIndent();
  stream << keywordString("for") << " ";
  op->var.accept(this);
  // Shouldn't need this.
  // stream << "::" << keywordString(util::toString(Datatype(op->var.type()).getKind()));
  stream << " in ";
  // This functionality (shifting to one based indices), should definitely be factored out.
  if(isa<Literal>(op->start)){
    auto lit = op->start.as<Literal>();
    if(lit->type.getKind() == Int32) {
      auto val = lit->getValue<int>();
      stream << val + 1;
    }
    else if(lit->type.getKind() == UInt32) {
        auto val = lit->getValue<unsigned int>();
        stream << val + 1;
    }
    else
      op->start.accept(this);
    }
  else {
    op->start.accept(this);
  }
  stream << keywordString(":");
  auto lit = op->increment.as<Literal>();
  if (lit == nullptr || !((lit->type.isInt()  && lit->equalsScalar(1)) ||
                         (lit->type.isUInt() && lit->equalsScalar(1)))) {
    stream << ":";
    op->increment.accept(this);
    stream << ":";
  }
  if(isa<Literal>(op->end)){
    auto lit = op->end.as<Literal>();
    if(lit->type.getKind() == Int32) {
      auto val = lit->getValue<int>();
      stream << val + 1;
    }
    else if(lit->type.getKind() == UInt32) {
        auto val = lit->getValue<unsigned int>();
        stream << val + 1;
    }
    else
      op->end.accept(this);
    }
  else {
    op->end.accept(this);
  }

  stream << "\n";

  op->contents.accept(this);
  doIndent();
  stream << "end";
  stream << endl;
}

void CodeGen_C::visit(const While* op) {
  // it's not clear from documentation that clang will vectorize
  // while loops
  // however, we'll output the pragmas anyway
  if (op->kind == LoopKind::Vectorized) {
    doIndent();
    out << genVectorizePragma(op->vec_width);
    out << "\n";
  }
  doIndent();
  stream << keywordString("while ");
  parentPrecedence = Precedence::TOP;
  op->cond.accept(this);
  stream << "\n";
  op->contents.accept(this);
  doIndent();
  stream << "end";
  stream << endl;
}

void CodeGen_C::visit(const GetProperty* op) {
  taco_iassert(varMap.count(op) > 0) <<
      "Property " << Expr(op) << " of " << op->tensor << " not found in varMap";
  out << varMap[op];
}

void CodeGen_C::visit(const Min* op) {
  if (op->operands.size() == 1) {
    op->operands[0].accept(this);
    return;
  }
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << "min(";
    op->operands[i].accept(this);
    stream << ",";
  }
  op->operands.back().accept(this);
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << ")";
  }
}

void CodeGen_C::visit(const Max* op) {
  if (op->operands.size() == 1) {
    op->operands[0].accept(this);
    return;
  }
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << "max(";
    op->operands[i].accept(this);
    stream << ",";
  }
  op->operands.back().accept(this);
  for (size_t i=0; i<op->operands.size()-1; i++) {
    stream << ")";
  }
}
// TODO:
void CodeGen_C::visit(const Allocate* op) {
  string elementType = printCType(op->var.type(), false);
  doIndent();
  op->var.accept(this);
  stream << " = ";
  if (op->is_realloc) {
    stream << "resize!(";
    op->var.accept(this);
    stream << ", ";
  }
  else {
    // If the allocation was requested to clear the allocated memory,
    // use calloc instead of malloc.
    if (op->clear) {
      stream << "zeros(" << elementType << ", ";
    } else {
      stream << "Vector{" << elementType << "}(";
    }
  }
  op->num_elements.accept(this);
  parentPrecedence = TOP;
  stream << ")";
  stream << endl;
}

void CodeGen_C::visit(const Sqrt* op) {
  taco_tassert(op->type.isFloat() && op->type.getNumBits() == 64) <<
      "Codegen doesn't currently support non-double sqrt";
  stream << "sqrt(";
  op->a.accept(this);
  stream << ")";
}

void CodeGen_C::visit(const Assign* op) {
  if (op->use_atomics) {
    doIndent();
    stream << getAtomicPragma();
  }
  doIndent();
  op->lhs.accept(this);
  parentPrecedence = Precedence::TOP;
  bool printed = false;
  if (simplify) {
    if (isa<ir::Add>(op->rhs)) {
      auto add = to<Add>(op->rhs);
      if (add->a == op->lhs) {
        const Literal* lit = add->b.as<Literal>();
        if (lit != nullptr && ((lit->type.isInt()  && lit->equalsScalar(1)) ||
                               (lit->type.isUInt() && lit->equalsScalar(1)))) {
          stream << "++";
        }
        else {
          stream << " += ";
          add->b.accept(this);
        }
        printed = true;
      }
    }
    else if (isa<Mul>(op->rhs)) {
      auto mul = to<Mul>(op->rhs);
      if (mul->a == op->lhs) {
        stream << " *= ";
        mul->b.accept(this);
        printed = true;
      }
    }
    else if (isa<BitOr>(op->rhs)) {
      auto bitOr = to<BitOr>(op->rhs);
      if (bitOr->a == op->lhs) {
        stream << " |= ";
        bitOr->b.accept(this);
        printed = true;
      }
    }
  }
  if (!printed) {
    stream << " = ";
    op->rhs.accept(this);
  }
  stream << endl;
}

void CodeGen_C::visit(const Free* op) {
  return;
}

void CodeGen_C::visit(const Continue*) {
  doIndent();
  stream << "continue" << endl;
}

void CodeGen_C::visit(const Break*) {
  doIndent();
  stream << "break" << endl;
}

void CodeGen_C::visit(const Print* op) {
  doIndent();
  stream << "printf(";
  stream << "\"" << op->fmt << "\"";
  for (auto e: op->params) {
    stream << ", ";
    e.accept(this);
  }
  stream << ")";
  stream << endl;
}

void CodeGen_C::visit(const Sort* op) {
  doIndent();
  stream << "sort(";
  parentPrecedence = Precedence::CALL;
  // This should be an acceptJoin, but I can't immediately figure out how to call it from here, so just copying for now.
  if (op->args.size() > 0) {
    op->args[0].accept(this);
  }
  for (size_t i=1; i < op->args.size(); ++i) {
    stream << ", ";
    op->args[i].accept(this);
  }
  stream << ")";
  stream << endl;
}

void CodeGen_C::visit(const Store* op) {
  if (op->use_atomics) {
    doIndent();
    stream << getAtomicPragma();
  }
  doIndent();
  op->arr.accept(this);
  stream << "[";
  parentPrecedence = Precedence::TOP;
  op->loc.accept(this);
  stream << "] = ";
  parentPrecedence = Precedence::TOP;
  op->data.accept(this);
  stream << endl;
}

void CodeGen_C::generateShim(const Stmt& func, stringstream &ret) {
  const Function *funcPtr = func.as<Function>();

  ret << "int _shim_" << funcPtr->name << "(void** parameterPack) {\n";
  ret << "  return " << funcPtr->name << "(";

  size_t i=0;
  string delimiter = "";

  const auto returnType = funcPtr->getReturnType();
  if (returnType.second != Datatype()) {
    ret << "(void**)(parameterPack[0]), ";
    ret << "(char*)(parameterPack[1]), ";
    ret << "(" << returnType.second << "*)(parameterPack[2]), ";
    ret << "(int32_t*)(parameterPack[3])";

    i = 4;
    delimiter = ", ";
  }

  for (auto output : funcPtr->outputs) {
    auto var = output.as<Var>();
    auto cast_type = var->is_tensor ? "taco_tensor_t*"
    : printCType(var->type, var->is_ptr);

    ret << delimiter << "(" << cast_type << ")(parameterPack[" << i++ << "])";
    delimiter = ", ";
  }
  for (auto input : funcPtr->inputs) {
    auto var = input.as<Var>();
    auto cast_type = var->is_tensor ? "taco_tensor_t*"
    : printCType(var->type, var->is_ptr);
    ret << delimiter << "(" << cast_type << ")(parameterPack[" << i++ << "])";
    delimiter = ", ";
  }
  ret << ");\n";
  ret << "}\n";
}
}
}
