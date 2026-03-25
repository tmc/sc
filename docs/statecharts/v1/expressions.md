---
title: statecharts.v1
description: API Specification for the statecharts.v1 package.
---

<a name="expressions-proto"></a><p align="right"><a href="#top">Top</a></p>

<!-- begin services -->

<!-- begin services -->



<a name="statecharts-v1-Expression"></a>

### Expression

Expression represents an evaluatable predicate or action body.

FORMAL DEFINITION:
An expression E = (τ, src, ast) where:
  - τ ∈ ExpressionType: determines evaluation semantics
  - src ∈ String: human-readable source representation
  - ast ∈ AST ∪ {⊥}: optional pre-parsed syntax tree

EVALUATION SEMANTICS:
For guard expressions, evaluation follows:
  eval: Expr × Context × Event → Bool

For action expressions, evaluation follows:
  exec: Expr × Context → Context'

WELL-FORMEDNESS CONSTRAINTS:
1. src ≠ ∅ for all expression types (source is required)
2. τ = CEL ⟹ src is valid CEL syntax
3. τ = STARLARK ⟹ src is valid Starlark syntax
4. τ = RAW ⟹ no evaluation semantics guaranteed

STRING-FIRST DESIGN:
The source string is the canonical representation. AST fields are
optional optimizations for:
  - Cross-SDK portability (different CEL parser implementations)
  - Compilation caching (avoid repeated parsing)
  - Provenance preservation (RAW → AST conversion)




| Field | Type | Description |
| ----- | ---- | ----------- |
| type |[ExpressionType](#statecharts-v1-ExpressionType)| Expression type: τ ∈ {RAW, CEL, STARLARK} Determines parsing and evaluation semantics   |
| source |string| Source string: src ∈ String (required) Human-readable expression in the syntax specified by type.

Examples by type: CEL: "damage > 100 && in_state('vulnerable')" STARLARK: "def on_enter(ctx): ctx.set('x', 1)" RAW: "(*((g_ram + 246))) & 128"   |
| cel_ast |bytes| Pre-parsed CEL AST: ast ∈ cel.expr.Expr ∪ {⊥} Optional serialized google.api.expr.v1alpha1.Expr protobuf. When present, SDKs may skip parsing and deserialize directly.

Stored as bytes to avoid hard dependency on cel-spec protos. Deserialize with: cel.expr.Expr.parseFrom(cel_ast)   |
| cel_checked |bytes| Type-checked CEL expression: checked ∈ cel.expr.CheckedExpr ∪ {⊥} Optional serialized google.api.expr.v1alpha1.CheckedExpr. Contains type information for validated expressions.

SEMANTIC GUARANTEE: If checked ≠ ⊥, the expression has passed static type checking against a declared environment.   |
| metadata |Struct| Expression metadata: provenance and context information Used for debugging, auditing, and tool interoperability.

Common keys: "original_file": source file where expression was extracted "original_line": line number in source file "extracted_by": tool that created this expression   |




 <!-- end nested messages -->

 <!-- end nested enums -->




<a name="statecharts-v1-ExpressionSecurityConfig"></a>

### ExpressionSecurityConfig

ExpressionSecurityConfig constrains expression evaluation for safety.

FORMAL DEFINITION:
Security config S = (d, t, m, F_allow, F_block, flags) where:
  - d ∈ ℕ: maximum AST depth
  - t ∈ ℕ: maximum execution time (ms)
  - m ∈ ℕ: maximum memory (bytes)
  - F_allow ⊆ Functions: allowed function set
  - F_block ⊆ Functions: blocked function set
  - flags: boolean security flags

ENFORCEMENT SEMANTICS:
For any expression E with security config S:
  1. depth(E) > S.d ⟹ reject at parse time
  2. time(eval(E)) > S.t ⟹ terminate with timeout error
  3. memory(eval(E)) > S.m ⟹ terminate with OOM error
  4. calls(E) ∩ S.F_block ≠ ∅ ⟹ reject at parse time
  5. S.F_allow ≠ ∅ ∧ calls(E) ⊄ S.F_allow ⟹ reject at parse time

DEFENSE IN DEPTH:
Multiple layers protect against malicious expressions:
  - Static analysis (depth, function whitelist)
  - Runtime limits (timeout, memory)
  - Type restrictions (Starlark disabled by default)




| Field | Type | Description |
| ----- | ---- | ----------- |
| max_depth |int32| Maximum AST depth: d ∈ ℕ Prevents stack overflow from deeply nested expressions. Recommended: 32-64 for typical expressions.   |
| max_execution_ms |int64| Maximum execution time: t ∈ ℕ (milliseconds) Enforces termination for potentially expensive expressions. Recommended: 100-1000ms for guards, 1000-5000ms for actions.   |
| max_memory_bytes |int64| Maximum memory allocation: m ∈ ℕ (bytes) Prevents memory exhaustion from large intermediate values. Recommended: 1-10 MB depending on context.   |
| allowed_functions[] |string| Allowed functions: F_allow ⊆ Functions If non-empty, only these functions may be called. Empty set means all functions allowed (subject to F_block).   |
| blocked_functions[] |string| Blocked functions: F_block ⊆ Functions These functions are never allowed, even if in F_allow. Takes precedence: effective = F_allow \ F_block   |
| allow_starlark |bool| Allow Starlark expressions: flags.starlark ∈ Bool Default false: Starlark disabled for security. Enable only for trusted expression sources.   |
| allow_starlark_loops |bool| Allow loops in Starlark: flags.loops ∈ Bool Default false: prevents infinite loops. Requires allow_starlark = true to have effect.   |




 <!-- end nested messages -->

 <!-- end nested enums -->


 <!-- end messages -->

<!-- begin file-level enums -->


<a name="statecharts-v1-ExpressionType"></a>

### ExpressionType
ExpressionType classifies expression syntax and evaluation semantics.

FORMAL DEFINITION:
ExpressionType τ determines the interpretation function:
  interp: ExpressionType → (String → AST)

SEMANTIC PROPERTIES:
- RAW: interp(RAW) = ⊥ (no standard interpretation)
- CEL: interp(CEL) = cel_parse (CEL specification parser)
- STARLARK: interp(STARLARK) = starlark_parse (Starlark parser)

SECURITY PROPERTIES [CEL Spec, Section 1]:
- CEL: Hermetic (no I/O), terminating (no unbounded loops)
- STARLARK: Deterministic, but may loop (requires timeout)
- RAW: No security guarantees (opaque)



| Name | Number | Description |
| ---- | ------ | ----------- |
| EXPRESSION_TYPE_UNSPECIFIED | 0 |  Invalid: must specify type  |
| EXPRESSION_TYPE_RAW | 1 | RAW: Opaque expression with tool-specific syntax.

SEMANTICS: RAW expressions cannot be evaluated by the statechart runtime. They serve as documentation and provenance preservation.

USE CASES: - Extraction tools preserving original source syntax - Expressions pending conversion to evaluatable format - Domain-specific notations not supported by CEL/Starlark

EXAMPLE: "(*((g_ram + 246))) & 128" (C pointer arithmetic)   |
| EXPRESSION_TYPE_CEL | 2 | CEL: Common Expression Language [CEL Spec].

SEMANTICS: CEL provides a safe, fast expression language designed for configuration and policy evaluation. Key properties: - Type-safe with gradual typing - No side effects (pure functions) - Bounded execution (no loops without macros) - Designed for untrusted input

GRAMMAR (simplified): expr := or_expr or_expr := and_expr ('||' and_expr)* and_expr := rel_expr ('&&' rel_expr)* rel_expr := add_expr (('<'|'>'|'<='|'>='|'=='|'!=') add_expr)? ...

EXAMPLE: "context.damage > 100 && in_state('vulnerable')"   |
| EXPRESSION_TYPE_STARLARK | 3 | STARLARK: Python dialect for complex actions [STAR].

SEMANTICS: Starlark provides a deterministic Python subset with: - Familiar Python syntax - First-class functions - Loops and conditionals - No I/O or threading by default

SECURITY CONSIDERATIONS: Starlark allows loops and recursion, requiring: - Execution timeout enforcement - Memory limits - Restricted builtin set

EXAMPLE: def on_enter(ctx): for i in range(3): ctx.emit("TICK", {"n": i})   |


 <!-- end file-level enums -->

<!-- begin file-level extensions -->
 <!-- end file-level extensions -->

