# Expression Language Support for Guards and Actions

**Status:** Draft
**Author:** sc maintainers
**Created:** 2024-12-25

## Problem Statement

The current sc schema uses opaque strings for guard expressions and action names:

```protobuf
message Guard {
  string expression = 1;  // e.g., "damage > 100"
}

message Action {
  string name = 1;
  google.protobuf.Struct parameters = 2;
}
```

This approach has limitations:

1. **No runtime evaluation** - Expressions are descriptive only, not executable
2. **No validation** - Syntax errors discovered only at runtime
3. **No portability** - Tool-specific expressions don't transfer between systems
4. **No type safety** - No way to verify expressions reference valid variables

## Goals

- Support evaluatable expressions for guards and actions
- Maintain backwards compatibility with existing string expressions
- Enable cross-language evaluation (Go, Rust, Python SDKs)
- Preserve extraction provenance (raw source expressions)
- Keep simple cases simple

## Non-Goals

- Full programming language support (use external scripts for complex logic)
- Visual expression builder (out of scope)
- Real-time expression editing (out of scope)

## Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **CEL** | Google Common Expression Language | Type-safe, fast, policy-focused, K8s/Firebase adoption | Learning curve |
| **Starlark** | Python dialect from Bazel | Familiar syntax, deterministic | Heavier runtime |
| **Expr** | Go expression library | Simple, fast | Go-only |
| **JsonLogic** | JSON-based rules | Language-agnostic | Verbose, hard to read |
| **Custom AST** | Define our own expression tree | Full control | Implementation burden |

## Proposed Solution

Support multiple expression types via a unified `Expression` message, with CEL as the primary evaluatable format.

### Expression Type Hierarchy

```
Expression
├── RAW      - Opaque string (extraction provenance, tool-specific)
├── CEL      - Primary evaluatable format (guards + actions)
│   └── cel_ast: cel.expr.Expr (canonical AST for portability)
└── STARLARK - Complex actions with control flow
```

### Use Cases by Type

| Use Case | Recommended Type | Example |
|----------|------------------|---------|
| Extracted guards | RAW | `(*((g_ram + 246))) & 128` |
| Authored guards | CEL | `context.damage > 100 && state == "vulnerable"` |
| Simple actions | CEL | `context.health = context.health - 10` |
| Complex actions | STARLARK | Multi-statement scripts with loops |
| Cross-SDK exchange | CEL + cel_ast | Portable via canonical `cel.expr.Expr` |

## Proto Schema Changes

### Using Google's Canonical CEL AST

Rather than defining a custom AST, we leverage Google's canonical CEL expression types from `cel.expr` (or `google.api.expr.v1alpha1`). This provides:

- **Complete** - All node types including uint64, bytes, comprehensions
- **Compatible** - cel-go's ParseSource() returns this exact type
- **Tooling** - CEL ecosystem tools work directly
- **Maintained** - Google maintains the spec
- **Tested** - Production use in K8s, Firebase, Istio

Reference: https://github.com/google/cel-spec/blob/master/proto/cel/expr/syntax.proto

```protobuf
import "cel/expr/syntax.proto";
import "cel/expr/checked.proto";  // optional: type info

// Expression represents an evaluatable expression
message Expression {
  ExpressionType type = 1;
  string source = 2;

  // Use canonical CEL AST (replaces custom ExpressionAST)
  cel.expr.Expr cel_ast = 3;

  // Type-checked version (optional, for static analysis)
  cel.expr.CheckedExpr checked = 4;

  // Optional metadata about expression origin
  google.protobuf.Struct metadata = 5;
}

enum ExpressionType {
  EXPRESSION_TYPE_UNSPECIFIED = 0;
  EXPRESSION_TYPE_RAW = 1;        // Opaque, tool-specific
  EXPRESSION_TYPE_CEL = 2;        // Common Expression Language
  EXPRESSION_TYPE_STARLARK = 3;   // Starlark Python dialect
}

// Updated Guard message
message Guard {
  // New: structured expression
  Expression condition = 2;

  // Deprecated: raw string (kept for backwards compat)
  string expression = 1 [deprecated = true];
}

// Updated Action message
message Action {
  string name = 1;
  google.protobuf.Struct parameters = 2;

  // New: executable action body
  Expression body = 3;
}

// Security configuration for expression evaluation
message ExpressionSecurityConfig {
  int32 max_depth = 1;           // Max AST nesting depth (default: 50)
  int32 max_size_bytes = 2;      // Max expression source size (default: 4096)
  int64 max_execution_ms = 3;    // Timeout for Starlark (default: 100)
  int64 max_memory_bytes = 4;    // Memory limit for Starlark (default: 10MB)
  int32 max_call_depth = 5;      // Max recursive call depth (default: 20)
  repeated string allowed_functions = 6;   // Whitelist of callable functions
  repeated string blocked_functions = 7;   // Blacklist (takes precedence)
  bool allow_starlark = 8;       // Enable Starlark (default: false for untrusted)
}
```

## Evaluation Context

Expressions are evaluated against a context containing:

```go
type EvalContext struct {
    // Current configuration
    ActiveStates []string

    // State variables
    Variables map[string]any

    // Event being processed
    Event *Event

    // Machine context
    Context *structpb.Struct

    // Custom functions
    Functions map[string]Function

    // Optional evaluator for RAW expressions (tool-specific)
    RawEvaluator func(expr string, ctx *EvalContext) (bool, error)

    // Security configuration
    Security *ExpressionSecurityConfig
}
```

### CEL Environment Setup

```go
import "github.com/google/cel-go/cel"

func NewGuardEvaluator(ctx *EvalContext) (*cel.Program, error) {
    env, _ := cel.NewEnv(
        cel.Variable("state", cel.StringType),
        cel.Variable("event", cel.StringType),
        cel.Variable("context", cel.MapType(cel.StringType, cel.DynType)),
        cel.Function("in_state", cel.Overload(
            "in_state_string",
            []*cel.Type{cel.StringType},
            cel.BoolType,
        )),
    )

    ast, _ := env.Compile(expr.Source)
    return env.Program(ast)
}
```

### Parsing to Canonical AST

CEL source can be parsed directly to the canonical AST:

```go
import (
    "github.com/google/cel-go/parser"
    exprpb "cel.dev/expr"
)

func ParseToCELAST(source string) (*exprpb.Expr, error) {
    ast, issues := parser.Parse(common.NewTextSource(source))
    if issues.Err() != nil {
        return nil, issues.Err()
    }
    return ast.Expr(), nil
}
```

### Converting RAW to CEL AST

For extraction tools that produce C-like expressions, convert to CEL AST:

```go
// Convert game extraction syntax to CEL AST
func ConvertRawToCEL(raw string) (*exprpb.Expr, error) {
    // Parse C-like: (*((g_ram + 246))) & 128
    // Emit CEL AST: _&_(g_ram[246], 128u)

    // 1. Parse the C-like expression
    cExpr, err := parseCExpression(raw)
    if err != nil {
        return nil, err
    }

    // 2. Transform to CEL semantics
    //    - Pointer arithmetic → index access
    //    - Bitwise ops → CEL function calls
    return transformToCEL(cExpr), nil
}
```

### Example Expressions

**Guards (CEL):**
```cel
// Simple comparison
damage > 100

// State check
in_state("vulnerable") && event == "ATTACK"

// Context access
context.player.health < 50

// Indexed access (for sprites)
sprites[k].ai_state == 2
```

**Actions (Starlark):**
```starlark
def on_enter(ctx):
    ctx.set("animation", "attack")
    ctx.emit("SOUND", {"name": "sword"})

def on_exit(ctx):
    ctx.set("animation", "idle")
```

## Implementation Plan

### Phase 1: Schema (Week 1)
- [ ] Add Expression, ExpressionAST messages to proto
- [ ] Update Guard and Action with new fields
- [ ] Regenerate SDK code
- [ ] Update validation to handle both old and new formats

### Phase 2: CEL Integration (Week 2)
- [ ] Add cel-go dependency
- [ ] Implement GuardEvaluator with CEL backend
- [ ] Add standard function library (in_state, has_event, etc.)
- [ ] Unit tests for expression evaluation

### Phase 3: AST Support (Week 3)
- [ ] Implement CEL → AST conversion
- [ ] Implement AST → CEL conversion
- [ ] Add AST evaluator for SDKs without CEL

### Phase 4: Starlark Actions (Week 4)
- [ ] Add go.starlark.net dependency
- [ ] Implement ActionExecutor with Starlark backend
- [ ] Sandbox configuration (allowed builtins)
- [ ] Integration tests

### Phase 5: SDK Parity (Week 5-6)
- [ ] Rust SDK: CEL via cel-rust or AST evaluator
- [ ] Python SDK: CEL via cel-python or AST evaluator
- [ ] Documentation and examples

## Migration Path

### Backwards Compatibility

The deprecated `Guard.expression` string field remains functional:

```go
func (g *Guard) Evaluate(ctx *EvalContext) (bool, error) {
    // Prefer new Expression field (takes precedence if both set)
    if g.Condition != nil {
        return evaluateExpression(g.Condition, ctx)
    }

    // Fall back to deprecated string
    if g.Expression != "" {
        // Treat as RAW type
        return evaluateRaw(g.Expression, ctx)
    }

    return true, nil  // No guard = always true
}

// evaluateRaw handles opaque expressions without a configured evaluator.
// Behavior is explicit and predictable:
func evaluateRaw(expr string, ctx *EvalContext) (bool, error) {
    // 1. Check for registered RAW evaluator (tool-specific)
    if ctx.RawEvaluator != nil {
        return ctx.RawEvaluator(expr, ctx)
    }

    // 2. Return error - RAW expressions require explicit handling
    return false, fmt.Errorf("no evaluator for RAW expression: %q", expr)
}
```

**Field Precedence:** If both `expression` (deprecated) and `condition` are set, `condition` takes precedence. This allows gradual migration while preserving backwards compatibility.

### Extraction Tool Updates

Tools like zelda3-extractor can:

1. **Short term:** Output RAW expressions preserving original syntax
2. **Medium term:** Parse to AST for portability
3. **Long term:** Convert to CEL for evaluation

```json
{
  "guard": {
    "condition": {
      "type": "RAW",
      "source": "(*((g_ram + 246))) & 128",
      "metadata": {
        "original_file": "sprites.c",
        "original_line": 1234
      }
    }
  }
}
```

## Security Considerations

### CEL
- Designed for untrusted input
- No I/O, no loops (unless explicitly enabled)
- Bounded execution time
- Built-in DoS protection

### Starlark
- Deterministic execution
- No filesystem/network access by default
- Configurable builtin restrictions

**Required limits for untrusted Starlark:**
- Execution timeout: 100ms default (via `ExpressionSecurityConfig.max_execution_ms`)
- Memory limit: 10MB default (via `ExpressionSecurityConfig.max_memory_bytes`)
- Call depth limit: 20 default (via `ExpressionSecurityConfig.max_call_depth`)
- Disabled builtins: `load`, `print` (no I/O)
- No access to `Thread.SetLocal` for sandboxing

### AST Evaluator
- Validates depth before evaluation (prevent stack overflow)
- Expression size limits (prevent memory exhaustion)
- No recursive descent beyond `max_depth`

### Recommendations
- Disable Starlark for untrusted statechart sources (`allow_starlark: false`)
- Use CEL-only mode for web/API contexts
- Audit custom functions for side effects
- Set explicit `ExpressionSecurityConfig` for each evaluation context

## Open Questions

1. **Should AST support be required for all SDKs?**
   - Pro: Guarantees portability
   - Con: Implementation burden

2. **Should we support expression compilation/caching?**
   - Pro: Performance for hot paths
   - Con: Memory overhead, complexity

3. **Custom function registry - global or per-machine?**
   - Global: Simpler, consistent
   - Per-machine: More flexible, isolated

4. **Should RAW expressions attempt CEL parsing as fallback?**
   - Pro: Easier migration
   - Con: Unpredictable behavior

## References

- [CEL Spec](https://github.com/google/cel-spec)
- [CEL-Go](https://github.com/google/cel-go)
- [Starlark Spec](https://github.com/bazelbuild/starlark)
- [go.starlark.net](https://pkg.go.dev/go.starlark.net/starlark)
- [JsonLogic](https://jsonlogic.com/)

## Changelog

### 2024-12-25: Design Review Feedback Incorporated

Based on peer review, the following changes were made:

1. **Use canonical CEL AST (major revision):**
   - Replaced custom ExpressionAST with Google's `cel.expr.Expr`
   - Added `cel.expr.CheckedExpr` for optional type information
   - Removed EXPRESSION_TYPE_AST (use CEL AST directly)
   - Reference: https://github.com/google/cel-spec/blob/master/proto/cel/expr/syntax.proto

2. **Security configuration:**
   - Added `ExpressionSecurityConfig` message with limits for depth, size, execution time, memory, and call depth
   - Documented required Starlark limits for untrusted input
   - Added AST evaluator security notes

3. **Migration path clarified:**
   - Defined explicit `evaluateRaw()` behavior (returns error without registered evaluator)
   - Documented field precedence when both deprecated and new fields are set
   - Added `RawEvaluator` and `Security` fields to EvalContext
   - Added ConvertRawToCEL() example for extraction tools

**Remaining considerations for future revisions:**
- Consider CEL-only approach (drop Starlark, use CEL extension functions)
- Add debugging/observability patterns
- Document testing patterns for expressions
- Address cross-SDK CEL compatibility (cel-rust is experimental)
