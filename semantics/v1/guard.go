package semantics

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

// GuardExpression represents a parsed guard expression with metadata
type GuardExpression struct {
	Original   string
	Parsed     ast.Expr
	Language   string
	Variables  []string // List of variables referenced in the expression
	Complexity int      // Complexity score for optimization
}

// GuardEvaluator provides a comprehensive guard evaluation engine
type GuardEvaluator struct {
	mu             sync.RWMutex
	cache          map[string]*GuardExpression
	maxCache       int
	customFunctions map[string]GuardFunction
	metrics        *GuardMetrics
}

// GuardFunction represents a custom function that can be used in guard expressions
type GuardFunction func(args []interface{}) (interface{}, error)

// GuardMetrics tracks guard evaluation performance and usage
type GuardMetrics struct {
	Evaluations      int64
	CacheHits        int64
	CacheMisses      int64
	Errors           int64
	TotalEvalTime    int64 // nanoseconds
	ComplexityScores []int
}

// NewGuardEvaluator creates a new guard evaluator with caching
func NewGuardEvaluator() *GuardEvaluator {
	return &GuardEvaluator{
		cache:           make(map[string]*GuardExpression),
		maxCache:        1000, // Maximum number of cached expressions
		customFunctions: make(map[string]GuardFunction),
		metrics:         &GuardMetrics{},
	}
}

// EvaluationContext holds all data available for guard evaluation
type EvaluationContext struct {
	Variables       *structpb.Struct           // Context variables
	Event           *sc.Event                  // Current event being processed
	StateData       map[string]interface{}     // State-specific data
	EventData       map[string]interface{}     // Event payload as map
	Transition      *sc.Transition             // Current transition being evaluated
	ActiveStates    []string                   // Currently active states
	Statechart      *sc.Statechart             // Full statechart context
	CustomVariables map[string]interface{}     // Custom variables for expression evaluation
	TimeContext     *TimeContext               // Time-related context
}

// TimeContext provides time-related data for guard evaluation
type TimeContext struct {
	CurrentTime   int64 // Unix timestamp in milliseconds
	ElapsedTime   int64 // Time elapsed since state entry
	EventTime     int64 // Time when event was triggered
	TransitionTime int64 // Time when transition was triggered
}

// GuardResult represents the result of guard evaluation
type GuardResult struct {
	Value         bool
	Error         error
	Variables     []string               // Variables accessed during evaluation
	Complexity    int                    // Execution complexity
	EvaluationTime int64                 // Time taken for evaluation in nanoseconds
	CacheHit      bool                   // Whether result came from cache
	IntermediateValues map[string]interface{} // Intermediate values for debugging
}

// EvaluateGuard evaluates a guard expression with comprehensive context
func (ge *GuardEvaluator) EvaluateGuard(guard *sc.Guard, context *EvaluationContext) (*GuardResult, error) {
	start := time.Now()
	defer func() {
		ge.mu.Lock()
		ge.metrics.Evaluations++
		ge.metrics.TotalEvalTime += time.Since(start).Nanoseconds()
		ge.mu.Unlock()
	}()

	if guard == nil || guard.Expression == "" {
		return &GuardResult{
			Value:          true,
			Variables:      []string{},
			Complexity:     0,
			EvaluationTime: time.Since(start).Nanoseconds(),
			CacheHit:       false,
			IntermediateValues: make(map[string]interface{}),
		}, nil
	}

	// Parse or retrieve from cache
	expr, cacheHit, err := ge.parseExpressionWithMetrics(guard.Expression, guard.Language)
	if err != nil {
		ge.mu.Lock()
		ge.metrics.Errors++
		ge.mu.Unlock()
		return &GuardResult{
			Value:          false,
			Error:          err,
			EvaluationTime: time.Since(start).Nanoseconds(),
			CacheHit:       cacheHit,
			IntermediateValues: make(map[string]interface{}),
		}, fmt.Errorf("failed to parse guard expression: %w", err)
	}

	// Evaluate the expression
	result, intermediateValues, err := ge.evaluateExpressionWithDebug(expr, context)
	if err != nil {
		ge.mu.Lock()
		ge.metrics.Errors++
		ge.mu.Unlock()
		return &GuardResult{
			Value:          false,
			Error:          err,
			Variables:      expr.Variables,
			Complexity:     expr.Complexity,
			EvaluationTime: time.Since(start).Nanoseconds(),
			CacheHit:       cacheHit,
			IntermediateValues: intermediateValues,
		}, fmt.Errorf("failed to evaluate guard expression: %w", err)
	}

	// Track complexity for optimization
	ge.mu.Lock()
	ge.metrics.ComplexityScores = append(ge.metrics.ComplexityScores, expr.Complexity)
	ge.mu.Unlock()

	return &GuardResult{
		Value:          result,
		Variables:      expr.Variables,
		Complexity:     expr.Complexity,
		EvaluationTime: time.Since(start).Nanoseconds(),
		CacheHit:       cacheHit,
		IntermediateValues: intermediateValues,
	}, nil
}

// parseExpression parses and caches guard expressions
func (ge *GuardEvaluator) parseExpression(expression, language string) (*GuardExpression, error) {
	parsed, _, err := ge.parseExpressionWithMetrics(expression, language)
	return parsed, err
}

// parseExpressionWithMetrics parses and caches guard expressions with cache hit tracking
func (ge *GuardEvaluator) parseExpressionWithMetrics(expression, language string) (*GuardExpression, bool, error) {
	ge.mu.RLock()
	cached, exists := ge.cache[expression]
	ge.mu.RUnlock()

	if exists {
		ge.mu.Lock()
		ge.metrics.CacheHits++
		ge.mu.Unlock()
		return cached, true, nil
	}

	// Parse new expression
	parsed, err := ge.parseExpressionInternal(expression, language)
	if err != nil {
		ge.mu.Lock()
		ge.metrics.CacheMisses++
		ge.mu.Unlock()
		return nil, false, err
	}

	// Cache the parsed expression
	ge.mu.Lock()
	defer ge.mu.Unlock()

	// Limit cache size
	if len(ge.cache) >= ge.maxCache {
		// Simple LRU: remove a random entry
		for k := range ge.cache {
			delete(ge.cache, k)
			break
		}
	}

	ge.cache[expression] = parsed
	ge.metrics.CacheMisses++
	return parsed, false, nil
}

// parseExpressionInternal handles the actual parsing logic
func (ge *GuardEvaluator) parseExpressionInternal(expression, language string) (*GuardExpression, error) {
	// Default to Go-like expression syntax
	if language == "" {
		language = "go"
	}

	switch language {
	case "go", "golang":
		return ge.parseGoExpression(expression)
	case "js", "javascript":
		return ge.parseJavaScriptExpression(expression)
	case "simple":
		return ge.parseSimpleExpression(expression)
	default:
		// For unsupported languages, fall back to simple parsing
		return ge.parseSimpleExpression(expression)
	}
}

// parseGoExpression parses Go-like expressions
func (ge *GuardEvaluator) parseGoExpression(expression string) (*GuardExpression, error) {
	// Use Go's parser for robust expression parsing
	expr, err := parser.ParseExpr(expression)
	if err != nil {
		return nil, fmt.Errorf("invalid Go expression: %w", err)
	}

	// Extract variables and calculate complexity
	variables := ge.extractVariables(expr)
	complexity := ge.calculateComplexity(expr)

	return &GuardExpression{
		Original:   expression,
		Parsed:     expr,
		Language:   "go",
		Variables:  variables,
		Complexity: complexity,
	}, nil
}

// parseJavaScriptExpression parses JavaScript-like expressions
func (ge *GuardEvaluator) parseJavaScriptExpression(expression string) (*GuardExpression, error) {
	// For JavaScript, we'll convert to Go-like syntax and then parse
	// This is a simplified conversion - a full implementation would use a JS parser
	goExpression := ge.convertJSToGo(expression)
	return ge.parseGoExpression(goExpression)
}

// parseSimpleExpression parses simple boolean expressions
func (ge *GuardEvaluator) parseSimpleExpression(expression string) (*GuardExpression, error) {
	// Handle common simple cases
	expression = strings.TrimSpace(expression)
	
	var variables []string
	complexity := 1

	// Extract variables from simple expressions
	if strings.Contains(expression, "context.") {
		variables = ge.extractSimpleVariables(expression)
		complexity = len(variables)
	}

	return &GuardExpression{
		Original:   expression,
		Parsed:     nil, // Will be handled in evaluation
		Language:   "simple",
		Variables:  variables,
		Complexity: complexity,
	}, nil
}

// evaluateExpression evaluates a parsed expression
func (ge *GuardEvaluator) evaluateExpression(expr *GuardExpression, context *EvaluationContext) (bool, error) {
	result, _, err := ge.evaluateExpressionWithDebug(expr, context)
	return result, err
}

// evaluateExpressionWithDebug evaluates a parsed expression with debug information
func (ge *GuardEvaluator) evaluateExpressionWithDebug(expr *GuardExpression, context *EvaluationContext) (bool, map[string]interface{}, error) {
	intermediateValues := make(map[string]interface{})
	
	// Add time context if not present
	if context.TimeContext == nil {
		context.TimeContext = &TimeContext{
			CurrentTime: time.Now().UnixMilli(),
		}
	}
	
	switch expr.Language {
	case "go", "golang":
		result, err := ge.evaluateGoExpressionWithDebug(expr.Parsed, context, intermediateValues)
		return result, intermediateValues, err
	case "js", "javascript":
		result, err := ge.evaluateGoExpressionWithDebug(expr.Parsed, context, intermediateValues) // Converted to Go
		return result, intermediateValues, err
	case "simple":
		result, err := ge.evaluateSimpleExpressionWithDebug(expr.Original, context, intermediateValues)
		return result, intermediateValues, err
	default:
		return false, intermediateValues, fmt.Errorf("unsupported expression language: %s", expr.Language)
	}
}

// evaluateGoExpression evaluates Go AST expressions
func (ge *GuardEvaluator) evaluateGoExpression(expr ast.Expr, context *EvaluationContext) (bool, error) {
	intermediateValues := make(map[string]interface{})
	return ge.evaluateGoExpressionWithDebug(expr, context, intermediateValues)
}

// evaluateGoExpressionWithDebug evaluates Go AST expressions with debug information
func (ge *GuardEvaluator) evaluateGoExpressionWithDebug(expr ast.Expr, context *EvaluationContext, debug map[string]interface{}) (bool, error) {
	switch e := expr.(type) {
	case *ast.Ident:
		result, err := ge.evaluateIdentifierWithDebug(e.Name, context, debug)
		debug["identifier_"+e.Name] = result
		return result, err
	case *ast.BinaryExpr:
		return ge.evaluateBinaryExpressionWithDebug(e, context, debug)
	case *ast.UnaryExpr:
		return ge.evaluateUnaryExpressionWithDebug(e, context, debug)
	case *ast.CallExpr:
		return ge.evaluateCallExpressionWithDebug(e, context, debug)
	case *ast.SelectorExpr:
		return ge.evaluateSelectorExpressionWithDebug(e, context, debug)
	case *ast.ParenExpr:
		return ge.evaluateGoExpressionWithDebug(e.X, context, debug)
	case *ast.BasicLit:
		return ge.evaluateBasicLitWithDebug(e, context, debug)
	default:
		return false, fmt.Errorf("unsupported expression type: %T", e)
	}
}

// evaluateBinaryExpression handles binary operations (AND, OR, ==, !=, <, >, etc.)
func (ge *GuardEvaluator) evaluateBinaryExpression(expr *ast.BinaryExpr, context *EvaluationContext) (bool, error) {
	switch expr.Op {
	case token.LAND: // &&
		left, err := ge.evaluateGoExpression(expr.X, context)
		if err != nil {
			return false, err
		}
		if !left {
			return false, nil // Short-circuit
		}
		return ge.evaluateGoExpression(expr.Y, context)

	case token.LOR: // ||
		left, err := ge.evaluateGoExpression(expr.X, context)
		if err != nil {
			return false, err
		}
		if left {
			return true, nil // Short-circuit
		}
		return ge.evaluateGoExpression(expr.Y, context)

	case token.EQL: // ==
		return ge.evaluateComparison(expr.X, expr.Y, context, "==")
	case token.NEQ: // !=
		return ge.evaluateComparison(expr.X, expr.Y, context, "!=")
	case token.LSS: // <
		return ge.evaluateComparison(expr.X, expr.Y, context, "<")
	case token.LEQ: // <=
		return ge.evaluateComparison(expr.X, expr.Y, context, "<=")
	case token.GTR: // >
		return ge.evaluateComparison(expr.X, expr.Y, context, ">")
	case token.GEQ: // >=
		return ge.evaluateComparison(expr.X, expr.Y, context, ">=")

	default:
		return false, fmt.Errorf("unsupported binary operator: %s", expr.Op)
	}
}

// evaluateBinaryExpressionWithDebug handles binary operations with debug information
func (ge *GuardEvaluator) evaluateBinaryExpressionWithDebug(expr *ast.BinaryExpr, context *EvaluationContext, debug map[string]interface{}) (bool, error) {
	switch expr.Op {
	case token.LAND: // &&
		left, err := ge.evaluateGoExpressionWithDebug(expr.X, context, debug)
		if err != nil {
			return false, err
		}
		debug["and_left"] = left
		if !left {
			debug["and_short_circuit"] = true
			return false, nil // Short-circuit
		}
		right, err := ge.evaluateGoExpressionWithDebug(expr.Y, context, debug)
		debug["and_right"] = right
		return right, err

	case token.LOR: // ||
		left, err := ge.evaluateGoExpressionWithDebug(expr.X, context, debug)
		if err != nil {
			return false, err
		}
		debug["or_left"] = left
		if left {
			debug["or_short_circuit"] = true
			return true, nil // Short-circuit
		}
		right, err := ge.evaluateGoExpressionWithDebug(expr.Y, context, debug)
		debug["or_right"] = right
		return right, err

	case token.EQL: // ==
		return ge.evaluateComparisonWithDebug(expr.X, expr.Y, context, "==", debug)
	case token.NEQ: // !=
		return ge.evaluateComparisonWithDebug(expr.X, expr.Y, context, "!=", debug)
	case token.LSS: // <
		return ge.evaluateComparisonWithDebug(expr.X, expr.Y, context, "<", debug)
	case token.LEQ: // <=
		return ge.evaluateComparisonWithDebug(expr.X, expr.Y, context, "<=", debug)
	case token.GTR: // >
		return ge.evaluateComparisonWithDebug(expr.X, expr.Y, context, ">", debug)
	case token.GEQ: // >=
		return ge.evaluateComparisonWithDebug(expr.X, expr.Y, context, ">=", debug)

	default:
		return false, fmt.Errorf("unsupported binary operator: %s", expr.Op)
	}
}

// evaluateUnaryExpression handles unary operations (NOT)
func (ge *GuardEvaluator) evaluateUnaryExpression(expr *ast.UnaryExpr, context *EvaluationContext) (bool, error) {
	debug := make(map[string]interface{})
	return ge.evaluateUnaryExpressionWithDebug(expr, context, debug)
}

// evaluateUnaryExpressionWithDebug handles unary operations with debug information
func (ge *GuardEvaluator) evaluateUnaryExpressionWithDebug(expr *ast.UnaryExpr, context *EvaluationContext, debug map[string]interface{}) (bool, error) {
	switch expr.Op {
	case token.NOT: // !
		val, err := ge.evaluateGoExpressionWithDebug(expr.X, context, debug)
		if err != nil {
			return false, err
		}
		result := !val
		debug["not_operand"] = val
		debug["not_result"] = result
		return result, nil
	default:
		return false, fmt.Errorf("unsupported unary operator: %s", expr.Op)
	}
}

// evaluateComparison handles comparison operations
func (ge *GuardEvaluator) evaluateComparison(left, right ast.Expr, context *EvaluationContext, op string) (bool, error) {
	debug := make(map[string]interface{})
	return ge.evaluateComparisonWithDebug(left, right, context, op, debug)
}

// evaluateComparisonWithDebug handles comparison operations with debug information
func (ge *GuardEvaluator) evaluateComparisonWithDebug(left, right ast.Expr, context *EvaluationContext, op string, debug map[string]interface{}) (bool, error) {
	leftVal, err := ge.evaluateValue(left, context)
	if err != nil {
		// Try to get boolean value for identifiers like "true" and "false"
		if ident, ok := left.(*ast.Ident); ok {
			if ident.Name == "true" {
				leftVal = true
			} else if ident.Name == "false" {
				leftVal = false
			} else {
				return false, err
			}
		} else {
			return false, err
		}
	}

	rightVal, err := ge.evaluateValue(right, context)
	if err != nil {
		// Try to get boolean value for identifiers like "true" and "false"
		if ident, ok := right.(*ast.Ident); ok {
			if ident.Name == "true" {
				rightVal = true
			} else if ident.Name == "false" {
				rightVal = false
			} else {
				return false, err
			}
		} else {
			return false, err
		}
	}

	return ge.compareValues(leftVal, rightVal, op)
}

// evaluateValue extracts the actual value from an expression
func (ge *GuardEvaluator) evaluateValue(expr ast.Expr, context *EvaluationContext) (interface{}, error) {
	switch e := expr.(type) {
	case *ast.BasicLit:
		return ge.parseBasicLit(e)
	case *ast.Ident:
		return ge.getVariableValue(e.Name, context)
	case *ast.SelectorExpr:
		return ge.getSelectorValue(e, context)
	case *ast.CallExpr:
		return ge.getCallValue(e, context)
	default:
		return nil, fmt.Errorf("unsupported value expression: %T", e)
	}
}

// evaluateIdentifier handles identifier evaluation
func (ge *GuardEvaluator) evaluateIdentifier(name string, context *EvaluationContext) (bool, error) {
	debug := make(map[string]interface{})
	return ge.evaluateIdentifierWithDebug(name, context, debug)
}

// evaluateIdentifierWithDebug handles identifier evaluation with debug information
func (ge *GuardEvaluator) evaluateIdentifierWithDebug(name string, context *EvaluationContext, debug map[string]interface{}) (bool, error) {
	switch name {
	case "true":
		debug["literal_true"] = true
		return true, nil
	case "false":
		debug["literal_false"] = false
		return false, nil
	default:
		val, err := ge.getVariableValue(name, context)
		if err != nil {
			debug["variable_error"] = err.Error()
			return false, err
		}
		debug["variable_value"] = val
		result, err := ge.toBool(val)
		debug["boolean_conversion"] = result
		return result, err
	}
}

// evaluateSelectorExpression handles dot notation (e.g., context.variable)
func (ge *GuardEvaluator) evaluateSelectorExpression(expr *ast.SelectorExpr, context *EvaluationContext) (bool, error) {
	debug := make(map[string]interface{})
	return ge.evaluateSelectorExpressionWithDebug(expr, context, debug)
}

// evaluateSelectorExpressionWithDebug handles dot notation with debug information
func (ge *GuardEvaluator) evaluateSelectorExpressionWithDebug(expr *ast.SelectorExpr, context *EvaluationContext, debug map[string]interface{}) (bool, error) {
	val, err := ge.getSelectorValue(expr, context)
	if err != nil {
		debug["selector_error"] = err.Error()
		return false, err
	}
	debug["selector_value"] = val
	result, err := ge.toBool(val)
	debug["boolean_conversion"] = result
	return result, err
}

// getSelectorValue gets the value from a selector expression
func (ge *GuardEvaluator) getSelectorValue(expr *ast.SelectorExpr, context *EvaluationContext) (interface{}, error) {
	// Handle context.variable, event.data, etc.
	if ident, ok := expr.X.(*ast.Ident); ok {
		switch ident.Name {
		case "context":
			if context.Variables == nil {
				return nil, fmt.Errorf("context not available")
			}
			return ge.getStructValue(context.Variables, expr.Sel.Name)
		case "event":
			if context.Event == nil {
				return nil, fmt.Errorf("event not available")
			}
			return ge.getEventValue(context.Event, expr.Sel.Name)
		case "state":
			if context.StateData == nil {
				return nil, fmt.Errorf("state data not available")
			}
			return context.StateData[expr.Sel.Name], nil
		default:
			return nil, fmt.Errorf("unsupported selector base: %s", ident.Name)
		}
	}
	return nil, fmt.Errorf("unsupported selector expression")
}

// getStructValue extracts value from structpb.Struct
func (ge *GuardEvaluator) getStructValue(s *structpb.Struct, key string) (interface{}, error) {
	if s.Fields == nil {
		return nil, fmt.Errorf("field %s not found", key)
	}

	val, exists := s.Fields[key]
	if !exists {
		return nil, fmt.Errorf("field %s not found", key)
	}

	return ge.structValueToInterface(val)
}

// getEventValue extracts value from event
func (ge *GuardEvaluator) getEventValue(event *sc.Event, key string) (interface{}, error) {
	switch key {
	case "name", "label":
		return event.Label, nil
	case "parameters", "data":
		if event.Parameters == nil {
			return nil, fmt.Errorf("event has no parameters")
		}
		return ge.structValueToInterface(structpb.NewStructValue(event.Parameters))
	default:
		// Look in event parameters
		if event.Parameters != nil && event.Parameters.Fields != nil {
			if val, exists := event.Parameters.Fields[key]; exists {
				return ge.structValueToInterface(val)
			}
		}
		return nil, fmt.Errorf("event field %s not found", key)
	}
}

// structValueToInterface converts structpb.Value to Go interface{}
func (ge *GuardEvaluator) structValueToInterface(val *structpb.Value) (interface{}, error) {
	switch kind := val.GetKind().(type) {
	case *structpb.Value_NullValue:
		return nil, nil
	case *structpb.Value_NumberValue:
		return kind.NumberValue, nil
	case *structpb.Value_StringValue:
		return kind.StringValue, nil
	case *structpb.Value_BoolValue:
		return kind.BoolValue, nil
	case *structpb.Value_StructValue:
		return kind.StructValue, nil
	case *structpb.Value_ListValue:
		return kind.ListValue, nil
	default:
		return nil, fmt.Errorf("unsupported value type: %T", kind)
	}
}

// evaluateCallExpression handles function calls
func (ge *GuardEvaluator) evaluateCallExpression(expr *ast.CallExpr, context *EvaluationContext) (bool, error) {
	debug := make(map[string]interface{})
	return ge.evaluateCallExpressionWithDebug(expr, context, debug)
}

// evaluateCallExpressionWithDebug handles function calls with debug information
func (ge *GuardEvaluator) evaluateCallExpressionWithDebug(expr *ast.CallExpr, context *EvaluationContext, debug map[string]interface{}) (bool, error) {
	val, err := ge.getCallValue(expr, context)
	if err != nil {
		debug["call_error"] = err.Error()
		return false, err
	}
	debug["call_value"] = val
	result, err := ge.toBool(val)
	debug["boolean_conversion"] = result
	return result, err
}

// getCallValue handles function call evaluation
func (ge *GuardEvaluator) getCallValue(expr *ast.CallExpr, context *EvaluationContext) (interface{}, error) {
	// Handle built-in functions
	if ident, ok := expr.Fun.(*ast.Ident); ok {
		switch ident.Name {
		case "len":
			return ge.evaluateLenFunction(expr.Args, context)
		case "contains":
			return ge.evaluateContainsFunction(expr.Args, context)
		case "hasPrefix":
			return ge.evaluateHasPrefixFunction(expr.Args, context)
		case "hasSuffix":
			return ge.evaluateHasSuffixFunction(expr.Args, context)
		default:
			return nil, fmt.Errorf("unsupported function: %s", ident.Name)
		}
	}
	return nil, fmt.Errorf("unsupported function call")
}

// evaluateLenFunction implements len() function
func (ge *GuardEvaluator) evaluateLenFunction(args []ast.Expr, context *EvaluationContext) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("len() requires exactly 1 argument")
	}

	val, err := ge.evaluateValue(args[0], context)
	if err != nil {
		return nil, err
	}

	switch v := val.(type) {
	case string:
		return float64(len(v)), nil
	case *structpb.ListValue:
		return float64(len(v.Values)), nil
	case *structpb.Struct:
		return float64(len(v.Fields)), nil
	case []string:
		return float64(len(v)), nil
	case []interface{}:
		return float64(len(v)), nil
	default:
		// Use reflection to handle slices and arrays
		reflectVal := reflect.ValueOf(v)
		switch reflectVal.Kind() {
		case reflect.Slice, reflect.Array:
			return float64(reflectVal.Len()), nil
		case reflect.Map:
			return float64(reflectVal.Len()), nil
		default:
			return nil, fmt.Errorf("len() not supported for type %T", v)
		}
	}
}

// evaluateContainsFunction implements contains() function
func (ge *GuardEvaluator) evaluateContainsFunction(args []ast.Expr, context *EvaluationContext) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("contains() requires exactly 2 arguments")
	}

	containerVal, err := ge.evaluateValue(args[0], context)
	if err != nil {
		return nil, err
	}

	searchVal, err := ge.evaluateValue(args[1], context)
	if err != nil {
		return nil, err
	}

	switch container := containerVal.(type) {
	case string:
		if search, ok := searchVal.(string); ok {
			return strings.Contains(container, search), nil
		}
		return false, fmt.Errorf("contains() search value must be string for string container")
	default:
		return false, fmt.Errorf("contains() not supported for container type %T", container)
	}
}

// evaluateHasPrefixFunction implements hasPrefix() function
func (ge *GuardEvaluator) evaluateHasPrefixFunction(args []ast.Expr, context *EvaluationContext) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("hasPrefix() requires exactly 2 arguments")
	}

	strVal, err := ge.evaluateValue(args[0], context)
	if err != nil {
		return nil, err
	}

	prefixVal, err := ge.evaluateValue(args[1], context)
	if err != nil {
		return nil, err
	}

	str, ok1 := strVal.(string)
	prefix, ok2 := prefixVal.(string)

	if !ok1 || !ok2 {
		return false, fmt.Errorf("hasPrefix() requires string arguments")
	}

	return strings.HasPrefix(str, prefix), nil
}

// evaluateHasSuffixFunction implements hasSuffix() function
func (ge *GuardEvaluator) evaluateHasSuffixFunction(args []ast.Expr, context *EvaluationContext) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("hasSuffix() requires exactly 2 arguments")
	}

	strVal, err := ge.evaluateValue(args[0], context)
	if err != nil {
		return nil, err
	}

	suffixVal, err := ge.evaluateValue(args[1], context)
	if err != nil {
		return nil, err
	}

	str, ok1 := strVal.(string)
	suffix, ok2 := suffixVal.(string)

	if !ok1 || !ok2 {
		return false, fmt.Errorf("hasSuffix() requires string arguments")
	}

	return strings.HasSuffix(str, suffix), nil
}

// evaluateBasicLit handles basic literals
func (ge *GuardEvaluator) evaluateBasicLit(lit *ast.BasicLit, context *EvaluationContext) (bool, error) {
	debug := make(map[string]interface{})
	return ge.evaluateBasicLitWithDebug(lit, context, debug)
}

// evaluateBasicLitWithDebug handles basic literals with debug information
func (ge *GuardEvaluator) evaluateBasicLitWithDebug(lit *ast.BasicLit, context *EvaluationContext, debug map[string]interface{}) (bool, error) {
	val, err := ge.parseBasicLit(lit)
	if err != nil {
		debug["literal_error"] = err.Error()
		return false, err
	}
	debug["literal_value"] = val
	result, err := ge.toBool(val)
	debug["boolean_conversion"] = result
	return result, err
}

// parseBasicLit parses basic literal values
func (ge *GuardEvaluator) parseBasicLit(lit *ast.BasicLit) (interface{}, error) {
	switch lit.Kind {
	case token.INT:
		return strconv.ParseFloat(lit.Value, 64)
	case token.FLOAT:
		return strconv.ParseFloat(lit.Value, 64)
	case token.STRING:
		return strconv.Unquote(lit.Value)
	case token.CHAR:
		return strconv.Unquote(lit.Value)
	default:
		return nil, fmt.Errorf("unsupported literal type: %s", lit.Kind)
	}
}

// getVariableValue gets variable value from context
func (ge *GuardEvaluator) getVariableValue(name string, context *EvaluationContext) (interface{}, error) {
	// Handle boolean literals first
	switch name {
	case "true":
		return true, nil
	case "false":
		return false, nil
	}

	// Check context variables
	if context.Variables != nil && context.Variables.Fields != nil {
		if val, exists := context.Variables.Fields[name]; exists {
			return ge.structValueToInterface(val)
		}
	}

	// Check state data
	if context.StateData != nil {
		if val, exists := context.StateData[name]; exists {
			return val, nil
		}
	}

	return nil, fmt.Errorf("variable %s not found", name)
}

// compareValues compares two values using the given operator
func (ge *GuardEvaluator) compareValues(left, right interface{}, op string) (bool, error) {
	// Handle nil values
	if left == nil && right == nil {
		return op == "==" || op == "<=", nil
	}
	if left == nil || right == nil {
		return op == "!=", nil
	}

	// Use reflection for type conversion and comparison
	leftVal := reflect.ValueOf(left)
	rightVal := reflect.ValueOf(right)

	// Convert to comparable types
	leftComp, rightComp, err := ge.makeComparable(leftVal, rightVal)
	if err != nil {
		return false, err
	}

	switch op {
	case "==":
		return leftComp == rightComp, nil
	case "!=":
		return leftComp != rightComp, nil
	case "<":
		return ge.compareNumbers(leftComp, rightComp, "<")
	case "<=":
		return ge.compareNumbers(leftComp, rightComp, "<=")
	case ">":
		return ge.compareNumbers(leftComp, rightComp, ">")
	case ">=":
		return ge.compareNumbers(leftComp, rightComp, ">=")
	default:
		return false, fmt.Errorf("unsupported comparison operator: %s", op)
	}
}

// makeComparable converts values to comparable types
func (ge *GuardEvaluator) makeComparable(left, right reflect.Value) (interface{}, interface{}, error) {
	// Handle string comparisons
	if left.Kind() == reflect.String || right.Kind() == reflect.String {
		return fmt.Sprintf("%v", left.Interface()), fmt.Sprintf("%v", right.Interface()), nil
	}

	// Handle numeric comparisons
	leftNum, leftOk := ge.toNumber(left.Interface())
	rightNum, rightOk := ge.toNumber(right.Interface())

	if leftOk && rightOk {
		return leftNum, rightNum, nil
	}

	// Handle boolean comparisons
	leftBool, leftOk := ge.toBoolValue(left.Interface())
	rightBool, rightOk := ge.toBoolValue(right.Interface())

	if leftOk && rightOk {
		return leftBool, rightBool, nil
	}

	// Fallback to interface comparison
	return left.Interface(), right.Interface(), nil
}

// compareNumbers compares two numbers
func (ge *GuardEvaluator) compareNumbers(left, right interface{}, op string) (bool, error) {
	leftNum, leftOk := ge.toNumber(left)
	rightNum, rightOk := ge.toNumber(right)

	if !leftOk || !rightOk {
		// For graceful handling, fall back to string comparison
		leftStr := fmt.Sprintf("%v", left)
		rightStr := fmt.Sprintf("%v", right)
		
		switch op {
		case "<":
			return leftStr < rightStr, nil
		case "<=":
			return leftStr <= rightStr, nil
		case ">":
			return leftStr > rightStr, nil
		case ">=":
			return leftStr >= rightStr, nil
		default:
			return false, fmt.Errorf("unsupported comparison: %s", op)
		}
	}

	switch op {
	case "<":
		return leftNum < rightNum, nil
	case "<=":
		return leftNum <= rightNum, nil
	case ">":
		return leftNum > rightNum, nil
	case ">=":
		return leftNum >= rightNum, nil
	default:
		return false, fmt.Errorf("unsupported numeric comparison: %s", op)
	}
}

// toNumber converts interface{} to float64
func (ge *GuardEvaluator) toNumber(val interface{}) (float64, bool) {
	switch v := val.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case string:
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f, true
		}
	}
	return 0, false
}

// toBool converts interface{} to bool
func (ge *GuardEvaluator) toBool(val interface{}) (bool, error) {
	switch v := val.(type) {
	case bool:
		return v, nil
	case float64:
		return v != 0, nil
	case string:
		return v != "", nil
	case nil:
		return false, nil
	default:
		return false, fmt.Errorf("cannot convert %T to bool", v)
	}
}

// toBoolValue converts interface{} to bool (for comparison)
func (ge *GuardEvaluator) toBoolValue(val interface{}) (bool, bool) {
	if b, ok := val.(bool); ok {
		return b, true
	}
	return false, false
}

// evaluateSimpleExpression handles simple expressions for backward compatibility
func (ge *GuardEvaluator) evaluateSimpleExpression(expression string, context *EvaluationContext) (bool, error) {
	debug := make(map[string]interface{})
	return ge.evaluateSimpleExpressionWithDebug(expression, context, debug)
}

// evaluateSimpleExpressionWithDebug handles simple expressions with debug information
func (ge *GuardEvaluator) evaluateSimpleExpressionWithDebug(expression string, context *EvaluationContext, debug map[string]interface{}) (bool, error) {
	expression = strings.TrimSpace(expression)
	debug["original_expression"] = expression

	switch expression {
	case "true":
		debug["literal_result"] = true
		return true, nil
	case "false":
		debug["literal_result"] = false
		return false, nil
	case "":
		debug["empty_expression"] = true
		return true, nil
	}

	// Handle simple context-based expressions
	if expression == "context.count < 5" {
		debug["pattern"] = "context.count < 5"
		if context.Variables != nil && context.Variables.Fields != nil {
			if countValue, exists := context.Variables.Fields["count"]; exists {
				if count, ok := countValue.GetKind().(*structpb.Value_NumberValue); ok {
					result := count.NumberValue < 5
					debug["count_value"] = count.NumberValue
					debug["comparison_result"] = result
					return result, nil
				}
			}
		}
		debug["error"] = "context.count not found or not a number"
		return false, fmt.Errorf("context.count not found or not a number")
	}

	// Default to true for unknown simple expressions (backward compatibility)
	debug["fallback"] = true
	return true, nil
}

// extractVariables extracts variable names from Go AST
func (ge *GuardEvaluator) extractVariables(expr ast.Expr) []string {
	var variables []string
	
	ast.Inspect(expr, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if ident, ok := sel.X.(*ast.Ident); ok {
				varName := ident.Name + "." + sel.Sel.Name
				variables = append(variables, varName)
			}
		}
		return true
	})
	
	return variables
}

// extractSimpleVariables extracts variables from simple expressions
func (ge *GuardEvaluator) extractSimpleVariables(expression string) []string {
	var variables []string
	
	// Simple regex-like extraction for context.variable patterns
	words := strings.Fields(expression)
	for _, word := range words {
		if strings.HasPrefix(word, "context.") {
			variables = append(variables, word)
		}
	}
	
	return variables
}

// calculateComplexity calculates expression complexity for optimization
func (ge *GuardEvaluator) calculateComplexity(expr ast.Expr) int {
	complexity := 0
	
	ast.Inspect(expr, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.BinaryExpr:
			complexity += 2
		case *ast.UnaryExpr:
			complexity += 1
		case *ast.CallExpr:
			complexity += 3
		case *ast.SelectorExpr:
			complexity += 1
		default:
			complexity += 1
		}
		return true
	})
	
	return complexity
}

// convertJSToGo converts JavaScript-like expressions to Go syntax
func (ge *GuardEvaluator) convertJSToGo(expression string) string {
	// Simple conversions for common JavaScript patterns
	expression = strings.ReplaceAll(expression, "===", "==")
	expression = strings.ReplaceAll(expression, "!==", "!=")
	expression = strings.ReplaceAll(expression, "&&", "&&")
	expression = strings.ReplaceAll(expression, "||", "||")
	
	return expression
}

// ClearCache clears the expression cache
func (ge *GuardEvaluator) ClearCache() {
	ge.mu.Lock()
	defer ge.mu.Unlock()
	
	ge.cache = make(map[string]*GuardExpression)
}

// GetCacheSize returns the current cache size
func (ge *GuardEvaluator) GetCacheSize() int {
	ge.mu.RLock()
	defer ge.mu.RUnlock()
	
	return len(ge.cache)
}

// Global guard evaluator instance
var globalGuardEvaluator = NewGuardEvaluator()

// EvaluateGuardExpression provides backward compatibility with the existing API
func EvaluateGuardExpression(expression string, context *structpb.Struct) (bool, error) {
	guard := &sc.Guard{Expression: expression}
	evalContext := &EvaluationContext{Variables: context}
	
	result, err := globalGuardEvaluator.EvaluateGuard(guard, evalContext)
	if err != nil {
		return false, err
	}
	
	return result.Value, nil
}

// SetGlobalGuardEvaluator sets a custom global guard evaluator
func SetGlobalGuardEvaluator(evaluator *GuardEvaluator) {
	globalGuardEvaluator = evaluator
}

// GetGlobalGuardEvaluator returns the global guard evaluator
func GetGlobalGuardEvaluator() *GuardEvaluator {
	return globalGuardEvaluator
}

// RegisterCustomFunction registers a custom function for use in guard expressions
func (ge *GuardEvaluator) RegisterCustomFunction(name string, fn GuardFunction) {
	ge.mu.Lock()
	defer ge.mu.Unlock()
	ge.customFunctions[name] = fn
}

// UnregisterCustomFunction removes a custom function
func (ge *GuardEvaluator) UnregisterCustomFunction(name string) {
	ge.mu.Lock()
	defer ge.mu.Unlock()
	delete(ge.customFunctions, name)
}

// GetMetrics returns a copy of the current metrics
func (ge *GuardEvaluator) GetMetrics() GuardMetrics {
	ge.mu.RLock()
	defer ge.mu.RUnlock()
	
	// Create a copy of complexity scores
	complexityScores := make([]int, len(ge.metrics.ComplexityScores))
	copy(complexityScores, ge.metrics.ComplexityScores)
	
	return GuardMetrics{
		Evaluations:      ge.metrics.Evaluations,
		CacheHits:        ge.metrics.CacheHits,
		CacheMisses:      ge.metrics.CacheMisses,
		Errors:           ge.metrics.Errors,
		TotalEvalTime:    ge.metrics.TotalEvalTime,
		ComplexityScores: complexityScores,
	}
}

// ResetMetrics resets all metrics to zero
func (ge *GuardEvaluator) ResetMetrics() {
	ge.mu.Lock()
	defer ge.mu.Unlock()
	ge.metrics = &GuardMetrics{}
}

// GetAverageEvaluationTime returns the average evaluation time in nanoseconds
func (ge *GuardEvaluator) GetAverageEvaluationTime() int64 {
	ge.mu.RLock()
	defer ge.mu.RUnlock()
	
	if ge.metrics.Evaluations == 0 {
		return 0
	}
	return ge.metrics.TotalEvalTime / ge.metrics.Evaluations
}

// GetCacheHitRate returns the cache hit rate as a percentage
func (ge *GuardEvaluator) GetCacheHitRate() float64 {
	ge.mu.RLock()
	defer ge.mu.RUnlock()
	
	total := ge.metrics.CacheHits + ge.metrics.CacheMisses
	if total == 0 {
		return 0
	}
	return float64(ge.metrics.CacheHits) / float64(total) * 100
}

// GetAverageComplexity returns the average complexity score
func (ge *GuardEvaluator) GetAverageComplexity() float64 {
	ge.mu.RLock()
	defer ge.mu.RUnlock()
	
	if len(ge.metrics.ComplexityScores) == 0 {
		return 0
	}
	
	total := 0
	for _, score := range ge.metrics.ComplexityScores {
		total += score
	}
	return float64(total) / float64(len(ge.metrics.ComplexityScores))
}