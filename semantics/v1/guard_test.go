package semantics

import (
	"testing"

	"github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestGuardEvaluatorBasicExpressions(t *testing.T) {
	evaluator := NewGuardEvaluator()

	tests := []struct {
		name       string
		expression string
		context    *EvaluationContext
		expected   bool
		shouldErr  bool
	}{
		{
			name:       "Always true guard",
			expression: "true",
			context:    &EvaluationContext{},
			expected:   true,
			shouldErr:  false,
		},
		{
			name:       "Always false guard",
			expression: "false",
			context:    &EvaluationContext{},
			expected:   false,
			shouldErr:  false,
		},
		{
			name:       "Empty expression defaults to true",
			expression: "",
			context:    &EvaluationContext{},
			expected:   true,
			shouldErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			guard := &sc.Guard{Expression: tt.expression}
			result, err := evaluator.EvaluateGuard(guard, tt.context)

			if tt.shouldErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result.Value != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result.Value)
			}
		})
	}
}

func TestGuardEvaluatorBooleanLogic(t *testing.T) {
	evaluator := NewGuardEvaluator()

	tests := []struct {
		name       string
		expression string
		expected   bool
	}{
		{
			name:       "AND true && true",
			expression: "true && true",
			expected:   true,
		},
		{
			name:       "AND true && false",
			expression: "true && false",
			expected:   false,
		},
		{
			name:       "OR false || true",
			expression: "false || true",
			expected:   true,
		},
		{
			name:       "OR false || false",
			expression: "false || false",
			expected:   false,
		},
		{
			name:       "NOT !true",
			expression: "!true",
			expected:   false,
		},
		{
			name:       "NOT !false",
			expression: "!false",
			expected:   true,
		},
		{
			name:       "Complex expression",
			expression: "(true && false) || (!false && true)",
			expected:   true,
		},
		{
			name:       "Nested parentheses",
			expression: "!((false || false) && true)",
			expected:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			guard := &sc.Guard{Expression: tt.expression}
			context := &EvaluationContext{}
			result, err := evaluator.EvaluateGuard(guard, context)

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result.Value != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result.Value)
			}
		})
	}
}

func TestGuardEvaluatorNumericComparisons(t *testing.T) {
	evaluator := NewGuardEvaluator()

	contextVars := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count":  structpb.NewNumberValue(10),
			"score":  structpb.NewNumberValue(85.5),
			"zero":   structpb.NewNumberValue(0),
			"name":   structpb.NewStringValue("test"),
			"active": structpb.NewBoolValue(true),
		},
	}

	tests := []struct {
		name       string
		expression string
		expected   bool
		shouldErr  bool
	}{
		{
			name:       "Less than - true",
			expression: "context.count < 15",
			expected:   true,
		},
		{
			name:       "Less than - false",
			expression: "context.count < 5",
			expected:   false,
		},
		{
			name:       "Greater than - true",
			expression: "context.score > 80",
			expected:   true,
		},
		{
			name:       "Greater than - false",
			expression: "context.score > 90",
			expected:   false,
		},
		{
			name:       "Equal to - true",
			expression: "context.count == 10",
			expected:   true,
		},
		{
			name:       "Equal to - false",
			expression: "context.count == 5",
			expected:   false,
		},
		{
			name:       "Not equal to - true",
			expression: "context.count != 5",
			expected:   true,
		},
		{
			name:       "Not equal to - false",
			expression: "context.count != 10",
			expected:   false,
		},
		{
			name:       "Less than or equal - true",
			expression: "context.count <= 10",
			expected:   true,
		},
		{
			name:       "Greater than or equal - true",
			expression: "context.score >= 85.5",
			expected:   true,
		},
		{
			name:       "Complex numeric expression",
			expression: "context.count > 5 && context.score < 90",
			expected:   true,
		},
		{
			name:       "Zero comparison",
			expression: "context.zero == 0",
			expected:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			guard := &sc.Guard{Expression: tt.expression}
			context := &EvaluationContext{Variables: contextVars}
			result, err := evaluator.EvaluateGuard(guard, context)

			if tt.shouldErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result.Value != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result.Value)
			}
		})
	}
}

func TestGuardEvaluatorStringComparisons(t *testing.T) {
	evaluator := NewGuardEvaluator()

	contextVars := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"name":     structpb.NewStringValue("hello"),
			"status":   structpb.NewStringValue("active"),
			"category": structpb.NewStringValue("test"),
		},
	}

	tests := []struct {
		name       string
		expression string
		expected   bool
	}{
		{
			name:       "String equality - true",
			expression: `context.name == "hello"`,
			expected:   true,
		},
		{
			name:       "String equality - false",
			expression: `context.name == "world"`,
			expected:   false,
		},
		{
			name:       "String inequality - true",
			expression: `context.status != "inactive"`,
			expected:   true,
		},
		{
			name:       "String inequality - false",
			expression: `context.status != "active"`,
			expected:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			guard := &sc.Guard{Expression: tt.expression}
			context := &EvaluationContext{Variables: contextVars}
			result, err := evaluator.EvaluateGuard(guard, context)

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result.Value != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result.Value)
			}
		})
	}
}

func TestGuardEvaluatorEventAccess(t *testing.T) {
	evaluator := NewGuardEvaluator()

	event := &sc.Event{
		Label: "USER_INPUT",
		Parameters: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"userId": structpb.NewNumberValue(123),
				"action": structpb.NewStringValue("click"),
				"valid":  structpb.NewBoolValue(true),
			},
		},
	}

	tests := []struct {
		name       string
		expression string
		expected   bool
	}{
		{
			name:       "Event name access",
			expression: `event.name == "USER_INPUT"`,
			expected:   true,
		},
		{
			name:       "Event parameter access - number",
			expression: "event.userId == 123",
			expected:   true,
		},
		{
			name:       "Event parameter access - string",
			expression: `event.action == "click"`,
			expected:   true,
		},
		{
			name:       "Event parameter access - boolean",
			expression: "event.valid == true",
			expected:   true,
		},
		{
			name:       "Combined event and context",
			expression: `event.action == "click" && event.userId > 100`,
			expected:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			guard := &sc.Guard{Expression: tt.expression}
			context := &EvaluationContext{Event: event}
			result, err := evaluator.EvaluateGuard(guard, context)

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result.Value != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result.Value)
			}
		})
	}
}

func TestGuardEvaluatorFunctions(t *testing.T) {
	evaluator := NewGuardEvaluator()

	contextVars := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"message": structpb.NewStringValue("hello world"),
			"tags": structpb.NewListValue(&structpb.ListValue{
				Values: []*structpb.Value{
					structpb.NewStringValue("tag1"),
					structpb.NewStringValue("tag2"),
				},
			}),
		},
	}

	tests := []struct {
		name       string
		expression string
		expected   bool
		shouldErr  bool
	}{
		{
			name:       "String length function",
			expression: "len(context.message) > 5",
			expected:   true,
		},
		{
			name:       "String contains function",
			expression: `contains(context.message, "world")`,
			expected:   true,
		},
		{
			name:       "String contains function - false",
			expression: `contains(context.message, "test")`,
			expected:   false,
		},
		{
			name:       "String hasPrefix function",
			expression: `hasPrefix(context.message, "hello")`,
			expected:   true,
		},
		{
			name:       "String hasSuffix function",
			expression: `hasSuffix(context.message, "world")`,
			expected:   true,
		},
		{
			name:       "List length function",
			expression: "len(context.tags) == 2",
			expected:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			guard := &sc.Guard{Expression: tt.expression}
			context := &EvaluationContext{Variables: contextVars}
			result, err := evaluator.EvaluateGuard(guard, context)

			if tt.shouldErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result.Value != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result.Value)
			}
		})
	}
}

func TestGuardEvaluatorStateData(t *testing.T) {
	evaluator := NewGuardEvaluator()

	transition := &sc.Transition{
		Label: "test_transition",
		From:  []string{"StateA"},
		To:    []string{"StateB"},
	}

	tests := []struct {
		name       string
		expression string
		expected   bool
	}{
		{
			name:       "Transition label access",
			expression: `state.transitionLabel == "test_transition"`,
			expected:   true,
		},
		{
			name:       "Source state access",
			expression: "len(state.sourceStates) == 1",
			expected:   true,
		},
		{
			name:       "Target state access",
			expression: "len(state.targetStates) == 1",
			expected:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			guard := &sc.Guard{Expression: tt.expression}
			context := &EvaluationContext{
				StateData: map[string]interface{}{
					"transitionLabel": transition.Label,
					"sourceStates":    transition.From,
					"targetStates":    transition.To,
				},
				Transition: transition,
			}
			result, err := evaluator.EvaluateGuard(guard, context)

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result.Value != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result.Value)
			}
		})
	}
}

func TestGuardEvaluatorComplexExpressions(t *testing.T) {
	evaluator := NewGuardEvaluator()

	contextVars := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count":    structpb.NewNumberValue(15),
			"status":   structpb.NewStringValue("active"),
			"score":    structpb.NewNumberValue(85),
			"enabled":  structpb.NewBoolValue(true),
			"category": structpb.NewStringValue("premium"),
		},
	}

	event := &sc.Event{
		Label: "UPDATE",
		Parameters: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"priority": structpb.NewStringValue("high"),
				"amount":   structpb.NewNumberValue(100),
			},
		},
	}

	tests := []struct {
		name       string
		expression string
		expected   bool
	}{
		{
			name: "Complex AND/OR expression",
			expression: `(context.count > 10 && context.status == "active") || 
						 (context.score >= 80 && context.enabled == true)`,
			expected: true,
		},
		{
			name: "Nested function calls",
			expression: `contains(context.category, "prem") && 
						 len(context.status) > 3`,
			expected: true,
		},
		{
			name: "Event and context combination",
			expression: `event.priority == "high" && 
						 context.count > 10 && 
						 event.amount >= context.score`,
			expected: true,
		},
		{
			name: "Complex boolean logic with functions",
			expression: `(hasPrefix(context.status, "act") || context.enabled) && 
						 !(context.count < 5 || context.score < 50)`,
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			guard := &sc.Guard{Expression: tt.expression}
			context := &EvaluationContext{
				Variables: contextVars,
				Event:     event,
			}
			result, err := evaluator.EvaluateGuard(guard, context)

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result.Value != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result.Value)
			}
		})
	}
}

func TestGuardEvaluatorErrorHandling(t *testing.T) {
	evaluator := NewGuardEvaluator()

	tests := []struct {
		name       string
		expression string
		context    *EvaluationContext
		shouldErr  bool
	}{
		{
			name:       "Invalid syntax",
			expression: "context.count < < 5",
			context:    &EvaluationContext{},
			shouldErr:  true,
		},
		{
			name:       "Undefined variable",
			expression: "context.undefined > 5",
			context:    &EvaluationContext{},
			shouldErr:  true,
		},
		{
			name:       "Invalid function call",
			expression: "invalidFunction(context.count)",
			context:    &EvaluationContext{},
			shouldErr:  true,
		},
		{
			name:       "Type mismatch in comparison",
			expression: `context.count > "string"`,
			context: &EvaluationContext{
				Variables: &structpb.Struct{
					Fields: map[string]*structpb.Value{
						"count": structpb.NewNumberValue(5),
					},
				},
			},
			shouldErr: false, // Should handle gracefully
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			guard := &sc.Guard{Expression: tt.expression}
			result, err := evaluator.EvaluateGuard(guard, tt.context)

			if tt.shouldErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if result == nil {
					t.Error("Expected result but got nil")
				}
			}
		})
	}
}

func TestGuardEvaluatorCaching(t *testing.T) {
	evaluator := NewGuardEvaluator()

	guard := &sc.Guard{Expression: "context.count > 5"}
	context := &EvaluationContext{
		Variables: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"count": structpb.NewNumberValue(10),
			},
		},
	}

	// First evaluation - should parse and cache
	result1, err := evaluator.EvaluateGuard(guard, context)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Check cache size
	if evaluator.GetCacheSize() != 1 {
		t.Errorf("Expected cache size 1, got %d", evaluator.GetCacheSize())
	}

	// Second evaluation - should use cache
	result2, err := evaluator.EvaluateGuard(guard, context)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if result1.Value != result2.Value {
		t.Error("Cached result should match original result")
	}

	// Clear cache and verify
	evaluator.ClearCache()
	if evaluator.GetCacheSize() != 0 {
		t.Errorf("Expected cache size 0 after clear, got %d", evaluator.GetCacheSize())
	}
}

func TestGuardEvaluatorLanguageSupport(t *testing.T) {
	evaluator := NewGuardEvaluator()

	contextVars := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(10),
		},
	}

	tests := []struct {
		name       string
		guard      *sc.Guard
		expected   bool
		shouldErr  bool
	}{
		{
			name: "Go language",
			guard: &sc.Guard{
				Expression: "context.count > 5",
				Language:   "go",
			},
			expected: true,
		},
		{
			name: "JavaScript language (converted)",
			guard: &sc.Guard{
				Expression: "context.count > 5",
				Language:   "js",
			},
			expected: true,
		},
		{
			name: "Simple language",
			guard: &sc.Guard{
				Expression: "context.count < 5",
				Language:   "simple",
			},
			expected: false,
		},
		{
			name: "Unsupported language (falls back to simple)",
			guard: &sc.Guard{
				Expression: "true",
				Language:   "python",
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			context := &EvaluationContext{Variables: contextVars}
			result, err := evaluator.EvaluateGuard(tt.guard, context)

			if tt.shouldErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result.Value != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result.Value)
			}
		})
	}
}

func TestGuardEvaluatorPerformance(t *testing.T) {
	evaluator := NewGuardEvaluator()

	// Create a complex guard expression
	guard := &sc.Guard{
		Expression: `context.count > 10 && (context.status == "active" || context.enabled == true) && 
					 len(context.name) > 3 && contains(context.category, "test")`,
	}

	contextVars := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count":    structpb.NewNumberValue(15),
			"status":   structpb.NewStringValue("active"),
			"enabled":  structpb.NewBoolValue(true),
			"name":     structpb.NewStringValue("test_user"),
			"category": structpb.NewStringValue("test_category"),
		},
	}

	context := &EvaluationContext{Variables: contextVars}

	// Benchmark multiple evaluations
	iterations := 1000
	for i := 0; i < iterations; i++ {
		result, err := evaluator.EvaluateGuard(guard, context)
		if err != nil {
			t.Fatalf("Unexpected error on iteration %d: %v", i, err)
		}
		if !result.Value {
			t.Errorf("Expected true result on iteration %d", i)
		}
	}

	// Verify caching is working
	if evaluator.GetCacheSize() != 1 {
		t.Errorf("Expected 1 cached expression after %d iterations, got %d", 
			iterations, evaluator.GetCacheSize())
	}
}

func TestGuardBackwardCompatibility(t *testing.T) {
	// Test backward compatibility with the old EvaluateGuardExpression function
	contextVars := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(3),
		},
	}

	tests := []struct {
		name       string
		expression string
		expected   bool
	}{
		{
			name:       "Simple true",
			expression: "true",
			expected:   true,
		},
		{
			name:       "Simple false",
			expression: "false",
			expected:   false,
		},
		{
			name:       "Context count less than 5",
			expression: "context.count < 5",
			expected:   true,
		},
		{
			name:       "Empty expression",
			expression: "",
			expected:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := EvaluateGuardExpression(tt.expression, contextVars)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}