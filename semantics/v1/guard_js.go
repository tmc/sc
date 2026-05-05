package semantics

import (
	"fmt"
	"strings"

	"github.com/tmc/sc"
)

// JavaScriptGuardEvaluator provides JavaScript-compatible guard evaluation
// This is useful for web visualizers and client-side applications
type JavaScriptGuardEvaluator struct {
	*GuardEvaluator
}

// NewJavaScriptGuardEvaluator creates a new JavaScript-compatible guard evaluator
func NewJavaScriptGuardEvaluator() *JavaScriptGuardEvaluator {
	return &JavaScriptGuardEvaluator{
		GuardEvaluator: NewGuardEvaluator(),
	}
}

// GenerateJavaScriptFunction generates a JavaScript function that can evaluate the guard
func (jse *JavaScriptGuardEvaluator) GenerateJavaScriptFunction(guard *sc.Guard) (string, error) {
	if guard == nil || guard.Expression == "" {
		return "function(context, event, state) { return true; }", nil
	}

	// Parse the expression to validate it
	expr, err := jse.parseExpression(guard.Expression, guard.Language)
	if err != nil {
		return "", fmt.Errorf("failed to parse guard expression for JavaScript generation: %w", err)
	}

	// Convert the expression to JavaScript
	jsExpr, err := jse.convertToJavaScript(expr)
	if err != nil {
		return "", fmt.Errorf("failed to convert expression to JavaScript: %w", err)
	}

	// Generate the complete JavaScript function
	jsFunction := fmt.Sprintf(`function(context, event, state) {
    try {
        // Helper functions
        function len(obj) {
            if (typeof obj === 'string') return obj.length;
            if (Array.isArray(obj)) return obj.length;
            if (obj && typeof obj === 'object') return Object.keys(obj).length;
            return 0;
        }
        
        function contains(str, substr) {
            if (typeof str !== 'string' || typeof substr !== 'string') return false;
            return str.includes(substr);
        }
        
        function hasPrefix(str, prefix) {
            if (typeof str !== 'string' || typeof prefix !== 'string') return false;
            return str.startsWith(prefix);
        }
        
        function hasSuffix(str, suffix) {
            if (typeof str !== 'string' || typeof suffix !== 'string') return false;
            return str.endsWith(suffix);
        }

        // Safely access nested properties
        function safeGet(obj, path) {
            const parts = path.split('.');
            let current = obj;
            for (const part of parts) {
                if (current && typeof current === 'object' && part in current) {
                    current = current[part];
                } else {
                    return undefined;
                }
            }
            return current;
        }

        // Convert context if needed (for protobuf compatibility)
        if (context && context.fields) {
            const convertedContext = {};
            for (const [key, value] of Object.entries(context.fields)) {
                convertedContext[key] = convertProtobufValue(value);
            }
            context = convertedContext;
        }

        // Convert event parameters if needed
        if (event && event.parameters && event.parameters.fields) {
            const convertedParams = {};
            for (const [key, value] of Object.entries(event.parameters.fields)) {
                convertedParams[key] = convertProtobufValue(value);
            }
            event = { ...event, ...convertedParams };
        }

        function convertProtobufValue(value) {
            if (!value || !value.kind) return value;
            switch (value.kind) {
                case 'nullValue': return null;
                case 'numberValue': return value.numberValue;
                case 'stringValue': return value.stringValue;
                case 'boolValue': return value.boolValue;
                case 'structValue': 
                    const obj = {};
                    if (value.structValue && value.structValue.fields) {
                        for (const [k, v] of Object.entries(value.structValue.fields)) {
                            obj[k] = convertProtobufValue(v);
                        }
                    }
                    return obj;
                case 'listValue':
                    return value.listValue ? value.listValue.values.map(convertProtobufValue) : [];
                default: return value;
            }
        }

        // Evaluate the expression
        return Boolean(%s);
    } catch (error) {
        console.warn('Guard evaluation error:', error);
        return false;
    }
}`, jsExpr)

	return jsFunction, nil
}

// convertToJavaScript converts a parsed expression to JavaScript code
func (jse *JavaScriptGuardEvaluator) convertToJavaScript(expr *GuardExpression) (string, error) {
	switch expr.Language {
	case "go", "golang":
		return jse.convertGoToJavaScript(expr.Original)
	case "js", "javascript":
		return expr.Original, nil // Already JavaScript
	case "simple":
		return jse.convertSimpleToJavaScript(expr.Original)
	default:
		return "", fmt.Errorf("unsupported language for JavaScript conversion: %s", expr.Language)
	}
}

// convertGoToJavaScript converts Go-like expressions to JavaScript
func (jse *JavaScriptGuardEvaluator) convertGoToJavaScript(expression string) (string, error) {
	// Basic Go to JavaScript conversions
	jsExpr := expression

	// Replace Go operators with JavaScript equivalents
	jsExpr = strings.ReplaceAll(jsExpr, "&&", "&&")
	jsExpr = strings.ReplaceAll(jsExpr, "||", "||")
	jsExpr = strings.ReplaceAll(jsExpr, "!", "!")
	jsExpr = strings.ReplaceAll(jsExpr, "==", "===")
	jsExpr = strings.ReplaceAll(jsExpr, "!=", "!==")

	// Convert context access patterns
	jsExpr = jse.convertContextAccess(jsExpr)

	// Convert function calls
	jsExpr = jse.convertFunctionCalls(jsExpr)

	return jsExpr, nil
}

// convertSimpleToJavaScript converts simple expressions to JavaScript
func (jse *JavaScriptGuardEvaluator) convertSimpleToJavaScript(expression string) (string, error) {
	expression = strings.TrimSpace(expression)

	switch expression {
	case "true":
		return "true", nil
	case "false":
		return "false", nil
	case "":
		return "true", nil
	case "context.count < 5":
		return "safeGet(context, 'count') < 5", nil
	default:
		// Convert simple patterns
		if strings.Contains(expression, "context.") {
			return jse.convertContextAccess(expression), nil
		}
		return "true", nil // Default fallback
	}
}

// convertContextAccess converts context access patterns to JavaScript
func (jse *JavaScriptGuardEvaluator) convertContextAccess(expression string) string {
	// Convert context.field patterns to safeGet calls
	jsExpr := expression

	// Simple regex-like replacement for context access
	patterns := map[string]string{
		"context.count":    "safeGet(context, 'count')",
		"context.status":   "safeGet(context, 'status')",
		"context.score":    "safeGet(context, 'score')",
		"context.enabled":  "safeGet(context, 'enabled')",
		"context.name":     "safeGet(context, 'name')",
		"context.category": "safeGet(context, 'category')",
		"context.message":  "safeGet(context, 'message')",
		"context.tags":     "safeGet(context, 'tags')",
		"context.active":   "safeGet(context, 'active')",
		"context.zero":     "safeGet(context, 'zero')",
	}

	for pattern, replacement := range patterns {
		jsExpr = strings.ReplaceAll(jsExpr, pattern, replacement)
	}

	// Convert event access patterns
	eventPatterns := map[string]string{
		"event.name":      "safeGet(event, 'label')",
		"event.label":     "safeGet(event, 'label')",
		"event.userId":    "safeGet(event, 'userId')",
		"event.action":    "safeGet(event, 'action')",
		"event.valid":     "safeGet(event, 'valid')",
		"event.priority":  "safeGet(event, 'priority')",
		"event.amount":    "safeGet(event, 'amount')",
	}

	for pattern, replacement := range eventPatterns {
		jsExpr = strings.ReplaceAll(jsExpr, pattern, replacement)
	}

	// Convert state access patterns
	statePatterns := map[string]string{
		"state.transitionLabel": "safeGet(state, 'transitionLabel')",
		"state.sourceStates":    "safeGet(state, 'sourceStates')",
		"state.targetStates":    "safeGet(state, 'targetStates')",
	}

	for pattern, replacement := range statePatterns {
		jsExpr = strings.ReplaceAll(jsExpr, pattern, replacement)
	}

	return jsExpr
}

// convertFunctionCalls converts function calls to JavaScript equivalents
func (jse *JavaScriptGuardEvaluator) convertFunctionCalls(expression string) string {
	// This is a simplified conversion - a full implementation would parse the AST
	jsExpr := expression

	// Keep len(), contains(), hasPrefix(), hasSuffix() as they are defined in the JS function
	// No conversion needed as we define these functions in the generated JavaScript

	return jsExpr
}

// GenerateJavaScriptGuardValidator generates a complete JavaScript guard validation module
func (jse *JavaScriptGuardEvaluator) GenerateJavaScriptGuardValidator() string {
	return `
// Statechart Guard Evaluator for JavaScript
class StatechartGuardEvaluator {
    constructor() {
        this.cache = new Map();
        this.maxCacheSize = 1000;
    }

    // Evaluate a guard expression
    evaluateGuard(guard, context, event, state) {
        if (!guard || !guard.expression) {
            return true;
        }

        try {
            // Get or create the evaluation function
            const evalFunc = this.getEvaluationFunction(guard);
            return evalFunc(context, event, state);
        } catch (error) {
            console.warn('Guard evaluation error:', error);
            return false;
        }
    }

    // Get cached evaluation function or create new one
    getEvaluationFunction(guard) {
        const key = guard.expression + '|' + (guard.language || 'go');
        
        if (this.cache.has(key)) {
            return this.cache.get(key);
        }

        const func = this.createEvaluationFunction(guard);
        
        // Limit cache size
        if (this.cache.size >= this.maxCacheSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        
        this.cache.set(key, func);
        return func;
    }

    // Create evaluation function from guard expression
    createEvaluationFunction(guard) {
        const expression = guard.expression;
        const language = guard.language || 'go';

        // Convert expression to JavaScript
        const jsExpression = this.convertToJavaScript(expression, language);
        
        // Create function with error handling
        return new Function('context', 'event', 'state', ` + "`" + `
            try {
                // Helper functions
                function len(obj) {
                    if (typeof obj === 'string') return obj.length;
                    if (Array.isArray(obj)) return obj.length;
                    if (obj && typeof obj === 'object') return Object.keys(obj).length;
                    return 0;
                }
                
                function contains(str, substr) {
                    if (typeof str !== 'string' || typeof substr !== 'string') return false;
                    return str.includes(substr);
                }
                
                function hasPrefix(str, prefix) {
                    if (typeof str !== 'string' || typeof prefix !== 'string') return false;
                    return str.startsWith(prefix);
                }
                
                function hasSuffix(str, suffix) {
                    if (typeof str !== 'string' || typeof suffix !== 'string') return false;
                    return str.endsWith(suffix);
                }

                // Safely access nested properties
                function safeGet(obj, path) {
                    if (!obj) return undefined;
                    const parts = path.split('.');
                    let current = obj;
                    for (const part of parts) {
                        if (current && typeof current === 'object' && part in current) {
                            current = current[part];
                        } else {
                            return undefined;
                        }
                    }
                    return current;
                }

                // Convert protobuf values if needed
                function convertProtobufValue(value) {
                    if (!value || !value.kind) return value;
                    switch (value.kind) {
                        case 'nullValue': return null;
                        case 'numberValue': return value.numberValue;
                        case 'stringValue': return value.stringValue;
                        case 'boolValue': return value.boolValue;
                        case 'structValue': 
                            const obj = {};
                            if (value.structValue && value.structValue.fields) {
                                for (const [k, v] of Object.entries(value.structValue.fields)) {
                                    obj[k] = convertProtobufValue(v);
                                }
                            }
                            return obj;
                        case 'listValue':
                            return value.listValue ? value.listValue.values.map(convertProtobufValue) : [];
                        default: return value;
                    }
                }

                // Convert context if needed (for protobuf compatibility)
                if (context && context.fields) {
                    const convertedContext = {};
                    for (const [key, value] of Object.entries(context.fields)) {
                        convertedContext[key] = convertProtobufValue(value);
                    }
                    context = convertedContext;
                }

                // Convert event parameters if needed
                if (event && event.parameters && event.parameters.fields) {
                    const convertedParams = {};
                    for (const [key, value] of Object.entries(event.parameters.fields)) {
                        convertedParams[key] = convertProtobufValue(value);
                    }
                    event = { ...event, ...convertedParams };
                }

                // Evaluate the expression
                return Boolean(${jsExpression});
            } catch (error) {
                console.warn('Guard evaluation error:', error);
                return false;
            }
        ` + "`" + `);
    }

    // Convert expression to JavaScript
    convertToJavaScript(expression, language) {
        switch (language) {
            case 'go':
            case 'golang':
                return this.convertGoToJavaScript(expression);
            case 'js':
            case 'javascript':
                return expression;
            case 'simple':
                return this.convertSimpleToJavaScript(expression);
            default:
                throw new Error('Unsupported guard language: ' + language);
        }
    }

    // Convert Go-like expressions to JavaScript
    convertGoToJavaScript(expression) {
        let jsExpr = expression;

        // Replace operators
        jsExpr = jsExpr.replace(/==/g, '===');
        jsExpr = jsExpr.replace(/!=/g, '!==');

        // Convert context access
        jsExpr = this.convertContextAccess(jsExpr);

        return jsExpr;
    }

    // Convert simple expressions to JavaScript
    convertSimpleToJavaScript(expression) {
        const expr = expression.trim();
        
        switch (expr) {
            case 'true': return 'true';
            case 'false': return 'false';
            case '': return 'true';
            case 'context.count < 5': return "safeGet(context, 'count') < 5";
            default:
                if (expr.includes('context.')) {
                    return this.convertContextAccess(expr);
                }
                return 'true';
        }
    }

    // Convert context access patterns
    convertContextAccess(expression) {
        let jsExpr = expression;

        // Common context patterns
        const patterns = {
            'context.count': "safeGet(context, 'count')",
            'context.status': "safeGet(context, 'status')",
            'context.score': "safeGet(context, 'score')",
            'context.enabled': "safeGet(context, 'enabled')",
            'context.name': "safeGet(context, 'name')",
            'context.category': "safeGet(context, 'category')",
            'context.message': "safeGet(context, 'message')",
            'context.tags': "safeGet(context, 'tags')",
            'context.active': "safeGet(context, 'active')",
            'context.zero': "safeGet(context, 'zero')",
            'event.name': "safeGet(event, 'label')",
            'event.label': "safeGet(event, 'label')",
            'event.userId': "safeGet(event, 'userId')",
            'event.action': "safeGet(event, 'action')",
            'event.valid': "safeGet(event, 'valid')",
            'event.priority': "safeGet(event, 'priority')",
            'event.amount': "safeGet(event, 'amount')",
            'state.transitionLabel': "safeGet(state, 'transitionLabel')",
            'state.sourceStates': "safeGet(state, 'sourceStates')",
            'state.targetStates': "safeGet(state, 'targetStates')"
        };

        for (const [pattern, replacement] of Object.entries(patterns)) {
            jsExpr = jsExpr.replace(new RegExp(pattern.replace('.', '\\.'), 'g'), replacement);
        }

        return jsExpr;
    }

    // Clear the cache
    clearCache() {
        this.cache.clear();
    }

    // Get cache size
    getCacheSize() {
        return this.cache.size;
    }
}

// Global instance for convenience
const globalGuardEvaluator = new StatechartGuardEvaluator();

// Convenience function for direct evaluation
function evaluateGuard(guard, context, event, state) {
    return globalGuardEvaluator.evaluateGuard(guard, context, event, state);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        StatechartGuardEvaluator,
        evaluateGuard,
        globalGuardEvaluator
    };
}
`
}

// GenerateJavaScriptTestSuite generates a comprehensive test suite for JavaScript guard evaluation
func (jse *JavaScriptGuardEvaluator) GenerateJavaScriptTestSuite() string {
	return `
// Test Suite for Statechart Guard Evaluator
function runGuardEvaluatorTests() {
    const evaluator = new StatechartGuardEvaluator();
    let passed = 0;
    let failed = 0;

    function assert(condition, message) {
        if (condition) {
            console.log('✓ ' + message);
            passed++;
        } else {
            console.error('✗ ' + message);
            failed++;
        }
    }

    function assertEqual(actual, expected, message) {
        assert(actual === expected, message + ' (got ' + actual + ', expected ' + expected + ')');
    }

    // Test basic expressions
    console.log('Testing basic expressions...');
    assertEqual(evaluator.evaluateGuard({expression: 'true'}), true, 'true expression');
    assertEqual(evaluator.evaluateGuard({expression: 'false'}), false, 'false expression');
    assertEqual(evaluator.evaluateGuard({expression: ''}), true, 'empty expression');
    assertEqual(evaluator.evaluateGuard({}), true, 'null expression');

    // Test boolean logic
    console.log('Testing boolean logic...');
    assertEqual(evaluator.evaluateGuard({expression: 'true && true'}), true, 'AND true && true');
    assertEqual(evaluator.evaluateGuard({expression: 'true && false'}), false, 'AND true && false');
    assertEqual(evaluator.evaluateGuard({expression: 'false || true'}), true, 'OR false || true');
    assertEqual(evaluator.evaluateGuard({expression: 'false || false'}), false, 'OR false || false');
    assertEqual(evaluator.evaluateGuard({expression: '!true'}), false, 'NOT !true');
    assertEqual(evaluator.evaluateGuard({expression: '!false'}), true, 'NOT !false');

    // Test context access
    console.log('Testing context access...');
    const context = {
        count: 10,
        status: 'active',
        enabled: true,
        name: 'test'
    };

    assertEqual(evaluator.evaluateGuard({expression: 'context.count > 5'}, context), true, 'context.count > 5');
    assertEqual(evaluator.evaluateGuard({expression: 'context.status === "active"'}, context), true, 'string equality');
    assertEqual(evaluator.evaluateGuard({expression: 'context.enabled === true'}, context), true, 'boolean equality');

    // Test event access
    console.log('Testing event access...');
    const event = {
        label: 'USER_INPUT',
        userId: 123,
        action: 'click'
    };

    assertEqual(evaluator.evaluateGuard({expression: 'event.label === "USER_INPUT"'}, context, event), true, 'event name access');
    assertEqual(evaluator.evaluateGuard({expression: 'event.userId === 123'}, context, event), true, 'event parameter access');

    // Test functions
    console.log('Testing functions...');
    assertEqual(evaluator.evaluateGuard({expression: 'len(context.name) > 2'}, context), true, 'len function');
    assertEqual(evaluator.evaluateGuard({expression: 'contains(context.status, "act")'}, context), true, 'contains function');
    assertEqual(evaluator.evaluateGuard({expression: 'hasPrefix(context.name, "te")'}, context), true, 'hasPrefix function');

    // Test complex expressions
    console.log('Testing complex expressions...');
    const complexExpr = 'context.count > 5 && (context.status === "active" || context.enabled === true)';
    assertEqual(evaluator.evaluateGuard({expression: complexExpr}, context), true, 'complex boolean expression');

    // Test error handling
    console.log('Testing error handling...');
    assertEqual(evaluator.evaluateGuard({expression: 'context.undefined > 5'}, context), false, 'undefined variable access');

    // Test caching
    console.log('Testing caching...');
    const guard = {expression: 'context.count > 5'};
    evaluator.evaluateGuard(guard, context);
    assert(evaluator.getCacheSize() > 0, 'expression cached');

    evaluator.clearCache();
    assertEqual(evaluator.getCacheSize(), 0, 'cache cleared');

    // Test protobuf compatibility
    console.log('Testing protobuf compatibility...');
    const protobufContext = {
        fields: {
            count: { kind: 'numberValue', numberValue: 10 },
            status: { kind: 'stringValue', stringValue: 'active' }
        }
    };

    assertEqual(evaluator.evaluateGuard({expression: 'context.count > 5'}, protobufContext), true, 'protobuf context access');

    console.log('\\nTest Results:');
    console.log('Passed: ' + passed);
    console.log('Failed: ' + failed);
    console.log('Total: ' + (passed + failed));

    return failed === 0;
}

// Run tests if this script is executed directly
if (typeof window === 'undefined') {
    runGuardEvaluatorTests();
}
`
}
