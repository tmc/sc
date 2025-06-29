/**
 * Validation Engine for Generated Code
 * 
 * Provides comprehensive validation for different statechart formats
 * including XState, React, Vue, SCXML, and JSON.
 */

class BaseValidator {
    constructor(name) {
        this.name = name;
        this.errors = [];
        this.warnings = [];
    }

    validate(code, options = {}) {
        this.errors = [];
        this.warnings = [];
        
        try {
            return this.performValidation(code, options);
        } catch (error) {
            this.errors.push(`Validation error: ${error.message}`);
            return false;
        }
    }

    performValidation(code, options) {
        // Override in subclasses
        return true;
    }

    getResults() {
        return {
            valid: this.errors.length === 0,
            errors: this.errors,
            warnings: this.warnings
        };
    }

    addError(message) {
        this.errors.push(message);
    }

    addWarning(message) {
        this.warnings.push(message);
    }
}

class XStateValidator extends BaseValidator {
    constructor() {
        super('XState');
    }

    performValidation(code, options) {
        // Validate XState machine structure
        if (!code.includes('createMachine')) {
            this.addError('XState code must include createMachine function');
        }

        // Check for proper state structure
        if (!code.includes('states:')) {
            this.addError('XState machine must define states');
        }

        // Check for initial state
        if (!code.includes('initial:')) {
            this.addWarning('XState machine should specify initial state');
        }

        // Validate JavaScript syntax
        try {
            new Function(code);
        } catch (syntaxError) {
            this.addError(`JavaScript syntax error: ${syntaxError.message}`);
        }

        return this.errors.length === 0;
    }
}

class ReactValidator extends BaseValidator {
    constructor() {
        super('React');
    }

    performValidation(code, options) {
        // Validate React component structure
        if (!code.includes('import React') && !code.includes('import { ')) {
            this.addError('React component must import React');
        }

        // Check for component export
        if (!code.includes('export default') && !code.includes('export const')) {
            this.addError('React component must be exported');
        }

        // Check for JSX return
        if (!code.includes('return (') && !code.includes('return <')) {
            this.addError('React component must return JSX');
        }

        // Validate hooks usage if present
        if (code.includes('useState') && !code.includes('import { useState')) {
            this.addError('useState hook must be imported from React');
        }

        return this.errors.length === 0;
    }
}

class VueValidator extends BaseValidator {
    constructor() {
        super('Vue');
    }

    performValidation(code, options) {
        // Validate Vue component structure
        if (!code.includes('<template>') && !code.includes('template:')) {
            this.addError('Vue component must have template');
        }

        // Check for script section in SFC
        if (code.includes('<template>') && !code.includes('<script>')) {
            this.addWarning('Vue SFC should include script section');
        }

        // Validate Vue 3 Composition API if used
        if (code.includes('setup(') && !code.includes('import { ref')) {
            this.addWarning('Vue Composition API should import reactive functions');
        }

        return this.errors.length === 0;
    }
}

class SCXMLValidator extends BaseValidator {
    constructor() {
        super('SCXML');
    }

    performValidation(code, options) {
        // Validate SCXML XML structure
        if (!code.includes('<?xml')) {
            this.addError('SCXML must include XML declaration');
        }

        if (!code.includes('<scxml')) {
            this.addError('SCXML must have root scxml element');
        }

        if (!code.includes('<state')) {
            this.addError('SCXML must define at least one state');
        }

        // Check for proper namespace
        if (!code.includes('xmlns="http://www.w3.org/2005/07/scxml"')) {
            this.addWarning('SCXML should include proper namespace');
        }

        // Basic XML validation
        try {
            if (typeof DOMParser !== 'undefined') {
                const parser = new DOMParser();
                const doc = parser.parseFromString(code, 'text/xml');
                const errors = doc.getElementsByTagName('parsererror');
                if (errors.length > 0) {
                    this.addError('Invalid XML structure');
                }
            }
        } catch (error) {
            this.addError(`XML parsing error: ${error.message}`);
        }

        return this.errors.length === 0;
    }
}

class JSONValidator extends BaseValidator {
    constructor() {
        super('JSON');
    }

    performValidation(code, options) {
        // Validate JSON structure
        try {
            const parsed = JSON.parse(code);
            
            // Check for required statechart properties
            if (!parsed.rootState) {
                this.addError('JSON statechart must have rootState property');
            }

            if (!parsed.transitions) {
                this.addWarning('JSON statechart should define transitions');
            }

            if (!parsed.events) {
                this.addWarning('JSON statechart should define events');
            }

        } catch (jsonError) {
            this.addError(`Invalid JSON: ${jsonError.message}`);
        }

        return this.errors.length === 0;
    }
}

class TypeScriptValidator extends BaseValidator {
    constructor() {
        super('TypeScript');
    }

    performValidation(code, options) {
        // Basic TypeScript validation
        
        // Check for type annotations
        if (!code.includes(': ') && !code.includes('interface ') && !code.includes('type ')) {
            this.addWarning('TypeScript code should include type annotations');
        }

        // Check for proper imports
        if (code.includes('from \'') && !code.includes('import')) {
            this.addError('TypeScript imports must use import statement');
        }

        // Validate function signatures
        const functionRegex = /function\s+\w+\s*\(/g;
        const typedFunctionRegex = /function\s+\w+\s*\([^)]*:\s*\w+/g;
        
        const functions = code.match(functionRegex) || [];
        const typedFunctions = code.match(typedFunctionRegex) || [];
        
        if (functions.length > typedFunctions.length) {
            this.addWarning('Functions should have typed parameters');
        }

        return this.errors.length === 0;
    }
}

class ValidationEngine {
    constructor() {
        this.validators = new Map();
        this.initializeValidators();
    }

    initializeValidators() {
        this.validators.set('xstate', new XStateValidator());
        this.validators.set('xstate-typescript', new TypeScriptValidator());
        this.validators.set('react', new ReactValidator());
        this.validators.set('vue', new VueValidator());
        this.validators.set('scxml', new SCXMLValidator());
        this.validators.set('json', new JSONValidator());
        this.validators.set('typescript', new TypeScriptValidator());
    }

    validate(code, format, options = {}) {
        const validator = this.validators.get(format.toLowerCase());
        
        if (!validator) {
            return {
                valid: false,
                errors: [`Unknown format: ${format}`],
                warnings: []
            };
        }

        validator.validate(code, options);
        return validator.getResults();
    }

    getAvailableFormats() {
        return Array.from(this.validators.keys());
    }

    addValidator(name, validator) {
        this.validators.set(name.toLowerCase(), validator);
    }
}

// Export classes for global access
if (typeof window !== 'undefined') {
    window.XStateValidator = XStateValidator;
    window.ReactValidator = ReactValidator;
    window.VueValidator = VueValidator;
    window.SCXMLValidator = SCXMLValidator;
    window.JSONValidator = JSONValidator;
    window.TypeScriptValidator = TypeScriptValidator;
    window.ValidationEngine = ValidationEngine;
}

// CommonJS/Node.js export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        XStateValidator,
        ReactValidator,
        VueValidator,
        SCXMLValidator,
        JSONValidator,
        TypeScriptValidator,
        ValidationEngine
    };
}