/**
 * Comprehensive Code Generation Engine
 * 
 * Transforms Protocol Buffer statecharts into industry-standard formats
 * with support for XState, React, Vue.js, TypeScript, and more.
 */

// Base Generator class for all format generators
class BaseGenerator {
    constructor() {
        this.name = 'Base Generator';
        this.version = '1.0';
        this.supportsTypeScript = false;
        this.fileExtensions = ['js'];
    }

    async generate(statechart, options = {}) {
        throw new Error('generate() must be implemented by subclass');
    }

    validateStatechart(statechart) {
        if (!statechart) {
            throw new Error('Statechart is required');
        }
        if (!statechart.root_state) {
            throw new Error('Statechart must have a root state');
        }
        return true;
    }

    countStates(state) {
        if (!state) return 0;
        let count = 1;
        if (state.children) {
            count += state.children.reduce((sum, child) => sum + this.countStates(child), 0);
        }
        return count;
    }

    escapeString(str) {
        return str.replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/\r/g, '\\r');
    }

    formatCode(code, options = {}) {
        // Basic code formatting
        const lines = code.split('\n');
        const formatted = [];
        let indentLevel = 0;
        const indent = options.indent || '  ';

        for (let line of lines) {
            const trimmed = line.trim();
            if (trimmed.endsWith('{') || trimmed.endsWith('[')) {
                formatted.push(indent.repeat(indentLevel) + trimmed);
                indentLevel++;
            } else if (trimmed.startsWith('}') || trimmed.startsWith(']')) {
                indentLevel = Math.max(0, indentLevel - 1);
                formatted.push(indent.repeat(indentLevel) + trimmed);
            } else if (trimmed) {
                formatted.push(indent.repeat(indentLevel) + trimmed);
            } else {
                formatted.push('');
            }
        }

        return formatted.join('\n');
    }
}

// Simple XState Generator
class SimpleXStateGenerator extends BaseGenerator {
    constructor() {
        super();
        this.name = 'XState v5';
        this.version = '5.0';
        this.supportsTypeScript = true;
        this.fileExtensions = ['js', 'ts'];
    }

    async generate(statechart, options = {}) {
        try {
            this.validateStatechart(statechart);
            
            const machineId = options.machineId || 'statechart';
            const useTypeScript = options.typescript || false;
            
            const code = this.generateXStateCode(statechart, machineId, useTypeScript);
            
            return {
                success: true,
                code: this.formatCode(code, options),
                metadata: {
                    machineId: machineId,
                    stateCount: this.countStates(statechart.root_state),
                    transitionCount: statechart.transitions?.length || 0,
                    format: 'xstate'
                }
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    generateXStateCode(statechart, machineId, useTypeScript) {
        const imports = useTypeScript 
            ? `import { createMachine, assign } from 'xstate';`
            : `import { createMachine, assign } from 'xstate';`;

        const statesCode = this.generateStates(statechart.root_state);
        const eventsCode = this.generateEvents(statechart.events || []);
        const transitionsCode = this.generateTransitions(statechart.transitions || []);

        return `${imports}

export const ${machineId}Machine = createMachine({
  id: '${machineId}',
  initial: '${this.getInitialState(statechart.root_state)}',
  states: {
${statesCode}
  },
  on: {
${transitionsCode}
  }
});`;
    }

    generateStates(rootState, indent = '    ') {
        if (!rootState.children || rootState.children.length === 0) {
            return `${indent}${rootState.label}: { type: 'final' }`;
        }

        const states = [];
        for (const child of rootState.children) {
            if (child.children && child.children.length > 0) {
                states.push(`${indent}${child.label}: {
${indent}  initial: '${this.getInitialState(child)}',
${indent}  states: {
${this.generateStates(child, indent + '    ')}
${indent}  }
${indent}}`);
            } else {
                const finalType = child.is_final ? ', type: "final"' : '';
                states.push(`${indent}${child.label}: {${finalType}}`);
            }
        }
        return states.join(',\n');
    }

    generateEvents(events) {
        return events.map(event => `    ${event.label}: {}`).join(',\n');
    }

    generateTransitions(transitions) {
        return transitions.map(transition => {
            const target = transition.to && transition.to.length > 0 ? transition.to[0] : null;
            return `    ${transition.event}: { target: '${target}' }`;
        }).join(',\n');
    }

    getInitialState(state) {
        if (state.children && state.children.length > 0) {
            const initial = state.children.find(child => child.is_initial);
            return initial ? initial.label : state.children[0].label;
        }
        return state.label;
    }
}

// Main CodeGenerator class
class CodeGenerator {
    constructor() {
        this.generators = new Map();
        this.cache = new Map();
        this.config = {
            indentation: '  ',
            lineEnding: '\n',
            maxLineLength: 100,
            generateComments: true
        };
        
        this.initializeGenerators();
    }

    initializeGenerators() {
        // Register built-in generators
        this.registerGenerator('xstate', new SimpleXStateGenerator());
        
        // Try to register additional generators if they exist
        this.tryRegisterAdvancedGenerators();
    }

    tryRegisterAdvancedGenerators() {
        // Try to load advanced generators if available
        try {
            if (typeof XStateGenerator !== 'undefined') {
                this.registerGenerator('xstate-advanced', new XStateGenerator());
            }
            if (typeof ReactGenerator !== 'undefined') {
                this.registerGenerator('react', new ReactGenerator());
            }
            if (typeof VueGenerator !== 'undefined') {
                this.registerGenerator('vue', new VueGenerator());
            }
        } catch (error) {
            console.debug('Advanced generators not available:', error.message);
        }
    }

    registerGenerator(formatId, generator) {
        this.generators.set(formatId, generator);
        console.debug(`Registered generator: ${formatId} (${generator.name})`);
    }

    getAvailableFormats() {
        return Array.from(this.generators.entries()).map(([id, generator]) => ({
            id: id,
            name: generator.name,
            version: generator.version,
            supportsTypeScript: generator.supportsTypeScript,
            extensions: generator.fileExtensions
        }));
    }

    async generate(statechart, format = 'xstate', options = {}) {
        if (!this.generators.has(format)) {
            throw new Error(`Unknown format: ${format}. Available formats: ${Array.from(this.generators.keys()).join(', ')}`);
        }

        const cacheKey = `${format}_${JSON.stringify(statechart)}_${JSON.stringify(options)}`;
        if (this.cache.has(cacheKey)) {
            console.debug('Returning cached result for', format);
            return this.cache.get(cacheKey);
        }

        const generator = this.generators.get(format);
        const result = await generator.generate(statechart, { ...this.config, ...options });
        
        if (result.success) {
            this.cache.set(cacheKey, result);
        }

        return result;
    }

    clearCache() {
        this.cache.clear();
    }

    setConfig(config) {
        this.config = { ...this.config, ...config };
    }
}

// Export globally for non-module scripts
if (typeof window !== 'undefined') {
    window.BaseGenerator = BaseGenerator;
    window.SimpleXStateGenerator = SimpleXStateGenerator;
    window.CodeGenerator = CodeGenerator;
}

// Export for ES6 modules if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BaseGenerator, SimpleXStateGenerator, CodeGenerator };
}