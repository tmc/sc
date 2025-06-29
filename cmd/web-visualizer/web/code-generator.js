/**
 * Comprehensive Code Generation Engine
 * 
 * Transforms Protocol Buffer statecharts into industry-standard formats
 * with support for XState, React, Vue.js, TypeScript, and more.
 */

class CodeGenerator {
    constructor() {
        this.templates = new Map();
        this.formatGenerators = new Map();
        this.validation = new ValidationEngine();
        this.cache = new Map();
        
        // Initialize format generators
        this.initializeGenerators();
        
        // Configuration
        this.config = {
            indentation: '  ', // 2 spaces
            lineEnding: '\n',
            maxLineLength: 100,
            generateComments: true,
            generateTypeDefinitions: true,
            includeImports: true,
            optimizeForSize: false
        };
    }

    /**
     * Initialize all format generators
     */
    initializeGenerators() {
        this.formatGenerators.set('xstate', new XStateGenerator());
        this.formatGenerators.set('xstate-typescript', new XStateTypeScriptGenerator());
        this.formatGenerators.set('react', new ReactGenerator());
        this.formatGenerators.set('vue', new VueGenerator());
        this.formatGenerators.set('scxml', new SCXMLGenerator());
        this.formatGenerators.set('mermaid', new MermaidGenerator());
        this.formatGenerators.set('json', new JSONGenerator());
        this.formatGenerators.set('yaml', new YAMLGenerator());
        this.formatGenerators.set('plantuml', new PlantUMLGenerator());
        this.formatGenerators.set('graphviz', new GraphvizGenerator());
    }

    /**
     * Generate code for a specific format
     * @param {Object} statechart - Protocol Buffer statechart
     * @param {string} format - Target format (xstate, react, vue, etc.)
     * @param {Object} options - Generation options
     * @returns {Promise<GenerationResult>}
     */
    async generate(statechart, format, options = {}) {
        const startTime = performance.now();
        
        try {
            // Validate inputs
            this.validateInputs(statechart, format);
            
            // Create cache key
            const cacheKey = this.createCacheKey(statechart, format, options);
            
            // Check cache
            if (this.cache.has(cacheKey) && !options.skipCache) {
                const cached = this.cache.get(cacheKey);
                return {
                    ...cached,
                    cached: true,
                    generationTime: performance.now() - startTime
                };
            }

            // Get format generator
            const generator = this.formatGenerators.get(format);
            if (!generator) {
                throw new Error(`Unsupported format: ${format}`);
            }

            // Merge options with defaults
            const mergedOptions = {
                ...this.config,
                ...options
            };

            // Pre-process statechart
            const processedStatechart = this.preprocessStatechart(statechart);

            // Generate code
            const result = await generator.generate(processedStatechart, mergedOptions);

            // Post-process result
            const finalResult = this.postprocessResult(result, format, mergedOptions);

            // Validate generated code
            if (mergedOptions.validate !== false) {
                const validationResult = await this.validation.validate(finalResult.code, format);
                finalResult.validation = validationResult;
            }

            // Cache result
            this.cache.set(cacheKey, finalResult);

            // Add metadata
            finalResult.generationTime = performance.now() - startTime;
            finalResult.format = format;
            finalResult.options = mergedOptions;
            finalResult.cached = false;

            return finalResult;

        } catch (error) {
            return {
                success: false,
                error: error.message,
                stack: error.stack,
                format: format,
                generationTime: performance.now() - startTime
            };
        }
    }

    /**
     * Generate multiple formats simultaneously
     * @param {Object} statechart - Protocol Buffer statechart
     * @param {Array<string>} formats - Target formats
     * @param {Object} options - Generation options
     * @returns {Promise<Map<string, GenerationResult>>}
     */
    async generateMultiple(statechart, formats, options = {}) {
        const promises = formats.map(format => 
            this.generate(statechart, format, options)
                .then(result => [format, result])
        );

        const results = await Promise.all(promises);
        return new Map(results);
    }

    /**
     * Get available formats
     * @returns {Array<FormatInfo>}
     */
    getAvailableFormats() {
        const formats = [];
        for (const [key, generator] of this.formatGenerators) {
            formats.push({
                id: key,
                name: generator.getDisplayName(),
                description: generator.getDescription(),
                fileExtension: generator.getFileExtension(),
                mimeType: generator.getMimeType(),
                supportsTypeScript: generator.supportsTypeScript?.() || false,
                features: generator.getSupportedFeatures?.() || [],
                examples: generator.getExamples?.() || []
            });
        }
        return formats;
    }

    /**
     * Validate inputs
     */
    validateInputs(statechart, format) {
        if (!statechart) {
            throw new Error('Statechart is required');
        }

        if (!format) {
            throw new Error('Format is required');
        }

        if (typeof statechart !== 'object') {
            throw new Error('Statechart must be an object');
        }

        if (!statechart.root_state) {
            throw new Error('Statechart must have a root_state');
        }
    }

    /**
     * Create cache key
     */
    createCacheKey(statechart, format, options) {
        const hash = this.hashObject({
            statechart: this.stripNonEssentialFields(statechart),
            format,
            options
        });
        return `${format}-${hash}`;
    }

    /**
     * Hash object for cache key
     */
    hashObject(obj) {
        const str = JSON.stringify(obj, Object.keys(obj).sort());
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return hash.toString(36);
    }

    /**
     * Strip non-essential fields for caching
     */
    stripNonEssentialFields(statechart) {
        const { description, ...essential } = statechart;
        return essential;
    }

    /**
     * Pre-process statechart before generation
     */
    preprocessStatechart(statechart) {
        // Deep clone to avoid mutations
        const processed = JSON.parse(JSON.stringify(statechart));

        // Normalize state structure
        this.normalizeStates(processed.root_state);

        // Validate and fix transitions
        this.validateTransitions(processed);

        // Add missing default values
        this.addDefaults(processed);

        return processed;
    }

    /**
     * Normalize state structure
     */
    normalizeStates(state) {
        if (!state) return;

        // Ensure state has required fields
        state.label = state.label || 'unnamed';
        state.type = state.type || 'BASIC';
        state.children = state.children || [];
        state.is_initial = state.is_initial || false;
        state.is_final = state.is_final || false;

        // Recursively normalize children
        if (state.children) {
            state.children.forEach(child => this.normalizeStates(child));
        }

        // Ensure exactly one initial state for compound states
        if (state.type === 'OR' || state.type === 'NORMAL') {
            this.ensureInitialState(state);
        }
    }

    /**
     * Ensure compound states have exactly one initial state
     */
    ensureInitialState(state) {
        if (!state.children || state.children.length === 0) return;

        const initialStates = state.children.filter(child => child.is_initial);
        
        if (initialStates.length === 0) {
            // No initial state, mark first child as initial
            state.children[0].is_initial = true;
        } else if (initialStates.length > 1) {
            // Multiple initial states, keep only the first one
            for (let i = 1; i < initialStates.length; i++) {
                initialStates[i].is_initial = false;
            }
        }
    }

    /**
     * Validate transitions
     */
    validateTransitions(statechart) {
        if (!statechart.transitions) {
            statechart.transitions = [];
            return;
        }

        const stateLabels = this.collectStateLabels(statechart.root_state);
        
        statechart.transitions = statechart.transitions.filter(transition => {
            // Validate from states
            if (!transition.from || transition.from.length === 0) {
                console.warn('Transition missing from states:', transition);
                return false;
            }

            // Validate to states
            if (!transition.to || transition.to.length === 0) {
                console.warn('Transition missing to states:', transition);
                return false;
            }

            // Check if states exist
            const invalidFromStates = transition.from.filter(state => !stateLabels.has(state));
            const invalidToStates = transition.to.filter(state => !stateLabels.has(state));

            if (invalidFromStates.length > 0) {
                console.warn('Transition references non-existent from states:', invalidFromStates);
                return false;
            }

            if (invalidToStates.length > 0) {
                console.warn('Transition references non-existent to states:', invalidToStates);
                return false;
            }

            return true;
        });
    }

    /**
     * Collect all state labels
     */
    collectStateLabels(state, labels = new Set()) {
        if (!state) return labels;

        labels.add(state.label);

        if (state.children) {
            state.children.forEach(child => this.collectStateLabels(child, labels));
        }

        return labels;
    }

    /**
     * Add default values
     */
    addDefaults(statechart) {
        statechart.name = statechart.name || 'StatechartMachine';
        statechart.description = statechart.description || 'Generated statechart machine';
        statechart.events = statechart.events || [];
        statechart.variables = statechart.variables || {};

        // Ensure events array contains all events referenced in transitions
        const referencedEvents = new Set();
        if (statechart.transitions) {
            statechart.transitions.forEach(transition => {
                if (transition.event) {
                    referencedEvents.add(transition.event);
                }
            });
        }

        // Add missing events
        const existingEvents = new Set(statechart.events.map(e => e.label));
        for (const eventLabel of referencedEvents) {
            if (!existingEvents.has(eventLabel)) {
                statechart.events.push({
                    label: eventLabel,
                    parameters: {}
                });
            }
        }
    }

    /**
     * Post-process generated result
     */
    postprocessResult(result, format, options) {
        if (!result.success) {
            return result;
        }

        // Format code
        if (options.formatCode !== false) {
            result.code = this.formatCode(result.code, format);
        }

        // Add file header
        if (options.includeHeader !== false) {
            result.code = this.addFileHeader(result.code, format, options);
        }

        // Add exports
        if (options.includeExports !== false) {
            result.code = this.addExports(result.code, format, options);
        }

        return result;
    }

    /**
     * Format generated code
     */
    formatCode(code, format) {
        try {
            // Basic formatting for JavaScript/TypeScript
            if (format.includes('xstate') || format === 'react' || format === 'vue') {
                return this.formatJavaScript(code);
            }

            // Format XML-based formats
            if (format === 'scxml') {
                return this.formatXML(code);
            }

            // Format JSON
            if (format === 'json') {
                return this.formatJSON(code);
            }

            return code;
        } catch (error) {
            console.warn('Code formatting failed:', error);
            return code;
        }
    }

    /**
     * Format JavaScript code
     */
    formatJavaScript(code) {
        return code
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0)
            .join('\n')
            .replace(/\n\s*\n/g, '\n')
            .replace(/{\s*\n\s*/g, '{\n  ')
            .replace(/\n\s*}/g, '\n}');
    }

    /**
     * Format XML code
     */
    formatXML(code) {
        // Basic XML formatting
        let formatted = code;
        let indent = 0;
        const lines = formatted.split('\n');
        
        return lines.map(line => {
            const trimmed = line.trim();
            if (trimmed.startsWith('</')) {
                indent--;
            }
            const indented = '  '.repeat(Math.max(0, indent)) + trimmed;
            if (trimmed.startsWith('<') && !trimmed.startsWith('</') && !trimmed.endsWith('/>')) {
                indent++;
            }
            return indented;
        }).join('\n');
    }

    /**
     * Format JSON code
     */
    formatJSON(code) {
        try {
            const parsed = JSON.parse(code);
            return JSON.stringify(parsed, null, 2);
        } catch (error) {
            return code;
        }
    }

    /**
     * Add file header
     */
    addFileHeader(code, format, options) {
        const timestamp = new Date().toISOString();
        const generator = this.formatGenerators.get(format);
        const extension = generator?.getFileExtension() || 'js';
        
        const header = `/**
 * Generated Statechart Machine
 * 
 * Format: ${format}
 * Generated: ${timestamp}
 * Generator: Statechart Code Generator v1.0
 * 
 * This file was automatically generated from a Protocol Buffer statechart.
 * Do not edit this file directly. Instead, modify the source statechart
 * and regenerate this file.
 */

`;

        return header + code;
    }

    /**
     * Add exports
     */
    addExports(code, format, options) {
        const generator = this.formatGenerators.get(format);
        if (generator?.addExports) {
            return generator.addExports(code, options);
        }
        return code;
    }

    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
    }

    /**
     * Get cache statistics
     */
    getCacheStats() {
        return {
            size: this.cache.size,
            entries: Array.from(this.cache.keys())
        };
    }

    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        this.clearCache(); // Clear cache when config changes
    }

    /**
     * Get current configuration
     */
    getConfig() {
        return { ...this.config };
    }
}

/**
 * Validation Engine for generated code
 */
class ValidationEngine {
    constructor() {
        this.validators = new Map();
        this.initializeValidators();
    }

    initializeValidators() {
        this.validators.set('xstate', new XStateValidator());
        this.validators.set('xstate-typescript', new XStateValidator());
        this.validators.set('react', new ReactValidator());
        this.validators.set('vue', new VueValidator());
        this.validators.set('scxml', new SCXMLValidator());
        this.validators.set('json', new JSONValidator());
    }

    async validate(code, format) {
        const validator = this.validators.get(format);
        if (!validator) {
            return {
                valid: true,
                warnings: [`No validator available for format: ${format}`]
            };
        }

        try {
            return await validator.validate(code);
        } catch (error) {
            return {
                valid: false,
                errors: [error.message],
                warnings: []
            };
        }
    }
}

/**
 * Base class for format generators
 */
class BaseGenerator {
    constructor() {
        this.templates = new Map();
    }

    /**
     * Generate code for the format
     * @param {Object} statechart - Preprocessed statechart
     * @param {Object} options - Generation options
     * @returns {Promise<GenerationResult>}
     */
    async generate(statechart, options) {
        throw new Error('generate method must be implemented by subclass');
    }

    /**
     * Get display name for this format
     */
    getDisplayName() {
        return 'Unknown Format';
    }

    /**
     * Get description for this format
     */
    getDescription() {
        return 'No description available';
    }

    /**
     * Get file extension
     */
    getFileExtension() {
        return 'txt';
    }

    /**
     * Get MIME type
     */
    getMimeType() {
        return 'text/plain';
    }

    /**
     * Generate imports section
     */
    generateImports(options) {
        return '';
    }

    /**
     * Generate exports section
     */
    generateExports(machineName, options) {
        return '';
    }

    /**
     * Escape string for target language
     */
    escapeString(str) {
        return str.replace(/"/g, '\\"').replace(/\n/g, '\\n');
    }

    /**
     * Generate identifier from label
     */
    generateIdentifier(label) {
        return label.replace(/[^a-zA-Z0-9_]/g, '_').replace(/^[0-9]/, '_$&');
    }

    /**
     * Capitalize first letter
     */
    capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    /**
     * Convert to camelCase
     */
    toCamelCase(str) {
        return str.replace(/[-_\s]+(.)?/g, (_, char) => char ? char.toUpperCase() : '');
    }

    /**
     * Convert to PascalCase
     */
    toPascalCase(str) {
        return this.capitalize(this.toCamelCase(str));
    }

    /**
     * Convert to kebab-case
     */
    toKebabCase(str) {
        return str.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase();
    }
}

// Export the main CodeGenerator class and utilities
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CodeGenerator, BaseGenerator, ValidationEngine };
} else if (typeof window !== 'undefined') {
    window.CodeGenerator = CodeGenerator;
    window.BaseGenerator = BaseGenerator;
    window.ValidationEngine = ValidationEngine;
}