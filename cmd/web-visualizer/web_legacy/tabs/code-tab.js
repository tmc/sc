/**
 * Enhanced Code Tab Component
 * 
 * Provides comprehensive code generation and editing for statecharts
 * with multi-format support, real-time generation, and advanced features.
 */
class CodeTab {
    constructor(container, stateManager) {
        this.container = container;
        this.stateManager = stateManager;
        this.editor = null;
        this.currentMachine = null;
        this.isInitialized = false;
        
        // Code generation
        this.codeGenerator = null;
        this.currentFormat = 'xstate';
        this.supportedFormats = new Map();
        this.generationCache = new Map();
        
        // Editor configuration
        this.editorConfig = {
            language: 'javascript',
            theme: 'vs-dark',
            automaticLayout: true,
            minimap: { enabled: true },
            fontSize: 14,
            wordWrap: 'on',
            lineNumbers: 'on',
            folding: true,
            bracketMatching: 'always',
            autoIndent: 'full',
            formatOnPaste: true,
            formatOnType: true,
            scrollBeyondLastLine: false,
            renderWhitespace: 'selection',
            tabSize: 2,
            insertSpaces: true
        };
        
        // Generation options
        this.generationOptions = {
            generateTypeDefinitions: true,
            includeComments: true,
            includeImports: true,
            formatCode: true,
            validateOutput: true
        };
        
        // Change tracking
        this.lastSavedContent = '';
        this.hasUnsavedChanges = false;
        this.changeTimeout = null;
        this.changeDebounceMs = 500;
        this.generationTimeout = null;
        this.generationDebounceMs = 300;
    }
    
    async initialize() {
        if (this.isInitialized) return;
        
        try {
            await this.initializeCodeGenerator();
            await this.loadSupportedFormats();
            await this.loadMonacoEditor();
            this.createEditor();
            this.setupEventHandlers();
            this.loadCurrentMachine();
            this.isInitialized = true;
        } catch (error) {
            console.error('Failed to initialize Code Tab:', error);
            this.showError(error.message);
        }
    }
    
    async initializeCodeGenerator() {
        // Wait for CodeGenerator to be available with retries
        let retries = 0;
        const maxRetries = 10;
        const retryDelay = 100;
        
        while (retries < maxRetries) {
            if (typeof window.CodeGenerator !== 'undefined') {
                try {
                    // Initialize the code generator with all format generators
                    this.codeGenerator = new window.CodeGenerator();
                    
                    // Load format-specific generators
                    await this.loadFormatGenerators();
                    
                    console.log('✅ Code generator initialized with formats:', 
                               this.codeGenerator.getAvailableFormats().map(f => f.id));
                    return; // Success!
                } catch (error) {
                    console.warn('CodeGenerator instantiation failed:', error);
                    // Fall through to retry
                }
            }
            
            // Wait before retrying
            await new Promise(resolve => setTimeout(resolve, retryDelay));
            retries++;
        }
        
        // If we get here, CodeGenerator failed to load after all retries
        console.warn('CodeGenerator not available after retries, using fallback');
        this.codeGenerator = this.createFallbackCodeGenerator();
    }
    
    createFallbackCodeGenerator() {
        // Minimal fallback implementation
        return {
            getAvailableFormats() {
                return [
                    { id: 'xstate', name: 'XState v5', version: '5.0' },
                    { id: 'json', name: 'JSON Export', version: '1.0' }
                ];
            },
            
            async generate(statechart, format = 'xstate', options = {}) {
                if (!statechart || !statechart.root_state) {
                    return { success: false, error: 'Invalid statechart' };
                }
                
                const machineId = options.machineId || 'machine';
                
                if (format === 'xstate') {
                    return {
                        success: true,
                        code: this.generateSimpleXStateCode(statechart, machineId),
                        metadata: { format: 'xstate', machineId: machineId }
                    };
                } else if (format === 'json') {
                    return {
                        success: true,
                        code: JSON.stringify(statechart, null, 2),
                        metadata: { format: 'json' }
                    };
                }
                
                return { success: false, error: 'Unsupported format: ' + format };
            },
            
            generateSimpleXStateCode(statechart, machineId) {
                const initialState = this.getInitialState(statechart.root_state);
                const states = this.generateStatesCode(statechart.root_state);
                
                return `import { createMachine } from 'xstate';

export const ${machineId}Machine = createMachine({
  id: '${machineId}',
  initial: '${initialState}',
  states: {
${states}
  }
});`;
            },
            
            getInitialState(state) {
                if (state.children && state.children.length > 0) {
                    const initial = state.children.find(child => child.is_initial);
                    return initial ? initial.label : state.children[0].label;
                }
                return state.label;
            },
            
            generateStatesCode(state, indent = '    ') {
                if (!state.children || state.children.length === 0) {
                    return `${indent}${state.label}: {}`;
                }
                
                const states = [];
                for (const child of state.children) {
                    if (child.children && child.children.length > 0) {
                        states.push(`${indent}${child.label}: {
${indent}  initial: '${this.getInitialState(child)}',
${indent}  states: {
${this.generateStatesCode(child, indent + '    ')}
${indent}  }
${indent}}`);
                    } else {
                        states.push(`${indent}${child.label}: {}`);
                    }
                }
                return states.join(',\n');
            }
        };
    }
    
    async loadFormatGenerators() {
        // This would normally load the generator modules
        // For now, we'll assume they're already loaded globally
        
        // The generators are loaded in the HTML file:
        // - XStateGenerator, XStateTypeScriptGenerator
        // - ReactGenerator
        // - VueGenerator  
        // - SCXMLGenerator, JSONGenerator, etc.
        
        // They are registered in the CodeGenerator constructor
    }
    
    async loadSupportedFormats() {
        try {
            const response = await fetch('/api/v1/export/formats');
            const data = await response.json();
            
            if (data.formats) {
                for (const [key, format] of Object.entries(data.formats)) {
                    this.supportedFormats.set(key, format);
                }
            }
        } catch (error) {
            console.warn('Failed to load supported formats from server:', error);
            // Use default formats
            this.loadDefaultFormats();
        }
    }
    
    loadDefaultFormats() {
        const defaultFormats = [
            { id: 'xstate', name: 'XState v5', extensions: ['js'] },
            { id: 'xstate-ts', name: 'XState TypeScript', extensions: ['ts'] },
            { id: 'react', name: 'React Components', extensions: ['jsx'] },
            { id: 'react-ts', name: 'React TypeScript', extensions: ['tsx'] },
            { id: 'vue', name: 'Vue Components', extensions: ['vue'] },
            { id: 'scxml', name: 'SCXML', extensions: ['scxml'] },
            { id: 'json', name: 'JSON', extensions: ['json'] },
            { id: 'mermaid', name: 'Mermaid', extensions: ['mmd'] }
        ];
        
        for (const format of defaultFormats) {
            this.supportedFormats.set(format.id, format);
        }
    }
    
    async loadMonacoEditor() {
        // Check if Monaco is already loaded
        if (window.monaco) {
            return;
        }
        
        // Load Monaco Editor from CDN
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs/loader.js';
            script.onload = () => {
                require.config({ 
                    paths: { 
                        'vs': 'https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs' 
                    } 
                });
                
                require(['vs/editor/editor.main'], () => {
                    this.setupMonacoLanguage();
                    resolve();
                });
            };
            script.onerror = () => reject(new Error('Failed to load Monaco Editor'));
            document.head.appendChild(script);
        });
    }
    
    setupMonacoLanguage() {
        // Register XState/JavaScript language features
        monaco.languages.typescript.javascriptDefaults.setCompilerOptions({
            target: monaco.languages.typescript.ScriptTarget.ES2020,
            allowNonTsExtensions: true,
            moduleResolution: monaco.languages.typescript.ModuleResolutionKind.NodeJs,
            module: monaco.languages.typescript.ModuleKind.CommonJS,
            noEmit: true,
            esModuleInterop: true,
            jsx: monaco.languages.typescript.JsxEmit.React,
            allowJs: true,
            typeRoots: ["node_modules/@types"]
        });
        
        // Add XState type definitions
        this.addXStateTypeDefinitions();
        
        // Setup custom themes
        this.setupCustomThemes();
    }
    
    addXStateTypeDefinitions() {
        const xstateTypes = `
            declare module 'xstate' {
                export interface MachineConfig {
                    id?: string;
                    initial?: string;
                    states: Record<string, StateConfig>;
                    context?: any;
                    on?: Record<string, Transition>;
                }
                
                export interface StateConfig {
                    initial?: string;
                    states?: Record<string, StateConfig>;
                    on?: Record<string, Transition>;
                    entry?: Action | Action[];
                    exit?: Action | Action[];
                    type?: 'atomic' | 'compound' | 'parallel' | 'final';
                }
                
                export type Transition = string | TransitionConfig;
                export interface TransitionConfig {
                    target?: string;
                    cond?: string | Function;
                    actions?: Action | Action[];
                }
                
                export type Action = string | Function | ActionObject;
                export interface ActionObject {
                    type: string;
                    [key: string]: any;
                }
                
                export function createMachine(config: MachineConfig): any;
            }
        `;
        
        monaco.languages.typescript.javascriptDefaults.addExtraLib(
            xstateTypes,
            'file:///node_modules/@types/xstate/index.d.ts'
        );
    }
    
    setupCustomThemes() {
        // Dark theme optimized for statecharts
        monaco.editor.defineTheme('statechart-dark', {
            base: 'vs-dark',
            inherit: true,
            rules: [
                { token: 'string.key.json', foreground: '9CDCFE' },
                { token: 'string.value.json', foreground: 'CE9178' },
                { token: 'number.json', foreground: 'B5CEA8' },
                { token: 'keyword.json', foreground: '569CD6' },
                { token: 'comment', foreground: '6A9955', fontStyle: 'italic' }
            ],
            colors: {
                'editor.background': '#1a1a1a',
                'editor.lineHighlightBackground': '#2d2d2d',
                'editorLineNumber.foreground': '#858585',
                'editorLineNumber.activeForeground': '#c6c6c6'
            }
        });
        
        this.editorConfig.theme = 'statechart-dark';
    }
    
    createEditor() {
        this.container.innerHTML = `
            <div class="code-tab">
                <div class="code-tab-header">
                    <div class="code-tab-title">
                        <span class="tab-icon">
                            <svg viewBox="0 0 24 24">
                                <path d="M14.6,16.6L19.2,12L14.6,7.4L16,6L22,12L16,18L14.6,16.6M9.4,16.6L4.8,12L9.4,7.4L8,6L2,12L8,18L9.4,16.6Z"/>
                            </svg>
                        </span>
                        <span class="tab-title-text">Code Generation</span>
                        <span class="unsaved-indicator" style="display: none;">●</span>
                    </div>
                    <div class="code-tab-controls">
                        <div class="format-selector">
                            <label for="format-select">Format:</label>
                            <select id="format-select" class="format-select">
                                ${this.generateFormatOptions()}
                            </select>
                        </div>
                        <div class="generation-options">
                            <button class="options-button" title="Generation Options" data-action="options">
                                <svg viewBox="0 0 24 24">
                                    <path d="M12,15.5A3.5,3.5 0 0,1 8.5,12A3.5,3.5 0 0,1 12,8.5A3.5,3.5 0 0,1 15.5,12A3.5,3.5 0 0,1 12,15.5M19.43,12.97C19.47,12.65 19.5,12.33 19.5,12C19.5,11.67 19.47,11.34 19.43,11L21.54,9.37C21.73,9.22 21.78,8.95 21.66,8.73L19.66,5.27C19.54,5.05 19.27,4.96 19.05,5.05L16.56,6.05C16.04,5.66 15.5,5.32 14.87,5.07L14.5,2.42C14.46,2.18 14.25,2 14,2H10C9.75,2 9.54,2.18 9.5,2.42L9.13,5.07C8.5,5.32 7.96,5.66 7.44,6.05L4.95,5.05C4.73,4.96 4.46,5.05 4.34,5.27L2.34,8.73C2.22,8.95 2.27,9.22 2.46,9.37L4.57,11C4.53,11.34 4.5,11.67 4.5,12C4.5,12.33 4.53,12.65 4.57,12.97L2.46,14.63C2.27,14.78 2.22,15.05 2.34,15.27L4.34,18.73C4.46,18.95 4.73,19.03 4.95,18.95L7.44,17.94C7.96,18.34 8.5,18.68 9.13,18.93L9.5,21.58C9.54,21.82 9.75,22 10,22H14C14.25,22 14.46,21.82 14.5,21.58L14.87,18.93C15.5,18.68 16.04,18.34 16.56,17.94L19.05,18.95C19.27,19.03 19.54,18.95 19.66,18.73L21.66,15.27C21.78,15.05 21.73,14.78 21.54,14.63L19.43,12.97Z"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                    <div class="code-tab-actions">
                        <button class="action-button" title="Regenerate Code" data-action="regenerate">
                            <svg viewBox="0 0 24 24">
                                <path d="M17.65,6.35C16.2,4.9 14.21,4 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20C15.73,20 18.84,17.45 19.73,14H17.65C16.83,16.33 14.61,18 12,18A6,6 0 0,1 6,12A6,6 0 0,1 12,6C13.66,6 15.14,6.69 16.22,7.78L13,11H20V4L17.65,6.35Z"/>
                            </svg>
                        </button>
                        <button class="action-button" title="Format Code" data-action="format">
                            <svg viewBox="0 0 24 24">
                                <path d="M3,3H21V5H3V3M3,7H15V9H3V7M3,11H21V13H3V11M3,15H15V17H3V15M3,19H21V21H3V19Z"/>
                            </svg>
                        </button>
                        <button class="action-button" title="Validate" data-action="validate">
                            <svg viewBox="0 0 24 24">
                                <path d="M12,2C13.1,2 14,2.9 14,4C14,5.1 13.1,6 12,6C10.9,6 10,5.1 10,4C10,2.9 10.9,2 12,2M21,9V7L15,1H5C3.89,1 3,1.89 3,3V21A2,2 0 0,0 5,23H19A2,2 0 0,0 21,21V9M19,9H14V4H5V21H19V9Z"/>
                            </svg>
                        </button>
                        <button class="action-button" title="Download" data-action="download">
                            <svg viewBox="0 0 24 24">
                                <path d="M5,20H19V18H5M19,9H15V3H9V9H5L12,16L19,9Z"/>
                            </svg>
                        </button>
                        <button class="action-button" title="Copy to Clipboard" data-action="copy">
                            <svg viewBox="0 0 24 24">
                                <path d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="code-generation-status">
                    <div class="generation-info">
                        <span class="generation-time"></span>
                        <span class="generation-size"></span>
                        <span class="generation-cache"></span>
                    </div>
                    <div class="generation-validation">
                        <span class="validation-indicator"></span>
                    </div>
                </div>
                <div class="code-editor-container" id="code-editor"></div>
                <div class="code-tab-footer">
                    <div class="code-info">
                        <span class="cursor-position">Line 1, Column 1</span>
                        <span class="selection-info"></span>
                        <span class="language-info">JavaScript</span>
                        <span class="file-info"></span>
                    </div>
                    <div class="code-status">
                        <span class="validation-status"></span>
                        <span class="dependencies-info"></span>
                    </div>
                </div>
                
                <!-- Generation Options Modal -->
                <div class="options-modal" id="options-modal" style="display: none;">
                    <div class="options-modal-content">
                        <div class="options-modal-header">
                            <h3>Code Generation Options</h3>
                            <button class="close-modal" data-action="close-options">×</button>
                        </div>
                        <div class="options-modal-body">
                            ${this.generateOptionsForm()}
                        </div>
                        <div class="options-modal-footer">
                            <button class="btn btn-secondary" data-action="reset-options">Reset to Defaults</button>
                            <button class="btn btn-primary" data-action="apply-options">Apply</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        const editorContainer = this.container.querySelector('#code-editor');
        
        this.editor = monaco.editor.create(editorContainer, {
            value: this.getDefaultMachineCode(),
            ...this.editorConfig
        });
        
        // Initial formatting
        setTimeout(() => {
            this.formatCode();
        }, 100);
    }
    
    generateFormatOptions() {
        let options = '';
        for (const [id, format] of this.supportedFormats) {
            const selected = id === this.currentFormat ? 'selected' : '';
            options += `<option value="${id}" ${selected}>${format.name || id}</option>`;
        }
        return options;
    }
    
    generateOptionsForm() {
        return `
            <div class="options-form">
                <div class="option-group">
                    <h4>Code Generation</h4>
                    <label class="option-item">
                        <input type="checkbox" name="generateTypeDefinitions" ${this.generationOptions.generateTypeDefinitions ? 'checked' : ''}>
                        <span>Generate TypeScript definitions</span>
                    </label>
                    <label class="option-item">
                        <input type="checkbox" name="includeComments" ${this.generationOptions.includeComments ? 'checked' : ''}>
                        <span>Include documentation comments</span>
                    </label>
                    <label class="option-item">
                        <input type="checkbox" name="includeImports" ${this.generationOptions.includeImports ? 'checked' : ''}>
                        <span>Include import statements</span>
                    </label>
                    <label class="option-item">
                        <input type="checkbox" name="formatCode" ${this.generationOptions.formatCode ? 'checked' : ''}>
                        <span>Auto-format generated code</span>
                    </label>
                </div>
                <div class="option-group">
                    <h4>Validation</h4>
                    <label class="option-item">
                        <input type="checkbox" name="validateOutput" ${this.generationOptions.validateOutput ? 'checked' : ''}>
                        <span>Validate generated code</span>
                    </label>
                </div>
                <div class="option-group">
                    <h4>Performance</h4>
                    <label class="option-item">
                        <input type="number" name="generationDebounceMs" value="${this.generationDebounceMs}" min="100" max="2000" step="100">
                        <span>Generation delay (ms)</span>
                    </label>
                </div>
            </div>
        `;
    }
    
    setupEventHandlers() {
        // Editor change events
        this.editor.onDidChangeModelContent(() => {
            this.onContentChange();
        });
        
        // Cursor position changes
        this.editor.onDidChangeCursorPosition((e) => {
            this.updateCursorPosition(e.position);
        });
        
        // Selection changes
        this.editor.onDidChangeCursorSelection((e) => {
            this.updateSelectionInfo(e.selection);
        });
        
        // Format selector
        const formatSelect = this.container.querySelector('#format-select');
        if (formatSelect) {
            formatSelect.addEventListener('change', (e) => {
                this.onFormatChanged(e.target.value);
            });
        }

        // Action buttons
        this.container.addEventListener('click', (e) => {
            const actionButton = e.target.closest('[data-action]');
            if (actionButton) {
                this.handleAction(actionButton.dataset.action);
            }
        });
        
        // State manager events
        this.stateManager.addEventListener('currentMachineChanged', (e) => {
            this.onMachineChanged(e.detail.machineId);
        });
        
        // Auto-save
        this.setupAutoSave();
    }
    
    onContentChange() {
        this.hasUnsavedChanges = this.getEditorContent() !== this.lastSavedContent;
        this.updateUnsavedIndicator();
        this.debounceValidation();
        this.debounceStateUpdate();
    }
    
    debounceValidation() {
        if (this.validationTimeout) {
            clearTimeout(this.validationTimeout);
        }
        
        this.validationTimeout = setTimeout(() => {
            this.validateCode();
        }, 1000);
    }
    
    debounceStateUpdate() {
        if (this.changeTimeout) {
            clearTimeout(this.changeTimeout);
        }
        
        this.changeTimeout = setTimeout(() => {
            this.updateStateFromCode();
        }, this.changeDebounceMs);
    }
    
    debounceCodeGeneration() {
        if (this.generationTimeout) {
            clearTimeout(this.generationTimeout);
        }
        
        this.generationTimeout = setTimeout(() => {
            this.generateCodeForCurrentMachine();
        }, this.generationDebounceMs);
    }
    
    updateStateFromCode() {
        try {
            const code = this.getEditorContent();
            // Parse and update machine definition in state
            // This would integrate with the backend API
            console.log('Code updated, would sync with backend');
        } catch (error) {
            console.warn('Failed to update state from code:', error);
        }
    }
    
    validateCode() {
        try {
            const code = this.getEditorContent();
            
            // Basic JavaScript syntax validation
            new Function(code);
            
            // TODO: Add XState-specific validation
            this.showValidationStatus('valid', 'Code is valid');
            
        } catch (error) {
            this.showValidationStatus('error', error.message);
        }
    }
    
    showValidationStatus(type, message) {
        const statusElement = this.container.querySelector('.validation-status');
        if (statusElement) {
            statusElement.className = `validation-status ${type}`;
            statusElement.textContent = message;
        }
    }
    
    updateCursorPosition(position) {
        const positionElement = this.container.querySelector('.cursor-position');
        if (positionElement) {
            positionElement.textContent = `Line ${position.lineNumber}, Column ${position.column}`;
        }
    }
    
    updateSelectionInfo(selection) {
        const selectionElement = this.container.querySelector('.selection-info');
        if (selectionElement) {
            if (selection.isEmpty()) {
                selectionElement.textContent = '';
            } else {
                const start = selection.getStartPosition();
                const end = selection.getEndPosition();
                selectionElement.textContent = `(${start.lineNumber},${start.column})-(${end.lineNumber},${end.column})`;
            }
        }
    }
    
    updateUnsavedIndicator() {
        const indicator = this.container.querySelector('.unsaved-indicator');
        if (indicator) {
            indicator.style.display = this.hasUnsavedChanges ? 'inline' : 'none';
        }
    }
    
    handleAction(action) {
        switch (action) {
            case 'regenerate':
                this.regenerateCode();
                break;
            case 'format':
                this.formatCode();
                break;
            case 'validate':
                this.validateCode();
                break;
            case 'download':
                this.downloadCode();
                break;
            case 'copy':
                this.copyToClipboard();
                break;
            case 'options':
                this.showOptionsModal();
                break;
            case 'close-options':
                this.hideOptionsModal();
                break;
            case 'apply-options':
                this.applyOptions();
                break;
            case 'reset-options':
                this.resetOptions();
                break;
        }
    }
    
    formatCode() {
        if (this.editor) {
            this.editor.getAction('editor.action.formatDocument').run();
        }
    }
    
    async copyToClipboard() {
        try {
            const code = this.getEditorContent();
            await navigator.clipboard.writeText(code);
            
            // Show temporary success message
            this.showTemporaryMessage('Copied to clipboard');
        } catch (error) {
            console.error('Failed to copy to clipboard:', error);
            this.showTemporaryMessage('Failed to copy', 'error');
        }
    }
    
    showTemporaryMessage(message, type = 'success') {
        const messageElement = document.createElement('div');
        messageElement.className = `temporary-message ${type}`;
        messageElement.textContent = message;
        
        this.container.appendChild(messageElement);
        
        setTimeout(() => {
            messageElement.remove();
        }, 2000);
    }
    
    setupAutoSave() {
        // Auto-save every 30 seconds if there are changes
        setInterval(() => {
            if (this.hasUnsavedChanges && this.currentMachine) {
                this.saveChanges();
            }
        }, 30000);
    }
    
    async saveChanges() {
        try {
            const code = this.getEditorContent();
            // TODO: Parse code and update machine via API
            this.lastSavedContent = code;
            this.hasUnsavedChanges = false;
            this.updateUnsavedIndicator();
            
            console.log('Changes saved');
        } catch (error) {
            console.error('Failed to save changes:', error);
        }
    }
    
    getEditorContent() {
        return this.editor ? this.editor.getValue() : '';
    }
    
    setEditorContent(content) {
        if (this.editor) {
            this.editor.setValue(content);
            this.lastSavedContent = content;
            this.hasUnsavedChanges = false;
            this.updateUnsavedIndicator();
        }
    }
    
    loadCurrentMachine() {
        this.currentMachine = this.stateManager.getCurrentMachine();
        
        if (this.currentMachine && this.currentMachine.statechart) {
            this.generateCodeForCurrentMachine();
        } else {
            this.setEditorContent(this.getDefaultMachineCode());
        }
    }
    
    async generateCodeForCurrentMachine() {
        if (!this.currentMachine || !this.currentMachine.statechart || !this.codeGenerator) {
            return;
        }
        
        try {
            this.showGenerationStatus('Generating code...');
            
            const startTime = performance.now();
            const result = await this.codeGenerator.generate(
                this.currentMachine.statechart, 
                this.currentFormat, 
                this.generationOptions
            );
            
            const generationTime = performance.now() - startTime;
            
            if (result.success) {
                this.setEditorContent(result.code);
                this.updateGenerationInfo(result, generationTime);
                this.updateEditorLanguage(this.currentFormat);
                this.updateFileInfo(result);
                this.updateDependenciesInfo(result.dependencies);
                
                if (result.validation) {
                    this.showValidationResult(result.validation);
                }
            } else {
                this.showGenerationError(result.error);
            }
        } catch (error) {
            console.error('Code generation failed:', error);
            this.showGenerationError(error.message);
        }
    }
    
    async regenerateCode() {
        this.generationCache.clear();
        await this.generateCodeForCurrentMachine();
    }
    
    onFormatChanged(newFormat) {
        if (newFormat !== this.currentFormat) {
            this.currentFormat = newFormat;
            this.generateCodeForCurrentMachine();
        }
    }
    
    updateEditorLanguage(format) {
        let language = 'javascript';
        
        switch (format) {
            case 'xstate-ts':
            case 'react-ts':
                language = 'typescript';
                break;
            case 'react':
            case 'react-jsx':
                language = 'javascript';
                break;
            case 'vue':
                language = 'html'; // Vue SFC
                break;
            case 'scxml':
                language = 'xml';
                break;
            case 'json':
                language = 'json';
                break;
            case 'yaml':
                language = 'yaml';
                break;
            case 'mermaid':
            case 'plantuml':
            case 'graphviz':
                language = 'plaintext';
                break;
        }
        
        if (this.editor && this.editor.getModel()) {
            monaco.editor.setModelLanguage(this.editor.getModel(), language);
        }
        
        // Update language info
        const languageInfo = this.container.querySelector('.language-info');
        if (languageInfo) {
            languageInfo.textContent = language.charAt(0).toUpperCase() + language.slice(1);
        }
    }
    
    showGenerationStatus(message) {
        const statusElement = this.container.querySelector('.generation-time');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }
    
    updateGenerationInfo(result, generationTime) {
        const timeElement = this.container.querySelector('.generation-time');
        const sizeElement = this.container.querySelector('.generation-size');
        const cacheElement = this.container.querySelector('.generation-cache');
        
        if (timeElement) {
            timeElement.textContent = `Generated in ${generationTime.toFixed(1)}ms`;
        }
        
        if (sizeElement) {
            const sizeKB = (result.code.length / 1024).toFixed(1);
            sizeElement.textContent = `${sizeKB} KB`;
        }
        
        if (cacheElement) {
            cacheElement.textContent = result.cached ? 'Cached' : 'Fresh';
        }
    }
    
    updateFileInfo(result) {
        const fileInfo = this.container.querySelector('.file-info');
        if (fileInfo && result.filename) {
            fileInfo.textContent = result.filename;
        }
    }
    
    updateDependenciesInfo(dependencies) {
        const depsInfo = this.container.querySelector('.dependencies-info');
        if (depsInfo && dependencies) {
            const depCount = Object.keys(dependencies).length;
            depsInfo.textContent = depCount > 0 ? `${depCount} dependencies` : '';
        }
    }
    
    showValidationResult(validation) {
        const indicator = this.container.querySelector('.validation-indicator');
        if (indicator) {
            if (validation.valid) {
                indicator.textContent = '✓ Valid';
                indicator.className = 'validation-indicator valid';
            } else {
                indicator.textContent = `✗ ${validation.errors.length} errors`;
                indicator.className = 'validation-indicator invalid';
            }
        }
    }
    
    showGenerationError(error) {
        const statusElement = this.container.querySelector('.generation-time');
        if (statusElement) {
            statusElement.textContent = `Error: ${error}`;
            statusElement.className = 'generation-time error';
        }
    }
    
    async downloadCode() {
        if (!this.currentMachine) return;
        
        try {
            const machineId = this.currentMachine.id;
            const response = await fetch(`/api/v1/machines/${machineId}/export/${this.currentFormat}?download=true`);
            
            if (response.ok) {
                const blob = await response.blob();
                const filename = response.headers.get('Content-Disposition')?.match(/filename="(.+)"/)?.[1] || 'code.txt';
                
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                this.showTemporaryMessage('File downloaded');
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            console.error('Download failed:', error);
            this.showTemporaryMessage('Download failed', 'error');
        }
    }
    
    showOptionsModal() {
        const modal = this.container.querySelector('#options-modal');
        if (modal) {
            modal.style.display = 'block';
        }
    }
    
    hideOptionsModal() {
        const modal = this.container.querySelector('#options-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    }
    
    applyOptions() {
        const form = this.container.querySelector('.options-form');
        if (!form) return;
        
        // Update generation options
        const formData = new FormData(form);
        for (const [key, value] of formData.entries()) {
            if (key === 'generationDebounceMs') {
                this.generationDebounceMs = parseInt(value, 10);
            } else {
                this.generationOptions[key] = value === 'on';
            }
        }
        
        // Update checkboxes (they don't appear in FormData if unchecked)
        const checkboxes = form.querySelectorAll('input[type="checkbox"]');
        for (const checkbox of checkboxes) {
            if (!formData.has(checkbox.name)) {
                this.generationOptions[checkbox.name] = false;
            }
        }
        
        this.hideOptionsModal();
        this.regenerateCode();
    }
    
    resetOptions() {
        this.generationOptions = {
            generateTypeDefinitions: true,
            includeComments: true,
            includeImports: true,
            formatCode: true,
            validateOutput: true
        };
        this.generationDebounceMs = 300;
        
        // Update form
        const form = this.container.querySelector('.options-form');
        if (form) {
            form.innerHTML = this.generateOptionsForm();
        }
    }
    
    onMachineChanged(machineId) {
        this.loadCurrentMachine();
    }
    
    generateMachineCode(statechart) {
        // Convert statechart to XState machine definition
        try {
            const machineConfig = this.convertStatechartToXState(statechart);
            return `import { createMachine, interpret } from 'xstate';

const machine = createMachine(${JSON.stringify(machineConfig, null, 2)});

// Create and start the service
const service = interpret(machine);
service.start();

export { machine, service };`;
        } catch (error) {
            console.warn('Failed to generate machine code:', error);
            return this.getDefaultMachineCode();
        }
    }
    
    convertStatechartToXState(statechart) {
        // Recursive conversion from statechart format to XState format
        const convertState = (state) => {
            const config = {};
            
            if (state.type === 'parallel') {
                config.type = 'parallel';
            }
            
            if (state.children && state.children.length > 0) {
                config.states = {};
                state.children.forEach(child => {
                    config.states[child.label] = convertState(child);
                });
                
                // Set initial state if not parallel
                if (state.type !== 'parallel' && state.children.length > 0) {
                    config.initial = state.children[0].label;
                }
            }
            
            return config;
        };
        
        const machine = {
            id: statechart.id || 'machine',
            initial: statechart.root_state.children?.[0]?.label || 'idle',
            context: {},
            states: {}
        };
        
        if (statechart.root_state.children) {
            statechart.root_state.children.forEach(state => {
                machine.states[state.label] = convertState(state);
            });
        }
        
        // Add transitions
        if (statechart.transitions) {
            statechart.transitions.forEach(transition => {
                // Add transition logic
                // TODO: Implement transition conversion
            });
        }
        
        return machine;
    }
    
    getDefaultMachineCode() {
        return `import { createMachine, interpret } from 'xstate';

const machine = createMachine({
  id: 'example',
  initial: 'idle',
  context: {
    count: 0
  },
  states: {
    idle: {
      on: {
        START: 'running'
      }
    },
    running: {
      on: {
        STOP: 'idle',
        PAUSE: 'paused'
      }
    },
    paused: {
      on: {
        RESUME: 'running',
        STOP: 'idle'
      }
    }
  }
});

// Create and start the service
const service = interpret(machine);
service.start();

export { machine, service };`;
    }
    
    showError(message) {
        this.container.innerHTML = `
            <div class="code-tab-error">
                <div class="error-icon">
                    <svg viewBox="0 0 24 24">
                        <path d="M13,14H11V10H13M13,18H11V16H13M1,21H23L12,2L1,21Z"/>
                    </svg>
                </div>
                <h3>Code Tab Error</h3>
                <p>${message}</p>
                <button class="retry-button" onclick="location.reload()">
                    Reload Application
                </button>
            </div>
        `;
    }
    
    // Public API
    
    onActivate() {
        if (this.editor) {
            this.editor.layout();
            this.editor.focus();
        }
    }
    
    onMachineChanged(machineId) {
        this.loadCurrentMachine();
    }
    
    onSelectionChanged(selection) {
        // Handle selection changes from other tabs
    }
    
    refresh() {
        this.loadCurrentMachine();
    }
    
    destroy() {
        if (this.editor) {
            this.editor.dispose();
        }
        
        if (this.changeTimeout) {
            clearTimeout(this.changeTimeout);
        }
        
        if (this.validationTimeout) {
            clearTimeout(this.validationTimeout);
        }
    }
}

export default CodeTab;