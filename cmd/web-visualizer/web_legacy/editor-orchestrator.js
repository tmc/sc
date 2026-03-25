/**
 * EditorOrchestrator - Central coordinator for the statechart editor
 *
 * Manages:
 * - Edit mode vs. view mode
 * - Tool selection and state
 * - Clipboard operations (copy, paste, cut)
 * - Multi-select coordination
 * - Keyboard shortcuts
 * - Integration between canvas, properties panel, and state manager
 */

class EditorOrchestrator extends EventTarget {
    constructor(options = {}) {
        super();

        // Dependencies (injected)
        this.stateManager = options.stateManager;
        this.commandManager = options.commandManager || new CommandManager();
        this.canvas = options.canvas;
        this.propertiesPanel = options.propertiesPanel;

        // Editor state
        this.mode = 'view'; // 'view' | 'edit' | 'simulation'
        this.currentTool = 'select'; // 'select' | 'pan' | 'zoom' | 'add-state' | 'add-transition'
        this.selection = new Set(); // Selected state/transition labels
        this.clipboard = null;
        this.isEditing = false;

        // Tool-specific state
        this.transitionDrawState = null; // { sourceLabel: string, tempLine: SVGElement }
        this.pendingState = null; // State being created

        // Configuration
        this.gridSize = options.gridSize || 20;
        this.snapToGrid = options.snapToGrid !== false;

        // Bind methods
        this.handleKeyDown = this.handleKeyDown.bind(this);
        this.handleToolChange = this.handleToolChange.bind(this);

        // Initialize
        this.setupKeyboardShortcuts();
        this.setupCommandManagerListeners();
    }

    // Mode Management
    setMode(mode) {
        if (this.mode === mode) return;

        const previousMode = this.mode;
        this.mode = mode;

        // Clean up any in-progress operations
        this.cancelCurrentOperation();

        // Update UI state
        this.dispatchEvent(new CustomEvent('modeChanged', {
            detail: { mode, previousMode }
        }));

        // Disable editing tools in view/simulation mode
        if (mode !== 'edit') {
            this.setTool('select');
        }
    }

    getMode() {
        return this.mode;
    }

    isEditMode() {
        return this.mode === 'edit';
    }

    // Tool Management
    setTool(tool) {
        if (this.currentTool === tool) return;

        const previousTool = this.currentTool;

        // Cancel any in-progress operation for the previous tool
        this.cancelCurrentOperation();

        this.currentTool = tool;

        this.dispatchEvent(new CustomEvent('toolChanged', {
            detail: { tool, previousTool }
        }));
    }

    getTool() {
        return this.currentTool;
    }

    handleToolChange(event) {
        const tool = event.target.closest('[data-tool]')?.dataset.tool;
        if (tool) {
            this.setTool(tool);
        }
    }

    cancelCurrentOperation() {
        // Cancel transition drawing
        if (this.transitionDrawState) {
            if (this.transitionDrawState.tempLine) {
                this.transitionDrawState.tempLine.remove();
            }
            this.transitionDrawState = null;
        }

        // Cancel pending state creation
        if (this.pendingState) {
            this.pendingState = null;
        }

        // Cancel any in-progress batch
        this.commandManager.cancelBatch();
    }

    // Selection Management
    select(label, addToSelection = false) {
        if (!addToSelection) {
            this.clearSelection();
        }
        this.selection.add(label);
        this.dispatchEvent(new CustomEvent('selectionChanged', {
            detail: { selection: Array.from(this.selection) }
        }));
    }

    deselect(label) {
        this.selection.delete(label);
        this.dispatchEvent(new CustomEvent('selectionChanged', {
            detail: { selection: Array.from(this.selection) }
        }));
    }

    toggleSelection(label) {
        if (this.selection.has(label)) {
            this.deselect(label);
        } else {
            this.select(label, true);
        }
    }

    clearSelection() {
        if (this.selection.size > 0) {
            this.selection.clear();
            this.dispatchEvent(new CustomEvent('selectionChanged', {
                detail: { selection: [] }
            }));
        }
    }

    getSelection() {
        return Array.from(this.selection);
    }

    isSelected(label) {
        return this.selection.has(label);
    }

    selectAll() {
        if (!this.stateManager) return;

        const machine = this.stateManager.getCurrentMachine();
        if (!machine?.statechart?.root_state) return;

        // Select all states
        const collectLabels = (state) => {
            const labels = [state.label];
            if (state.children) {
                state.children.forEach(child => {
                    labels.push(...collectLabels(child));
                });
            }
            return labels;
        };

        const allLabels = collectLabels(machine.statechart.root_state);
        allLabels.forEach(label => this.selection.add(label));

        this.dispatchEvent(new CustomEvent('selectionChanged', {
            detail: { selection: Array.from(this.selection) }
        }));
    }

    // Clipboard Operations
    copy() {
        if (this.selection.size === 0) return;

        const machine = this.stateManager?.getCurrentMachine();
        if (!machine?.statechart) return;

        // Collect selected states and their transitions
        const selectedLabels = Array.from(this.selection);
        const states = [];
        const transitions = [];

        const findState = (state, label) => {
            if (state.label === label) return state;
            if (state.children) {
                for (const child of state.children) {
                    const found = findState(child, label);
                    if (found) return found;
                }
            }
            return null;
        };

        selectedLabels.forEach(label => {
            const state = findState(machine.statechart.root_state, label);
            if (state) {
                states.push(JSON.parse(JSON.stringify(state)));
            }
        });

        // Include transitions between selected states
        if (machine.statechart.transitions) {
            machine.statechart.transitions.forEach(t => {
                const fromSelected = t.from.some(f => selectedLabels.includes(f));
                const toSelected = t.to.some(to => selectedLabels.includes(to));
                if (fromSelected && toSelected) {
                    transitions.push(JSON.parse(JSON.stringify(t)));
                }
            });
        }

        this.clipboard = {
            type: 'statechart-selection',
            states,
            transitions,
            timestamp: Date.now()
        };

        this.dispatchEvent(new CustomEvent('clipboard', {
            detail: { action: 'copy', count: states.length }
        }));
    }

    cut() {
        this.copy();
        this.deleteSelection();
    }

    paste(offset = { x: 20, y: 20 }) {
        if (!this.clipboard || this.clipboard.type !== 'statechart-selection') return;
        if (!this.isEditMode()) return;

        // Generate new unique labels for pasted states
        const labelMap = new Map();
        const newStates = this.clipboard.states.map(state => {
            const newLabel = this.generateUniqueLabel(state.label);
            labelMap.set(state.label, newLabel);
            return {
                ...state,
                label: newLabel,
                x: (state.x || 0) + offset.x,
                y: (state.y || 0) + offset.y
            };
        });

        // Update transition references
        const newTransitions = this.clipboard.transitions.map(t => ({
            ...t,
            label: this.generateUniqueLabel(t.label || `t_${Date.now()}`),
            from: t.from.map(f => labelMap.get(f) || f),
            to: t.to.map(to => labelMap.get(to) || to)
        }));

        // Execute as a batch command
        this.commandManager.beginBatch('Paste');

        newStates.forEach(state => {
            this.commandManager.execute(
                new AddStateCommand(state, '__root__', this.stateManager)
            );
        });

        newTransitions.forEach(t => {
            this.commandManager.execute(
                new AddTransitionCommand(t, this.stateManager)
            );
        });

        this.commandManager.endBatch();

        // Select pasted items
        this.clearSelection();
        newStates.forEach(state => this.select(state.label, true));

        this.dispatchEvent(new CustomEvent('clipboard', {
            detail: { action: 'paste', count: newStates.length }
        }));
    }

    generateUniqueLabel(baseLabel) {
        const machine = this.stateManager?.getCurrentMachine();
        if (!machine?.statechart) return baseLabel;

        const existingLabels = new Set();
        const collectLabels = (state) => {
            existingLabels.add(state.label);
            if (state.children) {
                state.children.forEach(collectLabels);
            }
        };
        collectLabels(machine.statechart.root_state);

        let label = baseLabel;
        let counter = 1;
        while (existingLabels.has(label)) {
            label = `${baseLabel}_${counter}`;
            counter++;
        }
        return label;
    }

    // Deletion
    deleteSelection() {
        if (this.selection.size === 0) return;
        if (!this.isEditMode()) return;

        const selectedLabels = Array.from(this.selection);

        this.commandManager.beginBatch('Delete selection');

        selectedLabels.forEach(label => {
            // Determine if it's a state or transition
            const isTransition = this.stateManager?.findTransition(label);
            if (isTransition) {
                this.commandManager.execute(
                    new RemoveTransitionCommand(label, this.stateManager)
                );
            } else {
                this.commandManager.execute(
                    new RemoveStateCommand(label, this.stateManager)
                );
            }
        });

        this.commandManager.endBatch();
        this.clearSelection();
    }

    // State Creation
    createState(position, options = {}) {
        if (!this.isEditMode()) return null;

        const label = this.generateUniqueLabel(options.label || 'State');
        const parentLabel = options.parentLabel || '__root__';

        const stateData = {
            label,
            type: options.type || 1, // BASIC by default
            x: this.snapToGrid ? this.snapPosition(position.x) : position.x,
            y: this.snapToGrid ? this.snapPosition(position.y) : position.y,
            is_initial: options.is_initial || false,
            is_final: options.is_final || false,
            children: []
        };

        this.commandManager.execute(
            new AddStateCommand(stateData, parentLabel, this.stateManager)
        );

        this.select(label);
        return stateData;
    }

    // Transition Creation
    startTransitionDraw(sourceLabel) {
        if (!this.isEditMode()) return;

        this.transitionDrawState = {
            sourceLabel,
            tempLine: null
        };

        this.dispatchEvent(new CustomEvent('transitionDrawStart', {
            detail: { sourceLabel }
        }));
    }

    completeTransitionDraw(targetLabel) {
        if (!this.transitionDrawState) return;

        const sourceLabel = this.transitionDrawState.sourceLabel;

        // Clean up temp line
        if (this.transitionDrawState.tempLine) {
            this.transitionDrawState.tempLine.remove();
        }
        this.transitionDrawState = null;

        // Create the transition
        const transitionData = {
            label: `${sourceLabel}_to_${targetLabel}`,
            from: [sourceLabel],
            to: [targetLabel],
            event: 'EVENT'
        };

        this.commandManager.execute(
            new AddTransitionCommand(transitionData, this.stateManager)
        );

        this.select(transitionData.label);

        this.dispatchEvent(new CustomEvent('transitionDrawComplete', {
            detail: { sourceLabel, targetLabel }
        }));
    }

    cancelTransitionDraw() {
        if (this.transitionDrawState) {
            if (this.transitionDrawState.tempLine) {
                this.transitionDrawState.tempLine.remove();
            }
            this.transitionDrawState = null;

            this.dispatchEvent(new CustomEvent('transitionDrawCancel'));
        }
    }

    isDrawingTransition() {
        return this.transitionDrawState !== null;
    }

    // Position Utilities
    snapPosition(value) {
        return Math.round(value / this.gridSize) * this.gridSize;
    }

    snapPositionPoint(point) {
        return {
            x: this.snapPosition(point.x),
            y: this.snapPosition(point.y)
        };
    }

    // Keyboard Shortcuts
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', this.handleKeyDown);
    }

    handleKeyDown(event) {
        // Don't handle shortcuts when typing in inputs
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return;
        }

        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        const ctrlOrCmd = isMac ? event.metaKey : event.ctrlKey;

        // Undo: Ctrl+Z / Cmd+Z
        if (ctrlOrCmd && !event.shiftKey && event.key === 'z') {
            event.preventDefault();
            this.commandManager.undo();
            return;
        }

        // Redo: Ctrl+Shift+Z / Cmd+Shift+Z or Ctrl+Y
        if ((ctrlOrCmd && event.shiftKey && event.key === 'z') ||
            (ctrlOrCmd && event.key === 'y')) {
            event.preventDefault();
            this.commandManager.redo();
            return;
        }

        // Copy: Ctrl+C / Cmd+C
        if (ctrlOrCmd && event.key === 'c') {
            event.preventDefault();
            this.copy();
            return;
        }

        // Cut: Ctrl+X / Cmd+X
        if (ctrlOrCmd && event.key === 'x') {
            event.preventDefault();
            this.cut();
            return;
        }

        // Paste: Ctrl+V / Cmd+V
        if (ctrlOrCmd && event.key === 'v') {
            event.preventDefault();
            this.paste();
            return;
        }

        // Select All: Ctrl+A / Cmd+A
        if (ctrlOrCmd && event.key === 'a') {
            event.preventDefault();
            this.selectAll();
            return;
        }

        // Delete: Delete or Backspace
        if (event.key === 'Delete' || event.key === 'Backspace') {
            event.preventDefault();
            this.deleteSelection();
            return;
        }

        // Escape: Cancel current operation or clear selection
        if (event.key === 'Escape') {
            event.preventDefault();
            if (this.isDrawingTransition()) {
                this.cancelTransitionDraw();
            } else {
                this.clearSelection();
            }
            return;
        }

        // Tool shortcuts (only in edit mode)
        if (this.isEditMode()) {
            switch (event.key) {
                case 'v':
                case 'V':
                    if (!ctrlOrCmd) {
                        this.setTool('select');
                    }
                    break;
                case 's':
                case 'S':
                    if (!ctrlOrCmd) {
                        this.setTool('add-state');
                    }
                    break;
                case 't':
                case 'T':
                    if (!ctrlOrCmd) {
                        this.setTool('add-transition');
                    }
                    break;
                case 'h':
                case 'H':
                    this.setTool('pan');
                    break;
            }
        }
    }

    setupCommandManagerListeners() {
        this.commandManager.addEventListener('historyChanged', () => {
            this.dispatchEvent(new CustomEvent('historyChanged', {
                detail: this.commandManager.getHistory()
            }));
        });

        this.commandManager.addEventListener('commandExecuted', (event) => {
            this.dispatchEvent(new CustomEvent('commandExecuted', {
                detail: event.detail
            }));
        });
    }

    // Cleanup
    destroy() {
        document.removeEventListener('keydown', this.handleKeyDown);
    }

    // State for UI
    getEditorState() {
        return {
            mode: this.mode,
            tool: this.currentTool,
            selection: Array.from(this.selection),
            canUndo: this.commandManager.canUndo(),
            canRedo: this.commandManager.canRedo(),
            clipboard: this.clipboard ? {
                count: this.clipboard.states?.length || 0,
                hasContent: true
            } : { hasContent: false },
            isDrawingTransition: this.isDrawingTransition()
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EditorOrchestrator };
}

// Make available globally for browser
if (typeof window !== 'undefined') {
    window.EditorOrchestrator = EditorOrchestrator;
}
