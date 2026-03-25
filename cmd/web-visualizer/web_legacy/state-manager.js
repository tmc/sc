/**
 * Centralized Application State Manager
 * 
 * Manages all application state including machines, UI state, selections,
 * and provides undo/redo functionality with event-driven updates.
 */
class StateManager extends EventTarget {
    constructor() {
        super();
        
        // Core application state
        this.state = {
            // Machine Management
            machines: new Map(),
            currentMachineId: null,
            
            // UI State
            activeTab: 'code',
            selectedStateId: null,
            selectedTransitionId: null,
            panelSizes: {
                left: 280,
                right: 320
            },
            
            // Editor State
            codeEditorContent: '',
            codeEditorCursor: { line: 0, column: 0 },
            
            // Context and Data
            contextData: {},
            testCases: [],
            
            // History and Events
            eventHistory: [],
            simulationRunning: false,
            
            // Connection State
            connected: false,
            lastError: null,
            
            // Search and Filters
            searchQuery: '',
            filters: {
                stateTypes: [],
                tags: []
            }
        };
        
        // Undo/Redo system
        this.history = {
            past: [],
            present: this.cloneState(),
            future: [],
            maxHistorySize: 50
        };
        
        // Subscriptions for cross-component communication
        this.subscriptions = new Map();
        
        // Auto-save timer
        this.autoSaveTimer = null;
        this.autoSaveDelay = 2000; // 2 seconds
        
        this.initializePersistedState();
    }
    
    /**
     * Get current state (read-only)
     */
    getState() {
        return { ...this.state };
    }
    
    /**
     * Get specific part of state
     */
    getStateSlice(path) {
        return this.getValueByPath(this.state, path);
    }
    
    /**
     * Update state with automatic history tracking
     */
    setState(updates, options = {}) {
        const { skipHistory = false, source = 'unknown' } = options;
        
        // Save current state to history before update
        if (!skipHistory) {
            this.saveToHistory();
        }
        
        // Apply updates
        const oldState = this.cloneState();
        this.state = { ...this.state, ...updates };
        
        // Emit change events
        this.emitStateChange(oldState, this.state, source);
        
        // Auto-save after delay
        this.scheduleAutoSave();
    }
    
    /**
     * Update nested state properties
     */
    setStateSlice(path, value, options = {}) {
        const updates = this.setValueByPath({}, path, value);
        this.setState(updates, options);
    }
    
    /**
     * Machine Management
     */
    addMachine(machine) {
        const machines = new Map(this.state.machines);
        machines.set(machine.id, machine);
        
        this.setState({
            machines,
            currentMachineId: machine.id
        }, { source: 'machine_added' });
    }
    
    updateMachine(machineId, updates) {
        const machines = new Map(this.state.machines);
        const machine = machines.get(machineId);
        
        if (machine) {
            machines.set(machineId, { ...machine, ...updates });
            this.setState({ machines }, { source: 'machine_updated' });
        }
    }
    
    removeMachine(machineId) {
        const machines = new Map(this.state.machines);
        machines.delete(machineId);
        
        const currentMachineId = this.state.currentMachineId === machineId 
            ? (machines.size > 0 ? machines.keys().next().value : null)
            : this.state.currentMachineId;
        
        this.setState({
            machines,
            currentMachineId
        }, { source: 'machine_removed' });
    }
    
    getCurrentMachine() {
        return this.state.machines.get(this.state.currentMachineId);
    }
    
    setCurrentMachine(machineId) {
        if (this.state.machines.has(machineId)) {
            this.setState({
                currentMachineId: machineId
            }, { source: 'machine_selected' });
        }
    }
    
    /**
     * Selection Management
     */
    selectState(stateId) {
        this.setState({
            selectedStateId: stateId,
            selectedTransitionId: null
        }, { source: 'state_selected' });
    }
    
    selectTransition(transitionId) {
        this.setState({
            selectedTransitionId: transitionId,
            selectedStateId: null
        }, { source: 'transition_selected' });
    }
    
    clearSelection() {
        this.setState({
            selectedStateId: null,
            selectedTransitionId: null
        }, { source: 'selection_cleared' });
    }
    
    /**
     * Tab Management
     */
    setActiveTab(tabId) {
        if (this.state.activeTab !== tabId) {
            this.setState({
                activeTab: tabId
            }, { source: 'tab_changed' });
        }
    }
    
    /**
     * Event History Management
     */
    addEventToHistory(event, result) {
        const eventHistory = [...this.state.eventHistory];
        eventHistory.unshift({
            id: `${Date.now()}-${Math.random()}`,
            event,
            result,
            timestamp: new Date().toISOString(),
            machineId: this.state.currentMachineId
        });
        
        // Limit history size
        if (eventHistory.length > 100) {
            eventHistory.splice(100);
        }
        
        this.setState({
            eventHistory
        }, { source: 'event_processed' });
    }
    
    clearEventHistory() {
        this.setState({
            eventHistory: []
        }, { source: 'history_cleared' });
    }
    
    /**
     * Context Data Management
     */
    updateContextData(contextData) {
        this.setState({
            contextData: { ...contextData }
        }, { source: 'context_updated' });
    }
    
    /**
     * Test Cases Management
     */
    addTestCase(testCase) {
        const testCases = [...this.state.testCases];
        testCases.push({
            id: `test-${Date.now()}`,
            ...testCase,
            createdAt: new Date().toISOString()
        });
        
        this.setState({
            testCases
        }, { source: 'test_case_added' });
    }
    
    updateTestCase(testId, updates) {
        const testCases = this.state.testCases.map(test =>
            test.id === testId ? { ...test, ...updates } : test
        );
        
        this.setState({
            testCases
        }, { source: 'test_case_updated' });
    }
    
    removeTestCase(testId) {
        const testCases = this.state.testCases.filter(test => test.id !== testId);
        
        this.setState({
            testCases
        }, { source: 'test_case_removed' });
    }
    
    /**
     * Connection State Management
     */
    setConnectionState(connected, error = null) {
        this.setState({
            connected,
            lastError: error
        }, { source: 'connection_changed' });
    }
    
    /**
     * Undo/Redo System
     */
    undo() {
        if (this.history.past.length === 0) return false;
        
        const previous = this.history.past[this.history.past.length - 1];
        const newPast = this.history.past.slice(0, this.history.past.length - 1);
        
        this.history = {
            past: newPast,
            present: previous,
            future: [this.history.present, ...this.history.future]
        };
        
        this.state = this.cloneState(previous);
        this.emitStateChange(this.history.present, previous, 'undo');
        
        return true;
    }
    
    redo() {
        if (this.history.future.length === 0) return false;
        
        const next = this.history.future[0];
        const newFuture = this.history.future.slice(1);
        
        this.history = {
            past: [...this.history.past, this.history.present],
            present: next,
            future: newFuture
        };
        
        this.state = this.cloneState(next);
        this.emitStateChange(this.history.present, next, 'redo');
        
        return true;
    }
    
    canUndo() {
        return this.history.past.length > 0;
    }
    
    canRedo() {
        return this.history.future.length > 0;
    }
    
    /**
     * Subscription System for Cross-Component Communication
     */
    subscribe(eventType, callback) {
        if (!this.subscriptions.has(eventType)) {
            this.subscriptions.set(eventType, new Set());
        }
        
        this.subscriptions.get(eventType).add(callback);
        
        // Return unsubscribe function
        return () => {
            const callbacks = this.subscriptions.get(eventType);
            if (callbacks) {
                callbacks.delete(callback);
            }
        };
    }
    
    /**
     * Emit state change events
     */
    emitStateChange(oldState, newState, source) {
        const event = new CustomEvent('stateChange', {
            detail: { oldState, newState, source }
        });
        
        this.dispatchEvent(event);
        
        // Emit specific change events
        this.emitSpecificChanges(oldState, newState, source);
    }
    
    emitSpecificChanges(oldState, newState, source) {
        // Machine changes
        if (oldState.machines !== newState.machines) {
            this.dispatchEvent(new CustomEvent('machinesChanged', {
                detail: { machines: newState.machines, source }
            }));
        }
        
        if (oldState.currentMachineId !== newState.currentMachineId) {
            this.dispatchEvent(new CustomEvent('currentMachineChanged', {
                detail: { machineId: newState.currentMachineId, source }
            }));
        }
        
        // Selection changes
        if (oldState.selectedStateId !== newState.selectedStateId ||
            oldState.selectedTransitionId !== newState.selectedTransitionId) {
            this.dispatchEvent(new CustomEvent('selectionChanged', {
                detail: {
                    stateId: newState.selectedStateId,
                    transitionId: newState.selectedTransitionId,
                    source
                }
            }));
        }
        
        // Tab changes
        if (oldState.activeTab !== newState.activeTab) {
            this.dispatchEvent(new CustomEvent('tabChanged', {
                detail: { activeTab: newState.activeTab, source }
            }));
        }
        
        // Connection changes
        if (oldState.connected !== newState.connected) {
            this.dispatchEvent(new CustomEvent('connectionChanged', {
                detail: { connected: newState.connected, source }
            }));
        }
    }
    
    /**
     * Persistence
     */
    initializePersistedState() {
        try {
            const saved = localStorage.getItem('statechart-visualizer-state');
            if (saved) {
                const parsed = JSON.parse(saved);
                // Only restore UI preferences, not machine data
                const { panelSizes, activeTab, filters } = parsed;
                if (panelSizes) this.state.panelSizes = panelSizes;
                if (activeTab) this.state.activeTab = activeTab;
                if (filters) this.state.filters = filters;
            }
        } catch (error) {
            console.warn('Failed to load persisted state:', error);
        }
    }
    
    scheduleAutoSave() {
        if (this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
        }
        
        this.autoSaveTimer = setTimeout(() => {
            this.saveToLocalStorage();
        }, this.autoSaveDelay);
    }
    
    saveToLocalStorage() {
        try {
            const stateToSave = {
                panelSizes: this.state.panelSizes,
                activeTab: this.state.activeTab,
                filters: this.state.filters
            };
            
            localStorage.setItem('statechart-visualizer-state', 
                JSON.stringify(stateToSave));
        } catch (error) {
            console.warn('Failed to save state to localStorage:', error);
        }
    }
    
    /**
     * History Management
     */
    saveToHistory() {
        const newPast = [...this.history.past, this.history.present];
        
        // Limit history size
        if (newPast.length > this.history.maxHistorySize) {
            newPast.shift();
        }
        
        this.history = {
            past: newPast,
            present: this.cloneState(),
            future: [] // Clear future when new action is performed
        };
    }
    
    /**
     * Utility Methods
     */
    cloneState(state = this.state) {
        return JSON.parse(JSON.stringify(state));
    }
    
    getValueByPath(obj, path) {
        return path.split('.').reduce((current, key) => 
            current && current[key] !== undefined ? current[key] : undefined, obj);
    }
    
    setValueByPath(obj, path, value) {
        const keys = path.split('.');
        const last = keys.pop();
        const target = keys.reduce((current, key) => {
            if (!current[key] || typeof current[key] !== 'object') {
                current[key] = {};
            }
            return current[key];
        }, obj);
        
        target[last] = value;
        return obj;
    }
    
    /**
     * Debug and Development
     */
    debug() {
        return {
            state: this.state,
            history: {
                pastLength: this.history.past.length,
                futureLength: this.history.future.length,
                canUndo: this.canUndo(),
                canRedo: this.canRedo()
            },
            subscriptions: Array.from(this.subscriptions.keys())
        };
    }
    
    reset() {
        this.state = {
            machines: new Map(),
            currentMachineId: null,
            activeTab: 'code',
            selectedStateId: null,
            selectedTransitionId: null,
            panelSizes: { left: 280, right: 320 },
            codeEditorContent: '',
            codeEditorCursor: { line: 0, column: 0 },
            contextData: {},
            testCases: [],
            eventHistory: [],
            simulationRunning: false,
            connected: false,
            lastError: null,
            searchQuery: '',
            filters: { stateTypes: [], tags: [] }
        };
        
        this.history = {
            past: [],
            present: this.cloneState(),
            future: []
        };
        
        this.emitStateChange({}, this.state, 'reset');
    }
}

// Export as singleton
const stateManager = new StateManager();
export default stateManager;